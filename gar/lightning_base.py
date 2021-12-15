import argparse
import json
import logging
import os
import random
from pathlib import Path

import colored_traceback.auto
import numpy as np
import pytorch_lightning as pl
# from pytorch_lightning.callbacks import LearningRateLogger
import rich
import torch

from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelWithLMHead,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

import utils_gen

logger = logging.getLogger(__name__)

MODEL_MODES = {
    "base": AutoModel,
    "sequence-classification": AutoModelForSequenceClassification,
    "question-answering": AutoModelForQuestionAnswering,
    "pretraining": AutoModelForPreTraining,
    "token-classification": AutoModelForTokenClassification,
    "language-modeling": AutoModelWithLMHead,
}


def set_seed(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpus > 0:
        torch.cuda.manual_seed_all(args.seed)


class BaseTransformer(pl.LightningModule):
    def __init__(
        self, 
        hparams: argparse.Namespace, 
        num_labels=None, 
        mode="base", 
        **config_kwargs,
        ):
        "Initialize a model."

        super().__init__()

        for k, v in vars(hparams).items():
            setattr(self.hparams, k, v)

        cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None
        self.config = AutoConfig.from_pretrained(
            self.hparams.config_name
            if self.hparams.config_name
            else self.hparams.model_name_or_path,
            **({"num_labels": num_labels} if num_labels is not None else {}),
            cache_dir=cache_dir,
            **config_kwargs,
        )

        extra_model_params = (
            "encoder_layerdrop",
            "decoder_layerdrop",
            "dropout",
            "attention_dropout",
        )
        for p in extra_model_params:
            if getattr(self.hparams, p, None):
                assert hasattr(self.config, p), (
                    f"model config doesn't have a `{p}` attribute"
                )
                setattr(self.config, p, getattr(self.hparams, p))

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.tokenizer_name
            if self.hparams.tokenizer_name
            else self.hparams.model_name_or_path,
            cache_dir=cache_dir,
        )
        
        # if 'gpt2' in self.hparams.model_name_or_path:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = MODEL_MODES[mode].from_pretrained(
            self.hparams.model_name_or_path,
            from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
            config=self.config,
            cache_dir=cache_dir,
        )

    def is_logger(self):
        return self.trainer.proc_rank <= 0

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() 
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() 
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, 
            lr=self.hparams.learning_rate, 
            eps=self.hparams.adam_epsilon,
            )
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(
        self, 
        epoch, 
        batch_idx, 
        optimizer, 
        optimizer_idx, 
        optimizer_closure=None,
        on_tpu=None,
        using_native_amp=False,
        using_lbfgs=False,
    ):

        if on_tpu:
            # `or on_tpu` was just added and not debugged.
            assert False 
            xm.optimizer_step(optimizer)
        else:
            optimizer.step(
                closure=optimizer_closure
                )

        optimizer.zero_grad()

        # By default, PL will only step every epoch.
        self.lr_scheduler.step()  
        lrs = {
            f"lr_group_{i}": 
            lr for i, lr 
            in enumerate(self.lr_scheduler.get_lr())
        }
        self.logger.log_metrics(lrs)

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_end(self, outputs):
        return self.validation_end(outputs)

    def train_dataloader(self):
        train_batch_size = self.hparams.train_batch_size
        dataloader = self.load_dataset(
            "train", 
            train_batch_size,
        )

        t_total = (
                (len(dataloader.dataset) // (train_batch_size * max(1, self.hparams.gpus)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, 
            num_warmup_steps=self.hparams.warmup_steps, 
            num_training_steps=t_total,
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        return self.load_dataset(
            "dev", 
            self.hparams.eval_batch_size,
        )

    def test_dataloader(self):
        return self.load_dataset(
            "test", 
            self.hparams.eval_batch_size,
        )

    def _feature_file(self, mode):
        return os.path.join(
            self.hparams.data_dir,
            "cached_{}_{}_{}".format(
                mode,
                list(filter(
                    None, 
                    self.hparams.model_name_or_path.split("/"))).pop(),
                str(self.hparams.max_seq_length),
            ),
        )


class LoggingCallback(pl.Callback):
    def on_validation_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule,
    ):
        logger.info("***** Validation results *****")
        metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                logger.info("{} = {}\n".format(
                    key, str(metrics[key]))
                )

    def on_test_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule,
    ):
        assert False
        logger.info("***** Test results *****")
        metrics = trainer.callback_metrics
        # Log and save results to file
        output_test_results_file = os.path.join(
            pl_module.hparams.output_dir, 
            "test_results_node_rank_{trainer.node_rank}_local_rank_{trainer.local_rank}.txt",
        )
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(
                        key, str(metrics[key]))
                    )
                    writer.write("{} = {}\n".format(
                        key, str(metrics[key]))
                    )

def generic_train(
    model: BaseTransformer, 
    args: argparse.Namespace, 
    logger=True, 
    resume_cp_file=None,
):
    # init model
    set_seed(args)
    odir = Path(model.hparams.output_dir)
    odir.mkdir(exist_ok=True)

    assert isinstance(args.save_top_k, int), type(args.save_top_k)
    assert args.save_top_k != 0, args.save_top_k
    # assert args.ckpt_metric, args.ckpt_metric
    # assert args.ckpt_mode, args.ckpt_mode
    assert args.output_dir, args.output_dir
    assert Path(args.output_dir).exists, args.output_dir
    
    # if (
    #     "rouge" in args.ckpt_metric.lower() 
    #     or "bleu" in args.ckpt_metric.lower()
    # ):
    #     assert args.ckpt_mode == "max", args.ckpt_mode

    # if (
    #     "ppl" in args.ckpt_metric.lower() or
    #     "loss" in args.ckpt_metric.lower()
    # ):
    #     assert args.ckpt_mode == "min", args.ckpt_mode

    # assert not args.ckpt_metric, args.ckpt_metric
    # assert not args.ckpt_mode, args.ckpt_mode
    assert args.save_top_k == -1, args.save_top_k

    checkpoint_params = dict(
        dirpath=args.output_dir, 
        # monitor=args.ckpt_metric, 
        # mode=args.ckpt_mode,
        # auto_insert_metric_name=True,
        save_top_k=args.save_top_k, 
        save_last=True,
        every_n_epochs=3,
        save_on_train_epoch_end=True, 
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        **checkpoint_params
    )

    train_params = {}
    if args.gpus > 1 or args.n_nodes > 1:
        if args.backend != "horovod":
            train_params["accelerator"] = "ddp"
        else:
            train_params["accelerator"] = "horovod"

    # lr_logger = LearningRateLogger()
    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        max_epochs=args.num_train_epochs,
        gradient_clip_val=args.max_grad_norm,
        callbacks=[LoggingCallback(), checkpoint_callback],
        resume_from_checkpoint=resume_cp_file,
        logger=logger,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        **train_params,

        # overfit_batches=2,
        # fast_dev_run=True,
    )

    if args.backend != "horovod":
        train_params["gpus"] = args.gpus
        train_params["num_nodes"] = args.n_nodes

    else:
        train_params["gpus"] = 1

    if args.fp16:
        train_params["precision"] = 16
        train_params["amp_level"] = args.fp16_opt_level

    if args.n_tpu_cores > 0:
        assert False
        global xm
        import torch_xla.core.xla_model as xm

        train_params["num_tpu_cores"] = args.n_tpu_cores
        train_params["gpus"] = 0

    rich.print(f"[green bold]Num GPUs: {train_params['gpus']}")
    rich.print(f"[green bold]Accelerator: {train_params['accelerator']}")

    trainer = pl.Trainer(**train_params)
    
    if trainer.is_global_zero:
        rich.print("[bold]train_params:", train_params)
        rich.print("[bold]checkpoint_params:", checkpoint_params)
        
        utils_gen.json_dump(
            vars(args), 
            Path(args.output_dir) / "real_args.json",
            default=utils_gen.json_default,
        )

        utils_gen.json_dump(
            train_params, 
            Path(args.output_dir) / "train_params.json",
            default=utils_gen.json_default,
        )

        utils_gen.json_dump(
            checkpoint_params, 
            Path(args.output_dir) / "checkpoint_params.json",
            default=utils_gen.json_default,
        )

    if args.do_train:
        trainer.fit(model)

    return trainer
