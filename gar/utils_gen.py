import json
import logging
import os
from pathlib import Path
import pickle
import random
import subprocess
import time
from multiprocessing import get_context, Pool
from time import sleep
from datetime import datetime
from typing import *

import pytorch_lightning as pl
import rich
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import transformers

# from transformers.tokenization_utils import trim_batch


def _handle_paths(obj: Any) -> str:
    assert isinstance(obj, Path)
    return str(obj)


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return _handle_paths(obj)
    elif isinstance(obj, pl.Callback):
        return str(obj)
    else:
        raise ValueError(type(obj).mro())


def pickle_load(path):
    with open(path, "rb") as fin:
        return pickle.load(fin)


def pickle_dump(obj, path):
    with open(path, "wb") as fout:
        pickle.dump(obj, fout)


def json_load(path, **kwargs):
    with open(path) as fin:
        return json.load(fin, **kwargs)


def json_dump(obj, path, **kwargs):
    with open(path, "w") as fout:
        return json.dump(obj, fout, **kwargs)


def get_local_rank():
    return os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]


def get_global_rank():
    return os.environ["OMPI_COMM_WORLD_NODE_RANK"]


def trim_batch(
        input_ids,
        pad_token_id,
        attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""

    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


def chunks(l, n):
    n = len(l) // n
    n = max(1, n)
    return (l[i:i + n] for i in range(0, len(l), n))


def multi_runs(f, para, f_combine=None, n=10):
    with get_context("spawn").Pool(n) as pool:
        # with Pool(n) as pool:
        res = pool.map(f, para)
        if f_combine is not None:
            res = f_combine(res)
        return res


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def encode_file(
    tokenizer,
    data_path,
    max_length,
    pad_to_max_length=True,
    return_tensors="pt",
):
    assert pad_to_max_length == True, f"{pad_to_max_length = }"
    assert return_tensors == "pt", f"{return_tensors = }"
    assert max_length > 1, f"{max_length = }"
    rich.print(f"Tokenizer: {tokenizer}")
    
    examples = []
    rich.print(f"[red purple]{max_length = }")
    import collections
    counter = collections.Counter()

    with open(data_path, encoding='utf8') as f:
        lens = []
        for text in tqdm(f.readlines()):
            tokenized = tokenizer.batch_encode_plus(
                [text],
                max_length=max_length,
                truncation=True,
                padding="max_length" if pad_to_max_length else False,
                return_tensors=return_tensors,
            )

            examples.append(tokenized)
            lens.append(tokenized["input_ids"].shape[1])
            
    counter.update(lens)
    rich.print(f"[bold purple]{counter = }")
    return examples


# workaround to pickle tokenization results
# https://github.com/huggingface/transformers/issues/4327
from transformers.tokenization_utils import BatchEncoding


def red(self):
    return BatchEncoding, (self.data,)


BatchEncoding.__reduce__ = red


class SummarizationDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            data_dir,
            type_path,
            max_source_length,
            max_target_length,
    ):
        super().__init__()
        data_dir = Path(data_dir)

        rich.print(f"[red bold]{data_dir = }")
        assert "bart" in str(tokenizer)
        self.tokenizer = tokenizer
        self.type_path = type_path
        if 'bart' in str(tokenizer):
            suffix = ''
        elif 't5' in str(tokenizer):
            suffix = '.t5'
        else:
            raise NotImplementedError


        pickled_source_path = data_dir / f"{type_path}.source.processed{suffix}"
        pickled_target_path = data_dir / f"{type_path}.target.processed{suffix}"
        text_source_path = data_dir / f"{type_path}.source"
        text_target_path = data_dir / f"{type_path}.target"

        # if pickled_source_path.exists() and pickled_target_path.exists():
        #     print(
        #         f"loading from {pickled_target_path} (pkl)... "
        #         f"make sure data is what you need"
        #     )
        #     self.source = pickle_load(pickled_source_path)
        #     self.target = pickle_load(pickled_target_path)
        # else:
        #     self.source = encode_file(tokenizer, text_source_path, max_source_length)
        #     self.target = encode_file(tokenizer, text_target_path, max_target_length)
        #     pickle_dump(self.source, pickled_source_path)
        #     pickle_dump(self.target, pickled_target_path)
        
        self.source = encode_file(tokenizer, text_source_path, max_source_length)
        self.target = encode_file(tokenizer, text_target_path, max_target_length)
        self.all_answers = None
        target_json = data_dir / f"{type_path}.target.json"
        if target_json.exists():
            self.all_answers = json_load(target_json)
            self.kw_labels_cache = {}


    def __len__(self):
        return len(self.source)

    def create_kw_labels(self, answers, target_ids):
        kw_labels = torch.zeros(target_ids.shape).type_as(target_ids)
        for a in answers:
            a_tokens = self.tokenizer.encode(
                a,
                add_special_tokens=False,
                return_tensors="pt",
            )[0]
            a_len = a_tokens.shape[0]
            target_len = target_ids.shape[0]
            for idx in range(target_len - a_len):
                if torch.all(target_ids[idx: idx + a_len] == a_tokens):
                    kw_labels[idx: idx + a_len] = 1
        return kw_labels

    def select_psg(self, src):
        q_ids, ctx_ids_l = src
        if self.type_path == 'train':
            if random.random() <= 1:
                top_k = random.randrange(0, 11)
            else:
                top_k = 10
            selected_psg = (
                list(range(top_k)) + 
                random.choices(list(range(top_k, len(ctx_ids_l))), k=10 - top_k)
            )
        else:
            selected_psg = range(10)
        source_ids = (
            [self.tokenizer.bos_token_id] + q_ids + [self.tokenizer.eos_token_id]
        )
        for idx in selected_psg:
            title_ids, text_ids = ctx_ids_l[idx]
            source_ids.extend(
                title_ids + [self.tokenizer.eos_token_id] 
                + text_ids + [self.tokenizer.eos_token_id]
            )
        source_ids = torch.LongTensor(source_ids[:1024])
        return source_ids

    def __getitem__(self, index):
        source_ids = self.source[index]["input_ids"].squeeze()
        target_ids = self.target[index]["input_ids"].squeeze()
        src_mask = self.source[index]["attention_mask"].squeeze()
        self.shuffle = False  # NB. whether shuffle input psg or not
        if self.shuffle and self.type_path == 'train':
            # no change to attention_mask since they are all 1s for top-10 psg
            idx_l = []
            last_idx = -1
            for ct, i in enumerate(torch.where(source_ids == 2)[0]):
                # since title is split by 2 too
                if ct % 2 == 0:
                    idx_l.append((last_idx + 1, i.item()))
                    last_idx = i.item()
            if last_idx != 1023:
                idx_l.append((last_idx + 1, 1023))
            psg_idx_l = idx_l[1:]
            random.shuffle(psg_idx_l)
            idx_l = idx_l[:1] + psg_idx_l
            new_source_ids = []
            for start, end in idx_l:
                new_source_ids.append(source_ids[start: end + 1])
            source_ids = torch.cat(new_source_ids)

        # whether add kw_labels (mark answer spans) when generating psg [not used]
        kw_labels = None
        # if self.all_answers is not None:
        #     if index not in self.kw_labels_cache:
        #         answers = self.all_answers[index]
        #         self.kw_labels_cache[index] = self.create_kw_labels(answers, target_ids)
        #     kw_labels = self.kw_labels_cache[index]

        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "kw_labels": kw_labels
        }

    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id):
        y = trim_batch(batch["target_ids"], pad_token_id)
        source_ids, source_mask = trim_batch(
            batch["source_ids"], pad_token_id, attention_mask=batch["source_mask"]
        )
        return source_ids, source_mask, y

    def collate_fn(self, batch):
        assert isinstance(batch, list), type(batch).mro()
        input_ids = torch.stack([x["source_ids"] for x in batch])
        masks = torch.stack([x["source_mask"] for x in batch])
        target_ids = torch.stack([x["target_ids"] for x in batch])
        pad_token_id = self.tokenizer.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        if batch[0]['kw_labels'] is not None:
            kw_labels = torch.stack([x["kw_labels"] for x in batch])
            kw_labels = kw_labels[:, :y.shape[1]]
            return {
                "source_ids": source_ids,
                "source_mask": source_mask,
                "target_ids": y,
                "kw_labels": kw_labels,
            }
        
        return {
            "source_ids": source_ids,
            "source_mask": source_mask,
            "target_ids": y,
        }


def freeze_params(model, except_para=None):
    if type(model) == dict:
        for name, par in model.items():
            if except_para is not None and except_para in name:
                par.requires_grad = True
            else:
                par.requires_grad = False
    else:
        for name, par in model.named_parameters():
            if except_para is not None and except_para in name:
                par.requires_grad = True
            else:
                par.requires_grad = False


def unfreeze_params(model, except_para=None):
    if type(model) == dict:
        for name, par in model.items():
            if except_para is not None and except_para in name:
                par.requires_grad = False
            else:
                par.requires_grad = True
    else:
        for name, par in model.named_parameters():
            if except_para is not None and except_para in name:
                par.requires_grad = False
            else:
                par.requires_grad = True
