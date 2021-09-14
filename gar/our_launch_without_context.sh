GEN_TARGET='answer' python train_generator.py          \
--remark generator_train_nq_A \
--output_dir /home/mila/g/gagnonju/GAR/gar/outputs/data_without_context/ \
--train_batch_size 128 \
--eval_batch_size 256 \
--ckpt_metric val-ROUGE-1 \
--data_dir /home/mila/g/gagnonju/GAR/data/nq-answer/ \
--learning_rate 5e-6
