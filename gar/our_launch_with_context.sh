#!/usr/bin/env bash

######################################################################
# Boilerplate to get the directory of the script.
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do 
  # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" 
done
SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
######################################################################

if [[ -z $1 ]] ; then
	echo "'\$1' Should be the output filename. Got nothing."
	return 1
fi
if [[ -z $2 ]] ; then
	echo "'\$2' Should be the type of inputs, one of 'sentence', 'title' or 'anser'. Got nothing."
	return 1
fi
if [[ "$2" != "sentence" && "$2" != "title" && "$2" != "answer" ]] ; then
  echo "'\$2' Should be the type of inputs, one of 'sentence', 'title' or 'anser'. Got $2"
  return 1
fi
if [[ "$3" != "horovod" && "$3" != "nccl" && "$3" != "gloo" ]] ; then
  echo "'\$3' Should be the type backend, one of 'horovod', 'nccl' and 'gloo'. Got '$3'."
  return 1
fi

ROOT_DIR="$(realpath "$SCRIPT_DIR/../../")"
DATA_DIR="$ROOT_DIR/outputs/${2}_data_with_context/"
echo "\$DATA_DIR: $DATA_DIR"


# This is the output dir for the generator training script
OUTPUT_DIR="$SCRIPT_DIR/outputs/$1"
if [[ ! -d "$OUTPUT_DIR" ]] ; then
  mkdir "$OUTPUT_DIR" || true
fi
echo "\$OUTPUT_DIR: $OUTPUT_DIR"

GEN_TARGET=$2 python "$SCRIPT_DIR"/train_generator.py \
--remark generator_train_nq_A \
--output_dir "$OUTPUT_DIR" \
--train_batch_size 8 \
--eval_batch_size 16 \
--ckpt_metric val-ROUGE-1 \
--max_source_length 768 \
--max_target_length 40 \
--num_train_epochs 150 \
--data_dir "$DATA_DIR" \
--learning_rate 5e-6 \
--backend "$3"
