SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do 
  # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" 
  # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

if [[ -z $1 ]] ; then
	echo "'$1' is empty."
	return 1
fi

TARGET="$DIR/outputs/$1"
if [[ ! -d $TARGET ]] ; then
	mkdir $TARGET
fi

GEN_TARGET='answer' python "$DIR"/train_generator.py \
--remark generator_train_nq_A \
--output_dir "$TARGET" \
--train_batch_size 8 \
--eval_batch_size 16 \
--ckpt_metric val-ROUGE-1 \
--max_source_length 768 \
--max_target_length 40 \
--num_train_epochs 150 \
--data_dir /home/mila/g/gagnonju/IteratedDecoding/outputs/data_with_context/ \
--learning_rate 5e-6
