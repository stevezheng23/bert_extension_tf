for i in "$@"
  do
    case $i in
      -g=*|--gpudevice=*)
      GPUDEVICE="${i#*=}"
      shift
      ;;
      -t=*|--taskname=*)
      TASKNAME="${i#*=}"
      shift
      ;;
      -r=*|--randomseed=*)
      RANDOMSEED="${i#*=}"
      shift
      ;;
      -p=*|--predicttag=*)
      PREDICTTAG="${i#*=}"
      shift
      ;;
      -m=*|--modeldir=*)
      MODELDIR="${i#*=}"
      shift
      ;;
      -d=*|--datadir=*)
      DATADIR="${i#*=}"
      shift
      ;;
      -o=*|--outputdir=*)
      OUTPUTDIR="${i#*=}"
      shift
      ;;
      --maxlen=*)
      MAXLEN="${i#*=}"
      shift
      ;;
      --batchsize=*)
      BATCHSIZE="${i#*=}"
      shift
      ;;
      --learningrate=*)
      LEARNINGRATE="${i#*=}"
      shift
      ;;
      --trainepochs=*)
      TRAINEPOCHS="${i#*=}"
      shift
      ;;
    esac
  done

echo "gpu device     = ${GPUDEVICE}"
echo "task name      = ${TASKNAME}"
echo "random seed    = ${RANDOMSEED}"
echo "predict tag    = ${PREDICTTAG}"
echo "model dir      = ${MODELDIR}"
echo "data dir       = ${DATADIR}"
echo "output dir     = ${OUTPUTDIR}"
echo "max len        = ${MAXLEN}"
echo "batch size     = ${BATCHSIZE}"
echo "learning rate  = ${LEARNINGRATE}"
echo "train epochs   = ${TRAINEPOCHS}"

alias python=python3

CUDA_VISIBLE_DEVICES=${GPUDEVICE} python run_ner.py \
--bert_config_file=${MODELDIR}/bert_config.json \
--init_checkpoint=${MODELDIR}/bert_model.ckpt \
--vocab_file=${MODELDIR}/vocab.txt \
--task_name=${TASKNAME} \
--random_seed=${RANDOMSEED} \
--predict_tag=${PREDICTTAG} \
--do_lower_case=false \
--data_dir=${DATADIR}/ \
--output_dir=${OUTPUTDIR}/debug \
--export_dir=${OUTPUTDIR}/export \
--max_seq_length=${MAXLEN} \
--train_batch_size=${BATCHSIZE} \
--learning_rate=${LEARNINGRATE} \
--num_train_epochs=${TRAINEPOCHS} \
--do_train=true \
--do_eval=false \
--do_predict=true \
--do_export=false

python tool/convert_token.py \
--input_file=${OUTPUTDIR}/debug/predict.${PREDICTTAG}.json \
--output_file=${OUTPUTDIR}/debug/predict.${PREDICTTAG}.txt

python tool/eval_token.py \
< ${OUTPUTDIR}/debug/predict.${PREDICTTAG}.txt \
> ${OUTPUTDIR}/debug/predict.${PREDICTTAG}.token

read -n 1 -s -r -p "Press any key to continue..."