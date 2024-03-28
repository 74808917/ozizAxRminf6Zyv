#!/bin/bash


help()
{
    echo ". run.training.mgpu.sh"
    echo "      --modeldata_dir              a_created_directory_for_saving_model_data"
    echo "      --dataset_dir                your_processed_dataset_directory"
    echo "      --datasource                 [cnndm|xsum]"
    echo "      --nlp_parsing_ver            [.]"
    echo "      --data_build_type            [struct|model_tokenization/bart-large-cnn|model_tokenization/bart-large-xsum]"
    echo "      --pretrained_model           [facebook/bart-large-cnn|facebook/bart-large-xsum]"
    echo "      --split_type                 [\"train\",\"validation\"/\"test\"]"
    echo "      --pair_type                  [\"document\",\"summary\"]|[\"article\",\"highlights\"]"
    echo "      --data_file_suffix           [.dataset.json|.json]"
    echo "      --struct_feature_level       [none|struct.ner]"
    echo "      --query_model_size           [true|false]"
    echo "      --gpu_ids                    [0,1,2]"
    echo "      --process_port_id            [9999]"
    echo "      --modeling_choice            [model_ner]"
    echo "      --dropout_rate               [none|{activation_dropout:0.1,classif_dropout:0.1}]"
    echo "      --config_folder              [bart]"
    echo "      --model_dir                  [none|your_model_folder_trained_on_dataset (cnndm or xsum)]"
    echo "      --glob_file_pattern          [none|a_checkpoint_folder]"
    echo "      --pyvenv_dir                 your_python_virtual_environment_dir_parent_to_bin_folder"
}

NUM_ARGUMENTS=$#
EXPECTED_N_ARGS=38
if [ "$NUM_ARGUMENTS" -ne ${EXPECTED_N_ARGS} ]; then
    help
    return
fi

while :
do
  case "$1" in
    --modeldata_dir )
      MODELDATA_DIR="$2"
      shift 2
      ;;
    --dataset_dir )
      DATASET_DIR="$2"
      shift 2
      ;;
    --datasource )
      DATASOURCE="$2"
      shift 2
      ;;
    --nlp_parsing_ver )
      NLP_PARSING_VER="$2"
      shift 2
      ;;
    --data_build_type )
      DATASET_BUILD_TYPE="$2"
      shift 2
      ;;
    --pretrained_model )
      PRETRAINED_MODEL_TYPE="$2"
      shift 2
      ;;
    --split_type )
      DATA_SPLIT_TYPE="$2"
      shift 2
      ;;
    --pair_type )
      DATA_PAIR_TYPE="$2"
      shift 2
      ;;
    --data_file_suffix )
      DATA_FILE_SUFFIX="$2"
      shift 2
      ;;
    --struct_feature_level )
      STRUCT_FEATURE_LEVEL="$2"
      shift 2
      ;;
    --query_model_size )
      QUERY_MODEL_SIZE="$2"
      shift 2
      ;;
    --gpu_ids )
      GPU_IDS="$2"
      shift 2
      ;;
    --process_port_id )
      PROCESS_PORT_ID="$2"
      shift 2
      ;;
    --modeling_choice )
      MODEL_CHOICE="$2"
      shift 2
      ;;
    --dropout_rate )
      DROPOUT_RATE="$2"
      shift 2
      ;;
    --config_folder )
      CONFIG_FOLDER="$2"
      shift 2
      ;;
    --model_dir )
      MODEL_DIR="$2"
      shift 2
      ;;
    --glob_file_pattern )
      GLOB_FILE_PATTERN="$2"
      shift 2
      ;;
    --pyvenv_dir )
      PYVENV_DIR="$2"
      shift 2
      ;;
    --)
      shift;
      break
      ;;
    *)
      # echo "Unexpected option: $1"
      # help
      break
      ;;
  esac
done


source ${PYVENV_DIR}/bin/activate
export PYTHONPATH="$PYTHONPATH:$PWD:$PWD/.."
# export PATH=/usr/local/cuda-11.3/bin:$PATH
# export CPATH=/usr/local/cuda-11.3/include:$CPATH
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.3/lib64
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO


FOLDER_NAME="`basename $PWD`"

QUERY_MODEL_SIZE=[ ${QUERY_MODEL_SIZE} == "true" ]

DATASET_ROOT=${DATASET_DIR}/${DATASOURCE}/${NLP_PARSING_VER}

RUN_TRACE_DIR="${MODELDATA_DIR}/${FOLDER_NAME}/run_trace"
[ -d ${RUN_TRACE_DIR} ] || mkdir -p ${RUN_TRACE_DIR}

today=`date '+%Y_%m_%d_%H_%M'`;
RUN_LOG="${RUN_TRACE_DIR}/${DATASOURCE}_train_results_$today.out"

echo ${RUN_LOG}
echo $HOSTNAME >${RUN_LOG}

echo "--modeldata_root:             ${MODELDATA_DIR}/${FOLDER_NAME}"
echo "--dataset_root:               ${DATASET_ROOT}"
echo "--config_folder:              ${CONFIG_FOLDER}/${DATASOURCE}"
echo "--dataset_folder:             ${DATASET_BUILD_TYPE}"
echo "--base_model_pretrained_name: ${PRETRAINED_MODEL_TYPE}"
echo "--tokenizer_name:             ${PRETRAINED_MODEL_TYPE}"
echo "--split_type:                 ${DATA_SPLIT_TYPE}"
echo "--pair_type:                  ${DATA_PAIR_TYPE}"
echo "--dataset_file:               {split_type}.{pair_type}${DATA_FILE_SUFFIX}"
echo "--struct_feature_level:       ${STRUCT_FEATURE_LEVEL}"
echo "--modeling_choice             ${MODEL_CHOICE}"
echo "--dropout_rate                ${DROPOUT_RATE}"
echo "--model_dir                   ${MODEL_DIR}"
echo "--glob_file_pattern           ${GLOB_FILE_PATTERN}"


#if [ 1 -eq 0 ]; then
PARAMS="--modeldata_root ${MODELDATA_DIR}/${FOLDER_NAME}  "
PARAMS=${PARAMS}"--dataset_root ${DATASET_ROOT}  "
PARAMS=${PARAMS}"--config_folder ${CONFIG_FOLDER}/${DATASOURCE}  "
PARAMS=${PARAMS}"--dataset_folder ${DATASET_BUILD_TYPE}  "
PARAMS=${PARAMS}"--base_model_pretrained_name ${PRETRAINED_MODEL_TYPE}  "
PARAMS=${PARAMS}"--tokenizer_name ${PRETRAINED_MODEL_TYPE}  "
PARAMS=${PARAMS}"--use_slow_tokenizer  "
PARAMS=${PARAMS}"--split_type ${DATA_SPLIT_TYPE}  "
PARAMS=${PARAMS}"--pair_type ${DATA_PAIR_TYPE}  "
PARAMS=${PARAMS}"--dataset_file {split_type}.{pair_type}${DATA_FILE_SUFFIX}  "
PARAMS=${PARAMS}"--struct_feature_level ${STRUCT_FEATURE_LEVEL}  "
PARAMS=${PARAMS}"--modeling_choice ${MODEL_CHOICE}  "
PARAMS=${PARAMS}"--dropout_rate ${DROPOUT_RATE}  "
#PARAMS=${PARAMS}"--ner_vocab_file ner.vocab.json  "
PARAMS=${PARAMS}"--seed 19786403  "
PARAMS=${PARAMS}"--time_limit -1  "
PARAMS=${PARAMS}"--query_model_size ${QUERY_MODEL_SIZE}  "
PARAMS=${PARAMS}"--early_stop_count_on_rouge 3 "
# PARAMS=${PARAMS}"--log_freq 1 "
if [[ "${MODEL_DIR}" != "none" ]]; then
    PARAMS=${PARAMS}"--model_dir ${MODEL_DIR} "
fi
if [[ "${GLOB_FILE_PATTERN}" != "none" ]]; then
    PARAMS=${PARAMS}"--glob_file_pattern ${GLOB_FILE_PATTERN} "
fi

MAIN_APP=train_main.py


GPU_ID_LIST=(${GPU_IDS//,/ })
NUM_GPUS=${#GPU_ID_LIST[@]}
echo "num_gpus:    ${NUM_GPUS}"

if [ "${NUM_GPUS}" -gt "1" ]; then
    echo "CUDA_VISIBLE_DEVICES=${GPU_IDS} accelerate launch --config_file ./accelerate_config.${DATASOURCE}.yaml ${MAIN_APP} ${PARAMS}"
    CUDA_VISIBLE_DEVICES=${GPU_IDS} accelerate launch --config_file ./accelerate_config.${DATASOURCE}.yaml ${MAIN_APP} ${PARAMS} >>${RUN_LOG} 2>&1 &
else
    echo "CUDA_VISIBLE_DEVICES=${GPU_IDS} accelerate launch --main_process_port=${PROCESS_PORT_ID} --mixed_precision=no --num_processes=1 --num_machines=1 ${MAIN_APP} ${PARAMS}"
    CUDA_VISIBLE_DEVICES=${GPU_IDS} accelerate launch --main_process_port=${PROCESS_PORT_ID} --mixed_precision=no --num_processes=1 --num_machines=1 ${MAIN_APP} ${PARAMS} >>${RUN_LOG} 2>&1 &
fi

#fi
