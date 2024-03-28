#!/bin/bash


if [ "$#" -ne 8 ]; then
    echo "Usage: . corenlp_annotate.sh"
    echo "              port[9000]"
    echo "              datasource[cnndm|xsum]"
    echo "              dataset_root[/directory/to/dataset/cnndm(or xsum)]"
    echo "              split_type[train|validation|test]"
    echo "              column_names[article,highlights|document,summary]"
    echo "              mode[PROP_DEFAULT|PROP_NER|PROP_NER_COREF]"
    echo "              output_dir[/directory/to/dataset/cnndm(or xsum)/by_stanford_corenlp_4.4.0]"
    echo "              pyvenv_dir[/directory/to/python/virtual/environment]"

elif ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "port should be integer"
else
    export PYTHONPATH=$PYTHONPATH:$PWD:$PWD/..:$PWD/../..
    export CLASSPATH=$CLASSPATH:$7/*:
    source $8/bin/activate

    RUN_TRACE_DIR="./run_trace"
    [ -d ${RUN_TRACE_DIR} ] || mkdir -p ${RUN_TRACE_DIR}

    today=`date '+%Y_%m_%d_%H_%M'`;
    LOGFILE="${RUN_TRACE_DIR}/$2.corenlp.parse.$4.$today.err"
    echo ${LOGFILE}
    echo $HOSTNAME >${LOGFILE}

    declare -A MODE_FOLDER_DICT
    MODE_FOLDER_DICT["PROP_DEFAULT"]=""
    MODE_FOLDER_DICT["PROP_NER"]=".ner"

    FOLDER_SUFFIX=${MODE_FOLDER_DICT["$6"]}
    OUTPUT_DIR=$7/corenlp.parse${FOLDER_SUFFIX}
    echo "Output directory: ${OUTPUT_DIR}"

    nohup python corenlp_annotate.py \
        --port $1 \
        --dataset_root $3 \
        --split_type $4 \
        --column_names $5 \
        --corenlp_mode $6 \
        --source_folder downloaded \
        --output_folder ${OUTPUT_DIR} 2>${LOGFILE} >/dev/null &
fi
