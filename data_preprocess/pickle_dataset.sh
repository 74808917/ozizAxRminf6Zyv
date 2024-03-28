#!/bin/bash


echo "Usage: pickle_dataset.sh"
echo "        --dataset_root /directory/to/dataset/cnndm/by_stanford_corenlp_4.4.0/bart-large-cnn"
echo "        --dataset_folder struct.ner"
echo "        --split_type \"train,validation,test\""
echo "        --pair_type \"article,highlights\""
echo "        --ext_type dataset.json"
echo "        --pyvenv_dir /directory/to/python/virtual/environment"
echo ""
echo "Press enter Y/y to continue, N/n to abort"
read -r input


if [[ "$input" = "y" ]] || [[ "$input" = "Y" ]]; then
    PYTHON_VENV_BIN_DIR=${12}/bin
    echo "Python venv bin dir: ${PYTHON_VENV_BIN_DIR}"
    source ${PYTHON_VENV_BIN_DIR}/activate
    export PYTHONPATH="$PYTHONPATH:$PWD:$PWD/..:$PWD/../.."

    python3 pickle_dataset.py \
        --dataset_root   $2 \
        --dataset_folder $4 \
        --split_type     $6 \
        --pair_type      $8 \
        --ext_type       ${10} &
else
    echo "--dataset_root       $2"
    echo "--dataset_folder     $4"
    echo "--split_type         $6"
    echo "--pair_type          $8"
    echo "--ext_type           ${10}"
    echo "--pyvenv_dir         ${12}"
    return
fi
