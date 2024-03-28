# Set up Python virtual environment
    Refer to requirements.txt

#### *All following multiline commands with backslash are for legibility. To run them in a shell console, they should be one-line commands where backslash replaced with whitespace.
#### *Training and evaluation assume two-GPU parallelism.

# Prepare datasets
## An example of setting up directory structure for preprocessing data
    dataset
        |- cnndm
        |    |- downloaded
        |    |- by_stanford_corenlp_4.4.0
        |    |      |- corenlp.parse.ner
        |    |      |- bart-large-cnn
        |    |              |- struct.ner
        |    |- model_tokenization
        |    |      |- bart-large-cnn
        |- xsum 
            |- downloaded
            |- by_stanford_corenlp_4.4.0
            |       |- corenlp.parse.ner
            |       |- bart-large-xsum
            |               |- struct.ner
            |- model_tokenization
                    |- bart-large-xsum
    where
        - downloaded: store the downloaded original dataset json files (e.g., train.json, validation.json, test.json).
        - by_stanford_corenlp_4.4.0: use Stanford CoreNLP version as directory for annotation data.
        - corenlp.parse.ner: Stanford CoreNLP annotated data directory.
        - bart-large-cnn: use model tokenization name as directory to make clear the model tokenization profile.
        - struct.ner: the folder where to store our datasets built from Stanford CoreNLP annotated data and tokenized by bart-large-cnn and bart-large-xsum respectively.
        - model_tokenization: preprocessed validation and test datasets are stored here. They are built off model tokenizer without Stanford CoreNLP annotation.

## Stanford CoreNLP annotation for NER data
    >Download cnndm and xsum datasets as json files into downloaded folders as above.
        - Refer to data_preprocess/download_datasets.py.
        - Troubleshooting: "The CNN/DailyMail dataset is hosted on Google Drive, which limits the number of downloads. You probably exceeded Google Drive's quota for this file, you'll   need to try again later (the quota is reset once a day)."
    >Download Stanford CoreNLP 4.4.0 tool (https://nlp.stanford.edu/software/stanford-corenlp-4.4.0.zip).
    >Unzip it to anywhere outside the above dataset structure.
    >Change to the unziped tool directory.
    >Launch the tool service as follows.
            nohup java -mx8g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port $port -timeout 60000 2>corenlp_svr_log_$port.out >/dev/null &
        Note:
            1. Replacing $port with any available port number, e.g., 9000.
    >Change to data_preprocess folder.
    >Run cnndm annotation as follows.
            . corenlp_annotate.sh \
                $port \
                cnndm \
                /directory/to/dataset/cnndm \
                train \
                "article,highlights" \
                PROP_NER \
                /directory/to/dataset/cnndm/by_stanford_corenlp_4.4.0 \
                /directory/to/python/virtual/environment
        Note:
            1. $port is the same port as the running Stanford CoreNLP service.
            2. Annotation of full datasets may take a day or two to complete.
    >Annotation of XSum is same but
        - replacing any cnndm with xsum.
        - replacing "article,highlights" with "document,summary".

## Build train dataset from annotation data
    >Change to data_preprocess folder.
    >Run build process for cnndm as follows.
            . annotated_dataset_builder.sh \
                --dataset_root        /directory/to/dataset \
                --datasource          cnndm \
                --downloaded_folder   downloaded \
                --source_folder       corenlp.parse.ner \
                --output_folder       bart-large-cnn/struct.ner \
                --split_types         train \
                --pair_types          [article,highlights] \
                --tokenizer_name      facebook/bart-large-cnn \
                --build_vocab         true \
                --build_compose       true \
                --reconcile_vocab     true \
                --build_stype2id      true \
                --use_named_entity    true \
                --annotated_data_dir  by_stanford_corenlp_4.4.0 \
                --pyvenv_dir          /directory/to/python/virtual/environment
    >Building xsum train dataset is same but 
        - replacing any cnndm with xsum.
        - replacing [article,highlights] with [document,summary].
        - replacing bart-large-cnn with bart-large-xsum.

## Preprocess validation and test datasets
    >Change to data_preprocess folder.
    >Run validation build process for cnndm as follows.
            . model_token_dataset_builder.sh \
                --dataset_root     /directory/to/dataset \
                --datasource_name  cnndm \
                --source_folder    downloaded \
                --output_folder    model_tokenization/bart-large-cnn \
                --source_file_ext  .json \
                --output_file_ext  .dataset.json \
                --split_type       validation \
                --column_names     [article,highlights] \
                --tokenizer_name   facebook/bart-large-cnn \
                --pyvenv_dir       /directory/to/python/virtual/environment
        Note:
            - To build test dataset, replace validation with test.
    >Preprocessing xsum is same but 
        - replacing any cnndm with xsum.
        - replacing [article,highlights] with [document,summary].
        - replacing bart-large-cnn with bart-large-xsum.
    >Copy or move the preprocessed validation and test datasets to the preprocessed annotated train dataset folders of cnndm and xsum respectively.

## Pickle datasets
    *Loading datasets runtime as text files is very slow. To speed up the load, pickle datasets as serializable data.
    >Change to data_preprocess folder.
    >Run pickling process for cnndm as follows.
            . pickle_dataset.sh \
                --dataset_root /directory/to/dataset/cnndm/by_stanford_corenlp_4.4.0/bart-large-cnn \
                --dataset_folder struct.ner \
                --split_type train \
                --pair_type "article,highlights" \
                --ext_type dataset.json \
                --pyvenv_dir /directory/to/python/virtual/environment
        Note:
            To pickle validation and test, replace train with validation and test respectively.
    >pickling process for xsum is same, but
        - replacing any cnndm with xsum.
        - replace bart-large-cnn with bart-large-xsum.
        - replacing "article,highlights" with "document,summary".


# Training session
## An example on CNNDM
    . run.training.mgpu.sh \
         --modeldata_dir /directory/to/saving/model/data \
         --dataset_dir /directory/to/dataset \
         --datasource cnndm \
         --nlp_parsing_ver by_stanford_corenlp_4.4.0 \
         --data_build_type bart-large-cnn/struct.ner \
         --pretrained_model facebook/bart-large-cnn \
         --split_type '["train","validation"]' \
         --pair_type '["article","highlights"]' \
         --data_file_suffix .dataset.json \
         --struct_feature_level struct.ner \
         --query_model_size true \
         --gpu_ids 0,1 \
         --process_port_id 9998 \
         --modeling_choice model_ner \
         --dropout_rate '{activation_dropout:0.1,classif_dropout:0.1}' \
         --config_folder bart \
         --model_dir model_data/bart_ats \
         --glob_file_pattern none
         --pyvenv_dir /directory/to/python/virtual/environment
    Note:
        - process_port_id: any available port number.

## Training on XSum
    >Same as training CNNDM, but
        - replacing cnndm with xsum.
        - replacing bart-large-cnn with bart-large-xsum.
        - replacing '["article","highlights"]' with '["document","summary"]'.

# Evaluation run
## An example on CNNDM
    . run.eval.mgpu.sh \
         --modeldata_dir  /directory/to/the/saving/model/data \
         --dataset_dir  /directory/to/dataset \
         --datasource cnndm \
         --nlp_parsing_ver by_stanford_corenlp_4.4.0 \
         --data_build_type bart-large-cnn/struct.ner \
         --pretrained_model facebook/bart-large-cnn \
         --split_type  '["test"]' \
         --pair_type '["article","highlights"]' \
         --data_file_suffix .dataset.json \
         --gpu_ids 0,1 \
         --process_port_id 9999 \
         --modeling_choice model_ner \
         --model_dir model_data/bart_ats \
         --glob_file_pattern _epoch_5_batch_49183 \
         --test_task generation \
         --result_folder none \
         --config_folder bart \
         --pyvenv_dir  /directory/to/python/virtual/environment
    Note:
        - process_port_id: any available port number.
        - glob_file_pattern: any saved checkpoint of your choice. Specifying it to none will load the last saved model weights.

## Model evaluation on XSum
    >Same as evaluation on CNNDM, but
        - replacing cnndm with xsum.
        - replacing bart-large-cnn with bart-large-xsum.
        - replacing '["article","highlights"]' with '["document","summary"]'.
