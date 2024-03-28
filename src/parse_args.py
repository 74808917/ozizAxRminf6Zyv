from __future__ import unicode_literals, print_function, division
from abc import ABC, abstractmethod
import argparse
from distutils.util import strtobool
from transformers import MODEL_MAPPING
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

MODELING_CHOICE = [
    "model_ner",
]

STRUCT_FEATURE_LEVEL = [
    "struct.ner"
]

class ArgsParsingBase(ABC):
    def parse(self):
        args = self.parse_args()
        args = self.post_parse(args)
        return args

    @abstractmethod
    def parse_args(self):
        raise NotImplementedError("ArgsParserBase.parse_args not implemented")

    def post_parse(self, args):
        return args


class TrainArgsParsing(ArgsParsingBase):
    def parse_args(self):
        parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
        parser.add_argument('--dataset_root', type=str, required=False, default=".",
                            help='The root directory in which training dataset directory resides.')
        parser.add_argument('--modeldata_root', type=str, required=False, default=".",
                            help='The root directory in which trained model data directory resides.')
        parser.add_argument("--config_folder", type=str, default=None,
                            help="The configuration folder related to data soure (e.g. cnndm or gigaword).")
        parser.add_argument("--split_type", type=str, default=None,
                            help="The dataset splits (e.g. train,valid).")
        parser.add_argument("--pair_type", type=str, default=None,
                            help="The dataset pairs (e.g. atcl,hlit).")
        parser.add_argument("--dataset_file", type=str, default=None,
                            help="A csv or a json file containing the training data.")
        parser.add_argument("--dataset_build_pickle", action="store_true",
                            help="Rebuild pickled dataset file if true regardless of existence.")
        parser.add_argument("--ner_vocab_file", type=str, default=None,
                            help="A pth or a json file containing the ner vocab.")
        parser.add_argument("--dataset_folder", type=str, default=None,
                            help="The directory where prebuilt datasets are saved and loaded.")
        parser.add_argument("--base_model_pretrained_name", type=str, default=None,
                            help="Use pretrained model configuration and weights.")
        parser.add_argument("--base_model_config_name", type=str, default=None,
                            help="Use pretrained model configuration without weights.")
        parser.add_argument("--tokenizer_name", type=str, default=None,
                            help="Pretrained tokenizer name or path if not the same as model_name")
        parser.add_argument("--use_slow_tokenizer", action="store_true",
                            help="If passed, will use a slow tokenizer (not backed by the Tokenizers library).")
        parser.add_argument("--model_dir", type=str, default=None, help="Where to store the final model.")
        parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
        parser.add_argument("--model_type", type=str, default=None,
                            help="Model type to use if training from scratch.",
                            choices=MODEL_TYPES)
        parser.add_argument("--glob_file_pattern", type=str, default=None,
                            help="A checkpoint folder or search pattern or none.")
        parser.add_argument("--modeling_choice", type=str, default="",
                            help="Modeling variant choice.",
                            choices=MODELING_CHOICE)
        parser.add_argument("--struct_feature_level", type=str, default="",
                            help="All extra modules should be excluded if none is chosen.",
                            choices=STRUCT_FEATURE_LEVEL)
        parser.add_argument("--max_train_steps", type=int, default=None,
                            help="Total number of training steps to perform. If provided, overrides num_train_epochs.",)
        parser.add_argument('--early_stop_count_on_rouge', type=int, required=False, default=None,
                            help='Early stop if rouge scores flatten consecutively for the specific times.')
        parser.add_argument('--skip_except', action="store_true",
                            help="Set the flag to continue training by skipping an iteration that"
                                "raises an except caused by some ill-formed batch data"
                                "(e.g. Gigaword coref data contain some coref annotation having no referent).")
        parser.add_argument('--time_limit', type=int, default=-1,
                            help="time_limit in seconds. less than or equal to 0 means no limit.")
        parser.add_argument('--log_freq', type=int, required=False, default=20,
                           help='Log frequency on screen.')
        parser.add_argument("--query_model_size", dest='query_model_size', type=lambda x: bool(strtobool(x)), default=False,
                            help="Query model size.")
        parser.add_argument("--dropout_rate", type=str,
                            default='{"activation_dropout":0.1,"classif_dropout":0.1}',
                            # default=None,
                            help="Change dropout or None.")
        args = parser.parse_args()
        return args

    def post_parse(self, args):
        args.split_type = eval(args.split_type)
        args.pair_type = eval(args.pair_type)
        return args


class TestArgsParsing(ArgsParsingBase):
    def parse_args(self):
        parser = argparse.ArgumentParser(description="Test a trained model on a summarization task")
        parser.add_argument('--dataset_root', type=str, required=False, default=".",
                            help='The root directory in which training dataset directory resides.')
        parser.add_argument('--modeldata_root', type=str, required=False, default=".",
                            help='The root directory in which trained model data directory resides.')
        parser.add_argument("--config_folder", type=str, default=None,
                            help="The configuration folder related to data soure (e.g. cnndm or gigaword).")
        parser.add_argument("--split_type", type=str, default=None,
                            help="The dataset splits (e.g. test).")
        parser.add_argument("--pair_type", type=str, default=None,
                            help="The dataset pairs (e.g. atcl,hlit).")
        parser.add_argument("--dataset_file", type=str, default=None,
                            help="A csv or a json file containing the training data.")
        parser.add_argument("--dataset_build_pickle", action="store_true",
                            help="Rebuild pickled dataset file if true regardless of existence.")
        parser.add_argument("--dataset_folder", type=str, default=None,
                            help="The directory where prebuilt datasets are saved and loaded.")
        parser.add_argument("--base_model_pretrained_name", type=str, default=None,
                            help="Use pretrained model configuration and weights.")
        parser.add_argument("--base_model_config_name", type=str, default=None,
                            help="Use pretrained model configuration without weights.")
        parser.add_argument("--tokenizer_name", type=str, default=None,
                            help="Pretrained tokenizer name or path if not the same as model_name")
        parser.add_argument("--use_slow_tokenizer", action="store_true",
                            help="If passed, will use a slow tokenizer (not backed by the Tokenizers library).")
        parser.add_argument("--model_dir", type=str, default=None, help="Where to store the final model.")
        parser.add_argument("--model_type", type=str, default=None,
                            help="Model type to use if training from scratch.",
                            choices=MODEL_TYPES)
        parser.add_argument("--glob_file_pattern", type=str, default=None,
                            help="A checkpoint folder or search pattern or none.")
        parser.add_argument("--modeling_choice", type=str, default="",
                            help="Modeling variant choice.",
                            choices=MODELING_CHOICE)
        parser.add_argument('--test_batch_size', type=int, default=1)
        parser.add_argument('--test_task', type=str, required=False, default="generation",
                            help='One of [generation,perplexity].')
        parser.add_argument('--evaluation_folder', type=str, required=False, default="evaluation.result",
                            help='Evaluate folder for storing test/eval results.')
        parser.add_argument('--result_folder', type=str, required=False, default=None,
                            help='The folder to which test/eval results are saved.')
        args = parser.parse_args()
        return args

    def post_parse(self, args):
        args.split_type = eval(args.split_type)
        args.pair_type = eval(args.pair_type)
        return args
