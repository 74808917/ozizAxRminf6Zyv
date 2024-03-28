from __future__ import unicode_literals, print_function, division
from parse_args import ModelTokenizeDataArgsParse
from one_input_model_tokenizer import OneInputFileModelTokenizer


def main():
    import logging
    logging.basicConfig(level=logging.NOTSET)
    logger = logging.getLogger(__name__)

    args = ModelTokenizeDataArgsParse()()

    if args.split_type is not None:
        args.split_type = args.split_type.strip(" []").split(",")
    if args.column_names is not None:
        args.column_names = args.column_names.strip(" []").split(",")

    # Build dataset
    if args.datasource_name in ["cnndm", "xsum"]:
        OneInputFileModelTokenizer()(args, logger)
    else:
        raise ValueError("No handler for the datasource.")
    logger.info("Done!")


if __name__ == "__main__":
    main()
