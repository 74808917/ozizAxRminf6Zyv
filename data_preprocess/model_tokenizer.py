from __future__ import unicode_literals, print_function, division
from abc import ABC, abstractmethod
from transformers import AutoTokenizer

class ModelTokenizer(ABC):
    def __call__(self, args, logger):
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        for split_type in args.split_type:
            self.process(args=args,
                         split_type=split_type,
                         tokenizer=tokenizer,
                         logger=logger)

    @abstractmethod
    def process(
        self,
        args,
        split_type,
        tokenizer,
        logger
    ):
        raise NotImplementedError("Abstract method.")
