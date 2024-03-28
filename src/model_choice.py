from __future__ import unicode_literals, print_function, division

def import_model(args, options, vocabs, logger, **kwargs):
    try:
        modeling_choice = options.training.train_state["modeling_choice"]
    except Exception as ex:
        modeling_choice = None
    modeling_choice = args.modeling_choice if modeling_choice is None else modeling_choice
    assert modeling_choice == args.modeling_choice, \
            f"The specific {args.modeling_choice} is not compatible with the saved {modeling_choice}"
    if modeling_choice == "model_ner":
        logger.info("Model is chosen from model.model_ner")
        from model.model_ner import Model
    else:
        raise ValueError("Chosen model name is unknown.")
    return Model(args, options, vocabs=vocabs, logger=logger, **kwargs)
