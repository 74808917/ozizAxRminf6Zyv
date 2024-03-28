from __future__ import unicode_literals, print_function, division


def query_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size = (param_size + buffer_size) / 1024**2 # in MB unit
    return size

def customize_model_dropout(args, model, accelerator, logger):
    import json
    if args.dropout_rate is None or args.dropout_rate.capitalize() == 'None':
        logger.warning(f"No customized dropout changes on the pretrained model {model.__class__.__name__}")
        return model

    try:
        dropout_rate = json.loads(args.dropout_rate)
    except:
        drs = args.dropout_rate.strip("{}")
        drs = drs.split(",")
        dropout_rate = {}
        for kv in drs:
            k, v = kv.split(":")
            dropout_rate[f"{k}"] = float(v)
        logger.info(f"activation dropout_rate: {dropout_rate}")

    if accelerator is not None:
        model = accelerator.unwrap_model(model)

    if "bart" in model.seq2seq.__class__.__name__.lower():
        for name, module in model.named_modules():
            for k, v in dropout_rate.items():
                if hasattr(module, k):
                    logger.info(f"Change {name} ({type(module)}) dropout")
                    setattr(module, k, v)
        logger.info(f"Customized dropout on the pretrained model {model.seq2seq.__class__.__name__}")
    return model
