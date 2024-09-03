import logging

from wenet.ASRLLM.ASRLLM import ASRLLM_Model
# from wenet.transformer.whisper_encoder import OpenAIWhisperEncoder
from wenet.transformer.hubert_encoder import S3prlFrontend
from wenet.transformer.whisper_encoder import OpenAIWhisperEncoder
from wenet.utils.checkpoint import load_checkpoint, load_trained_modules


def init_ASLP_ASRLLM(args, configs, is_inference=False):
    llm_path = configs["llm_path"]
    low_resource = configs["low_resource"]
    lora = configs["use_lora"]
    lora_alpha = configs["lora_alpha"]
    lora_rank = configs["lora_rank"]
    lora_dropout = configs["lora_dropout"]
    llama_model_generate_min_length = configs["llama_model_generate_min_length"]
    llama_model_generate_num_beams = configs["llama_model_generate_num_beams"]
    llama_model_generate_do_sample = configs["llama_model_generate_do_sample"]
    llama_model_generate_top_p = configs["llama_model_generate_top_p"]
    llama_model_generate_repetition_penalty = configs["llama_model_generate_repetition_penalty"]
    llama_model_generate_length_penalty = configs["llama_model_generate_length_penalty"]
    llama_model_generate_temperature = configs["llama_model_generate_temperature"]

    if configs['encoder'] == 'whisper':
        encoder = OpenAIWhisperEncoder(**configs['encoder_conf'])
    elif configs['encoder'] == 'hubert':
        encoder = S3prlFrontend(**configs['encoder_conf'])
    else:
        encoder = None
    model = ASRLLM_Model(
        encoder=encoder,
        llm_path=llm_path,
        lora=lora,
        lora_alpha=lora_alpha,
        lora_rank=lora_rank,
        lora_dropout=lora_dropout,
        low_resource=low_resource,
        llama_model_generate_min_length=llama_model_generate_min_length,
        llama_model_generate_num_beams=llama_model_generate_num_beams,
        llama_model_generate_do_sample=llama_model_generate_do_sample,
        llama_model_generate_top_p=llama_model_generate_top_p,
        llama_model_generate_repetition_penalty=llama_model_generate_repetition_penalty,
        llama_model_generate_length_penalty=llama_model_generate_length_penalty,
        llama_model_generate_temperature=llama_model_generate_temperature,
        encoder_type=configs['encoder'],
        is_inference=is_inference,
        downsample_rate=configs.get('downsample_rate', 1),
    )
    logging.info(f'ASLP-LOG：init_salmonn()：开始加载初始化模型')
    if hasattr(args, 'checkpoint') and args.checkpoint is not None:
        logging.info(f'ASLP-LOG： 设置了初始化模型位置，开始加载，参数文件位置：{args.checkpoint}')
        infos = load_checkpoint(model, args.checkpoint)
    elif hasattr(args, 'checkpoint') and args.enc_init is not None:
        infos = load_trained_modules(model, args)
    else:
        infos = {}
    configs["init_infos"] = infos
    print(configs)
    logging.info('ASLP-LOG：加载初始化模型完毕')

    logging.info('ASLP-LOG：开始选择性冻结模块')
    fire_module = configs.get("fire_module", None)
    if fire_module is None:
        logging.info('ASLP-LOG：没有选择解冻的模块,也就是没有训练参数，直接报错返回')
        raise ValueError('没有选择解冻的模块,也就是没有训练参数，直接报错返回')
    for k, p in model.named_parameters():
        if fire_module == 'link':
            # link 包括下采样块， transformer块， 前后linear块
            if k.startswith("llama_model") or k.startswith("speech_encoder"):
                p.requires_grad = False
        elif fire_module == 'encoder':
            if not k.startswith("speech_encoder"):
                p.requires_grad = False
        elif fire_module == 'llm':
            if not k.startswith("llama_model"):
                p.requires_grad = False
        logging.info(f"{k} {p.requires_grad}")
    logging.info('ASLP-LOG：冻结完毕')

    return model, configs
