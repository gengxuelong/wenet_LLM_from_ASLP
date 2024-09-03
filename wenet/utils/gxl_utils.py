import logging
import os

import gxl_ai_utils.utils.utils_file
import torch
import whisper
from torch import nn
from torch.utils.data import DataLoader

# from wenet.dataset.dataset import Dataset
from wenet.text.base_tokenizer import BaseTokenizer
from wenet.text.whisper_tokenizer import WhisperTokenizer
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.encoder import TransformerEncoder
from wenet.utils.checkpoint import load_checkpoint
from wenet.whisper.convert_whisper_to_wenet_config_and_ckpt_gxl import main as convert_to_wenet
from wenet.whisper.whisper import Whisper as WenetWhisper


class Whisper_Utils:
    def __init__(self):
        """"""

    @staticmethod
    def print_all_release():
        print('Available models:')
        for item in (whisper.available_models()):
            print(item)
        print('Available models end.total: {}'.format(len(whisper.available_models())))

    @staticmethod
    def load_whisper(model_name):
        temp_source_pt_dir = "./.cache/whisper/source_pt/"
        temp_wenet_pt_dir = "./.cache/whisper/wenet_pt/"
        os.makedirs(temp_source_pt_dir, exist_ok=True)
        os.makedirs(temp_wenet_pt_dir, exist_ok=True)
        if not os.path.exists(os.path.join(temp_wenet_pt_dir, f"wenet_whisper_{model_name}.pt")):
            whisper.load_model(model_name, download_root=temp_source_pt_dir, device="cpu")
            convert_to_wenet(temp_wenet_pt_dir, temp_source_pt_dir, model_name)

        config_path = os.path.join(temp_wenet_pt_dir, f"train_{model_name}.yaml")
        ckpt_path = os.path.join(temp_wenet_pt_dir, f"wenet_whisper_{model_name}.pt")
        vocab_path = os.path.join(temp_wenet_pt_dir, f'units_whisper_{model_name}.txt')
        config_dict = gxl_ai_utils.utils.utils_file.load_dict_from_yaml(config_path)
        vocab_size = config_dict['output_dim']
        encoder = TransformerEncoder(config_dict['input_dim'],
                                     **config_dict['encoder_conf'])
        decoder = TransformerDecoder(config_dict['output_dim'], encoder.output_size(),
                                     **config_dict['decoder_conf'])
        ctc = CTC(config_dict['output_dim'], encoder.output_size(),
                  blank_id=config_dict['ctc_conf']['ctc_blank_id']
                  if 'ctc_conf' in config_dict else 0)
        model = WenetWhisper(
            vocab_size=vocab_size,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            **config_dict['model_conf'])
        gxl_ai_utils.utils.utils_file.set_logging()
        load_checkpoint(model, ckpt_path)
        logging.info('将embedding层的参数初始化到decoder的output层')
        model.decoder.output_layer.weight.data.copy_(model.decoder.embed[0].weight.data)
        logging.info('将embedding层的参数初始化到decoder的output层 end')
        logging.info('将decoder的output层的参数的bias用零初始化')
        nn.init.zeros_(model.decoder.output_layer.bias)
        logging.info('将decoder的output层的参数的bias用零初始化 end')
        return model, config_dict


class Tokenizer_Utils:
    def __init__(self):
        pass

    @staticmethod
    def get_whisper_tokenizer(configs: dict):
        tokenizer = WhisperTokenizer(
            multilingual=configs['whisper_conf']['is_multilingual'],
            num_languages=configs['whisper_conf']['num_languages'])
        return tokenizer


class Inference_Utils:
    def __init__(self):
        pass

    @staticmethod
    def do_inference_for_file(model: nn.Module,
                              tokenizer: BaseTokenizer,
                              data_list_file: str,
                              data_type: str,
                              data_config=None,
                              mode_list=None,
                              output_dir: str = './output/inference',
                              device: torch.device = torch.device('cpu'), ):
        if data_config is None:
            data_config = dict()
        os.makedirs(output_dir, exist_ok=True)
        from wenet.dataset.dataset import Dataset
        data_loader = DataLoader(Dataset(data_type, data_list_file, tokenizer, data_config, False), batch_size=None,
                                 num_workers=0)
        model = model.to(device)
        model.eval()
        files = {}
        if mode_list is None:
            mode_list = ['ctc_greedy_search', 'ctc_prefix_beam_search', 'attention', 'attention_rescoring']
        max_format_len = max([len(mode) for mode in mode_list])
        for mode in mode_list:
            dir_name = os.path.join(output_dir, mode)
            os.makedirs(dir_name, exist_ok=True)
            file_name = os.path.join(dir_name, 'text')
            files[mode] = open(file_name, 'w')
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                keys = batch["keys"]
                feats = batch["feats"].to(device)
                feats_lengths = batch["feats_lengths"].to(device)
                results = model.decode(
                    mode_list,
                    feats,
                    feats_lengths,
                    beam_size=10,
                    decoding_chunk_size=-1,
                    num_decoding_left_chunks=-1,
                    ctc_weight=0.3,
                    simulate_streaming=False,
                    reverse_weight=0.5,
                    context_graph=None)
                for i, key in enumerate(keys):
                    for mode, hyps in results.items():
                        tokens = hyps[i].tokens
                        line = '{} {}'.format(key, tokenizer.detokenize(tokens)[0])
                        logging.info('{} {}'.format(mode.ljust(max_format_len),
                                                    line))
                        files[mode].write(line + '\n')
        for mode, f in files.items():
            f.close()



if __name__ == "__main__":
    model, _ = Whisper_Utils.load_whisper('small')
    print(model)
    from gxl_ai_utils.utils import utils_file
    utils_file.print_model_size(model)
    utils_file.print_model_size(model.encoder)
    utils_file.print_model_size(model.decoder)
