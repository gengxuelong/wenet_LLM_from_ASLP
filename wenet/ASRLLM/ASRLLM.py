import logging
import sys

sys.path.append('../../')
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from wenet.transformer.encoder import TransformerEncoder
from wenet.utils.common import add_sos_eos4speech_llm


class ASLPConv1dSubsampling2(nn.Module):
    """Conv1d subsampling module.

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int):
        """Construct an Conv1dSubsampling object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(idim, odim, 3, 1),
            torch.nn.GELU(),
            torch.nn.Conv1d(odim, odim, 3, 2),
            torch.nn.GELU(),
        )

    def forward(self, x):
        """

        Args:
            x: (B, T, idim)

        Returns:
        """
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x


class ASLPConv1dSubsampling4(nn.Module):
    """Conv1d subsampling module.

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int):
        """Construct an Conv1dSubsampling object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(idim, odim, 3, 1),
            torch.nn.GELU(),
            torch.nn.Conv1d(odim, odim, 3, 2),
            torch.nn.GELU(),
            torch.nn.Conv1d(odim, odim, 3, 2),
            torch.nn.GELU(),
        )

    def forward(self, x):
        """

        Args:
            x: (B, T, idim)

        Returns:
        """
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x


class ASLPConv1dSubsampling6(nn.Module):
    """Conv1d subsampling module.

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int):
        """Construct an Conv1dSubsampling object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(idim, odim, 3, 1),
            torch.nn.GELU(),
            torch.nn.Conv1d(odim, odim, 3, 2),
            torch.nn.GELU(),
            torch.nn.Conv1d(odim, odim, 3, 3),
            torch.nn.GELU(),
        )

    def forward(self, x):
        """

        Args:
            x: (B, T, idim)

        Returns:
        """
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x


class ASLPConv1dSubsampling8(nn.Module):
    """Conv1d subsampling module.

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int):
        """Construct an Conv1dSubsampling object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(idim, odim, 3, 1),
            torch.nn.GELU(),
            torch.nn.Conv1d(odim, odim, 3, 2),
            torch.nn.GELU(),
            torch.nn.Conv1d(odim, odim, 3, 2),
            torch.nn.GELU(),
            torch.nn.Conv1d(odim, odim, 3, 2),
            torch.nn.GELU(),
        )

    def forward(self, x):
        """

        Args:
            x: (B, T, idim)

        Returns:
        """
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x




class ASRLLM_Model(nn.Module):
    def __init__(self, encoder, llm_path, lora=True, lora_alpha=32,
                 lora_rank=8, lora_dropout=0.1, low_resource=False,
                 prompt_pattern="{}：<Speech><SpeechHere></Speech>",
                 llama_model_generate_max_length=200, llama_model_generate_min_length=1,
                 llama_model_generate_num_beams=4, llama_model_generate_do_sample=True, 
                 llama_model_generate_top_p=0.9,llama_model_generate_repetition_penalty=1.0, 
                 llama_model_generate_length_penalty=1.0, llama_model_generate_temperature=1.0, 
                 is_inference=False, downsample_rate=1,**kwargs):
        """"""
        super().__init__()
        self.downsample_rate = downsample_rate
        self.prompt = "转录如下音频。"
        self.speech_encoder = encoder
        """
        hubert的dim是1024， whisper的dim的1280， 通过线性层转换
        """
        self.encoder_type = kwargs.get("encoder_type", "whisper")
        logging.info(f'ASLP-LOG： encoder_type: {self.encoder_type}')
        self.hubert_dim2whisper_dim = nn.Linear(encoder.output_size(),
                                                1280) if self.encoder_type == "hubert" else nn.Identity()
        self.ln_speech = nn.LayerNorm(1280)

        # 连接层,transformer类型 51.6M，Qformer类型的代码请参考SALMONN官方代码
        # https://github.com/bytedance/SALMONN
        self.speech_transformer = TransformerEncoder(
            input_size=1280,
            output_size=1280,
            attention_heads=4,
            linear_units=2560,
            num_blocks=4,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.0,
            input_layer="linear",
            pos_enc_layer_type="abs_pos",
            normalize_before=True
        )

        # 默认情况下，推理事LLM的数据类型为fp16, 可以改为fp32
        # 本团队测试： LLM的数据类型为fp16还是为fp32并不改变推理结果
        self.low_resource = low_resource
        if not low_resource:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                llm_path,
                torch_dtype=torch.float16,
                # torch_dtype=torch.float32 if is_inference else torch.float16,
                trust_remote_code=True
            )
        else:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                llm_path,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True
            )

        self.max_length = llama_model_generate_max_length
        self.min_length = llama_model_generate_min_length
        self.num_beams = llama_model_generate_num_beams
        self.do_sample = llama_model_generate_do_sample
        self.top_p = llama_model_generate_top_p
        self.repetition_penalty = llama_model_generate_repetition_penalty
        self.length_penalty = llama_model_generate_length_penalty
        self.temperature = llama_model_generate_temperature

        # lora
        self.lora = lora
        if lora:
            logging.info("ASLP-LOG： 使用lora了")
            target_modules = ['W_pack', 'o_proj', 'gate_proj', 'down_proj']
            if is_inference:
                self.peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=True,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=target_modules,
                )
            else:
                self.peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=target_modules,
                )
            self.llama_model = get_peft_model(self.llama_model, self.peft_config)

        self.llama_tokenizer = AutoTokenizer.from_pretrained(
            llm_path, use_fast=False, trust_remote_code=True)

        self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llama_tokenizer.padding_side = "right"

        # 中间层与LLM的映射模块
        self.speech_llama_proj = nn.Linear(
            1280, self.llama_model.config.hidden_size)

        self.prompt_pattern = prompt_pattern

        self.down_sample_2 = nn.Identity()
        if self.downsample_rate == 2:
            self.down_sample_2 = ASLPConv1dSubsampling2(1280, 1280)
        elif self.downsample_rate == 4:
            self.down_sample_2 = ASLPConv1dSubsampling4(1280, 1280)
        elif self.downsample_rate == 8:
            self.down_sample_2 = ASLPConv1dSubsampling8(1280, 1280)
        elif self.downsample_rate == 6:
            self.down_sample_2 = ASLPConv1dSubsampling6(1280, 1280)


    def forward(self,
                batch,
                device,
                ):
        """"""
        wavs = batch['feats'].to(device)
        wavs_len = batch['feats_lengths'].to(device)
        labels = batch['target'].to(device)

        """
        首先 得到音频编码的特征
        speech_embeds ： 为输入LLM的音频编码特征， 已经对齐特征维度。 shape:(b, t, 4096)
        """
        speech_embeds, speech_lens = self.speech_encoder(wavs, wavs_len)
        speech_embeds = self.hubert_dim2whisper_dim(speech_embeds)
        speech_embeds = self.down_sample_2(speech_embeds)
        speech_embeds = self.ln_speech(speech_embeds)  # 特征维度： 1280
        B, T, C = speech_embeds.shape
        speech_embeds, speech_masks = self.speech_transformer(speech_embeds, speech_lens)
        speech_embeds = self.speech_llama_proj(speech_embeds)

        """
        接着处理prompt， 将其首先使用分词器编码成数字序列shape(1,N), 接着使用LLM的Embedding层对其进行编码shape(1,N, 4096)
        embed_tokens： nn.Embedding(65000, 4096). 
        # prompt-> :  USER:转录如下音频. <Speech>speech_embeds</Speech>\nASSISTANT:(Atom模型)
        # prompt-> :  chat_user_id 转录如下音频. <Speech>speech_embeds</Speech>chat_assistant_id  # Baichuan-7B-chat
        # embed_tokens-> ： nn.Embedding(65000, 4096)
        """
        prompt = self.prompt
        embed_tokens = self.llama_model.model.model.embed_tokens if self.lora else self.llama_model.model.embed_tokens

        prompt_left, prompts_right = self.prompt_pattern.format(prompt).split(
            '<SpeechHere>')

        prompt_left_ids = self.llama_tokenizer(  # shape: [1, 7]
            prompt_left,
            return_tensors="pt",
            add_special_tokens=False
        ).to(speech_embeds.device).input_ids
        prompt_left_embeds = embed_tokens(prompt_left_ids).repeat_interleave(B, dim=0)  # torch.Size([17, 7, 4096])
        prompt_left_ids = prompt_left_ids.repeat_interleave(B, dim=0)  # torch.Size([17, 7])

        prompt_right_ids = self.llama_tokenizer(
            prompts_right,
            return_tensors="pt",
            add_special_tokens=False
        ).to(speech_embeds.device).input_ids
        prompt_right_embeds = embed_tokens(prompt_right_ids).repeat_interleave(B, dim=0)  # torch.Size([17, 14, 4096])
        prompt_right_ids = prompt_right_ids.repeat_interleave(B, dim=0)

        """
        处理labels, labels本本身是已经padding过的，shape:(B , T)
        首先对其经过sos_eos处理， 得到两个padded_labels_in和padded_labels_out,
        labels_in不加入bos 
        然后使用Embedding层对padded_labels_in进行编码
        接着得到bos_ids和bos_embeds, eos_ids和eos_embeds.shape: (B,1),  (B,1, 4096)
        """
        labels_ids = labels  # torch.Size([17, 13])
        labels_in, labels_out = add_sos_eos4speech_llm(labels_ids, self.llama_tokenizer.bos_token_id,
                                                       self.llama_tokenizer.eos_token_id, ignore_id=-100)
        labels_in_embeds = embed_tokens(labels_in)  # torch.Size([17, 13, 4096])
        bos_ids = torch.ones([B, 1], dtype=torch.long,  # torch.Size([17, 1]), true value is 1
                             device=speech_embeds.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = embed_tokens(bos_ids)  # torch.Size([17, 1, 4096])

        eos_ids = torch.ones([B, 1], dtype=torch.long,  # torch.Size([17, 1]), true value is 2
                             device=speech_embeds.device) * self.llama_tokenizer.eos_token_id
        eos_embeds = embed_tokens(eos_ids)
        user_ids = torch.ones([B, 1], dtype=torch.long,  # torch.Size([17, 1]), true value is 1
                              device=speech_embeds.device) * 195
        user_embeds = embed_tokens(user_ids)
        assistant_ids = torch.ones([B, 1], dtype=torch.long,  # torch.Size([17, 1]), true value is 1
                                   device=speech_embeds.device) * 196
        assistant_embeds = embed_tokens(assistant_ids)
        """
        将左prompt 音频 右prompt label_in 的高纬特征拼接在一起。
        将左prompt 音频 右prompt label_out 的id拼接在一起作为ground truth
        依据Transformers的使用规范， 输入内容和Labels只需要严格对齐即可
        """
        speech_embeds_B, speech_embeds_T = speech_embeds.size(0), speech_embeds.size(1)
        speech_ids = torch.ones([speech_embeds_B, speech_embeds_T], dtype=torch.long, device=speech_embeds.device)
        concat_ids = torch.cat([bos_ids, user_ids, prompt_left_ids, speech_ids, prompt_right_ids, assistant_ids], dim=1)
        filled_ids = concat_ids.fill_(-100)  # In CrossEntropyLoss(), ignore_index = -100
        embeds = torch.cat(
            [bos_embeds, user_embeds, prompt_left_embeds, speech_embeds, prompt_right_embeds, assistant_embeds,
             labels_in_embeds, eos_embeds], dim=1)
        labels = torch.cat([filled_ids, labels_out], dim=1)

        if self.low_resource:
            embeds = embeds.to(torch.int8).to(torch.float16)
        outputs = self.llama_model(
            inputs_embeds=embeds,
            labels=labels,
        )
        loss = outputs['loss']  # 0维张量，纯数字
        return {"loss": loss}

    def generate(
            self,
            wavs,
            wavs_len,
    ):
        prompt = self.prompt
        speech_embeds, speech_lens = self.speech_encoder(wavs, wavs_len)
        speech_embeds = self.hubert_dim2whisper_dim(speech_embeds)
        speech_embeds = self.down_sample_2(speech_embeds)
        B, T, C = speech_embeds.shape
        speech_embeds, speech_masks = self.speech_transformer(speech_embeds, speech_lens)
        speech_embeds = self.speech_llama_proj(speech_embeds)

        embed_tokens = self.llama_model.model.model.embed_tokens if self.lora else self.llama_model.model.embed_tokens
        prompt_left, prompts_right = self.prompt_pattern.format(prompt).split(
            '<SpeechHere>')
        prompt_left_ids = self.llama_tokenizer(
            prompt_left,
            return_tensors="pt",
            add_special_tokens=False
        ).to(
            speech_embeds.device).input_ids
        prompt_left_embeds = embed_tokens(prompt_left_ids)  # torch.Size([1, 7, 4096])
        prompt_right_ids = self.llama_tokenizer(
            prompts_right,
            return_tensors="pt",
            add_special_tokens=False
        ).to(
            speech_embeds.device).input_ids
        prompt_right_embeds = embed_tokens(prompt_right_ids)  # torch.Size([1, 14, 4096])

        bos_embeds = self.llama_model.model.embed_tokens(
            torch.ones(
                [B, 1],
                dtype=torch.long,
                device=speech_embeds.device,
            ) * self.llama_tokenizer.bos_token_id
        ) if not self.lora else self.llama_model.model.model.embed_tokens(
            torch.ones(
                [B, 1],
                dtype=torch.long,
                device=speech_embeds.device,
            ) * self.llama_tokenizer.bos_token_id
        )  # torch.Size([1, 14, 4096])
        user_embeds = embed_tokens(
            torch.ones(
                [1, 1],
                dtype=torch.long,
                device=speech_embeds.device,
            ) * 195
        )
        assistant_embeds = embed_tokens(
            torch.ones(
                [1, 1],
                dtype=torch.long,
                device=speech_embeds.device,
            ) * 196
        )
        # embeds = torch.cat([bos_embeds, prompt_left_embeds, speech_embeds, prompt_right_embeds], dim=1)
        embeds = torch.cat([user_embeds, prompt_left_embeds, speech_embeds, prompt_right_embeds, assistant_embeds],
                           dim=1)
        atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)
        if self.embed_tokens.weight.dtype == torch.float16:
            logging.info('generate(): self.embed_tokens.weight.dtype == torch.float16')
            embeds = embeds.to(torch.float16)
            atts = atts.half()
        outputs = self.llama_model.generate(
            inputs_embeds=embeds,
            max_new_tokens=self.max_length,
            num_beams=self.num_beams,
            do_sample=self.do_sample,
            min_length=self.min_length,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.length_penalty,
            temperature=self.temperature,
            attention_mask=atts,
            bos_token_id=self.llama_tokenizer.bos_token_id,
            eos_token_id=self.llama_tokenizer.eos_token_id,
            pad_token_id=self.llama_tokenizer.pad_token_id,
        )
        output_text = self.llama_tokenizer.batch_decode(outputs, add_special_tokens=False, skip_special_tokens=True)
        return output_text
