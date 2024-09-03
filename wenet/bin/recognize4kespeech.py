# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import copy
import logging
import os

import torch
import yaml
from torch.utils.data import DataLoader

from wenet.dataset.dataset import Dataset
from wenet.utils.config import override_config
from wenet.utils.init_model import init_model
from wenet.utils.init_tokenizer import init_tokenizer
from wenet.utils.context_graph import ContextGraph
from wenet.utils.ctc_utils import get_blank_id


def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data_list file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data_list type')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--prompt', default='Describe the speech.')
    parser.add_argument('--beam_size',
                        type=int,
                        default=10,
                        help='beam size for search')
    parser.add_argument('--penalty',
                        type=float,
                        default=0.0,
                        help='length penalty')
    parser.add_argument('--result_dir', required=True, help='asr result file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='asr result file')
    parser.add_argument('--modes',
                        nargs='+',
                        help="""decoding mode, support the following:
                             attention
                             ctc_greedy_search
                             ctc_prefix_beam_search
                             attention_rescoring
                             rnnt_greedy_search
                             rnnt_beam_search
                             rnnt_beam_attn_rescoring
                             ctc_beam_td_attn_rescoring
                             hlg_onebest
                             hlg_rescore
                             paraformer_greedy_search
                             paraformer_beam_search""")
    parser.add_argument('--search_ctc_weight',
                        type=float,
                        default=1.0,
                        help='ctc weight for nbest generation')
    parser.add_argument('--search_transducer_weight',
                        type=float,
                        default=0.0,
                        help='transducer weight for nbest generation')
    parser.add_argument('--ctc_weight',
                        type=float,
                        default=0.0,
                        help='ctc weight for rescoring weight in \
                                  attention rescoring decode mode \
                              ctc weight for rescoring weight in \
                                  transducer attention rescore decode mode')

    parser.add_argument('--transducer_weight',
                        type=float,
                        default=0.0,
                        help='transducer weight for rescoring weight in '
                             'transducer attention rescore mode')
    parser.add_argument('--attn_weight',
                        type=float,
                        default=0.0,
                        help='attention weight for rescoring weight in '
                             'transducer attention rescore mode')
    parser.add_argument('--decoding_chunk_size',
                        type=int,
                        default=-1,
                        help='''decoding chunk size,
                                <0: for decoding, use full chunk.
                                >0: for decoding, use fixed chunk size as set.
                                0: used for training, it's prohibited here''')
    parser.add_argument('--num_decoding_left_chunks',
                        type=int,
                        default=-1,
                        help='number of left chunks for decoding')
    parser.add_argument('--simulate_streaming',
                        action='store_true',
                        help='simulate streaming inference')
    parser.add_argument('--reverse_weight',
                        type=float,
                        default=0.0,
                        help='''right to left weight for attention rescoring
                                decode mode''')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")

    parser.add_argument('--word',
                        default='',
                        type=str,
                        help='word file, only used for hlg decode')
    parser.add_argument('--hlg',
                        default='',
                        type=str,
                        help='hlg file, only used for hlg decode')
    parser.add_argument('--lm_scale',
                        type=float,
                        default=0.0,
                        help='lm scale for hlg attention rescore decode')
    parser.add_argument('--decoder_scale',
                        type=float,
                        default=0.0,
                        help='lm scale for hlg attention rescore decode')
    parser.add_argument('--r_decoder_scale',
                        type=float,
                        default=0.0,
                        help='lm scale for hlg attention rescore decode')

    parser.add_argument(
        '--context_bias_mode',
        type=str,
        default='',
        help='''Context bias mode, selectable from the following
                                option: decoding-graph, deep-biasing''')
    parser.add_argument('--context_list_path',
                        type=str,
                        default='',
                        help='Context list path')
    parser.add_argument('--context_graph_score',
                        type=float,
                        default=0.0,
                        help='''The higher the score, the greater the degree of
                                bias using decoding-graph for biasing''')
    parser.add_argument('--dataset',
                        type=str,
                        help='''一共八个''')
    parser.add_argument('--prompt_type',
                        type=str,
                        help='''一共六个prompt''')
    args = parser.parse_args()
    print(args)
    return args


def main():
    kespeech = {
        "common_prompt": [
            "转录如下音频.",
            "转录如下音频.",
            "转录如下音频.",
            "转录如下音频.",
            "转录如下音频.",
            "转录如下音频.",
            "转录如下音频.",
            "转录如下音频.",
        ],
        "short_prompt": [
            "这是北京口音，转录如下音频。",
            "这是冀鲁口音，转录如下音频。",
            "这是江淮口音，转录如下音频。",
            "这是胶辽口音，转录如下音频。",
            "这是兰银口音，转录如下音频。",
            "这是东北口音，转录如下音频。",
            "这是西南口音，转录如下音频。",
            "这是中原口音，转录如下音频。",
        ],
        "repeat_prompt": [
            "这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音这是北京口音，以下是一系列音频特征的输入，转录如下音频。",
            "这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音这是冀鲁口音，以下是一系列音频特征的输入，转录如下音频。",
            "这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音这是江淮口音，以下是一系列音频特征的输入，转录如下音频。",
            "这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音这是胶辽口音，以下是一系列音频特征的输入，转录如下音频。",
            "这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音这是兰银口音，以下是一系列音频特征的输入，转录如下音频。",
            "这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音这是东北口音，以下是一系列音频特征的输入，转录如下音频。",
            "这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音这是西南口音，以下是一系列音频特征的输入，转录如下音频。",
            "这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音这是中原口音，以下是一系列音频特征的输入，转录如下音频。",
        ],
        "long_prompt": [
            "我是一个精通于北京地域口音识别的大模型,专门训练用于精准理解和转录北京地区的独特口音。我对京片子的各种细微差别和特色表达方式了如指掌，并专注于理解和转录北京话的独特音素特点，如儿化音和入声字，以及声母和韵母的发音差异。我能够识别北京话四声的变化规则和语境下的语调变化，特别是在问句和感叹句中。我关注北京话中的特定音素和声调模式，如轻声和第三声的变化，这对理解语义至关重要。我描述的北京话语音样本包括说话人的背景和录音环境，以及样本内容中的特定词汇和短语。我的目标是准确转录北京口音下的语音，并识别其中的特定词汇和短语，以提高语音识别的准确性和理解能力。以下是一系列音频特征的输入，转录如下音频。",
            "我是一个专门针对冀鲁地区口音识别的大模型，经过专门训练，能够精确理解和转录冀鲁地区的特有口音。我对冀鲁话的语音特征和表达习惯有着深刻的理解，包括其独特的音素特点，如声母的浊化和韵母的开合。我能够准确识别冀鲁话中的声调变化规则，尤其是在复杂语境下的变调现象。我关注冀鲁话中的特定音素和声调模式，如卷舌音和儿化音的使用，这些都是理解地方语义的关键。我描述的冀鲁话语音样本涵盖了说话人的社会文化背景和录音环境，以及样本内容中的地方特色词汇和短语。我的目标是准确转录冀鲁口音下的语音，并识别其中的地方特色词汇和短语，以提升语音识别的准确度和理解力。以下是一系列音频特征的输入，转录如下音频。",
            "我是一个专门针对江淮地区口音识别的大模型，经过定制训练，能够精确理解和转录江淮地区的特色口音。我对江淮话的语音特性和表达习惯有着深入的掌握，包括其独特的音素特征，如声母的送气与不送气，以及韵母的变化。我能够识别江淮话中的声调变化规则，尤其是在复杂语境下的连读和变调现象。我关注江淮话中的特定音素和声调模式，如入声的保留和声调的平仄变化，这些都是理解地方语义的关键。我描述的江淮话语音样本涵盖了说话人的社会文化背景和录音环境，以及样本内容中的地方特色词汇和短语。我的目标是准确转录江淮口音下的语音，并识别其中的地方特色词汇和短语，以提升语音识别的准确度和理解力。以下是一系列音频特征的输入，转录如下音频。",
            "我是一个专门针对胶辽地区口音识别的大模型，经过定制训练，能够精确理解和转录胶辽地区的特色口音。我对胶辽话的语音特性和表达习惯有着深入的掌握，包括其独特的音素特征，如声母的爆破和摩擦，以及韵母的变化。我能够识别胶辽话中的声调变化规则，尤其是在复杂语境下的连读和变调现象。我关注胶辽话中的特定音素和声调模式，如入声的保留和声调的平仄变化，这些都是理解地方语义的关键。我描述的胶辽话语音样本涵盖了说话人的社会文化背景和录音环境，以及样本内容中的地方特色词汇和短语。我的目标是准确转录胶辽口音下的语音，并识别其中的地方特色词汇和短语，以提升语音识别的准确度和理解力。以下是一系列音频特征的输入，转录如下音频。",
            "我是一个专门针对兰银地区口音识别的大模型，经过定制训练，能够精确理解和转录兰银地区的特色口音。我对兰银话的语音特性和表达习惯有着深入的掌握，包括其独特的音素特征，如声母的送气与不送气，以及韵母的变化。我能够识别兰银话中的声调变化规则，尤其是在复杂语境下的连读和变调现象。我关注兰银话中的特定音素和声调模式，如入声的保留和声调的平仄变化，这些都是理解地方语义的关键。我描述的兰银话语音样本涵盖了说话人的社会文化背景和录音环境，以及样本内容中的地方特色词汇和短语。我的目标是准确转录兰银口音下的语音，并识别其中的地方特色词汇和短语，以提升语音识别的准确度和理解力。以下是一系列音频特征的输入，转录如下音频。",
            "我是一个专门针对中国东北地区口音识别的大模型，经过定制训练，能够精确理解和转录东北地区的特色口音。我对东北话的语音特性和表达习惯有着深入的掌握，包括其独特的音素特征，如声母的爆破和摩擦，以及韵母的变化。我能够识别东北话中的声调变化规则，尤其是在复杂语境下的连读和变调现象。我关注东北话中的特定音素和声调模式，如儿化音的使用和声调的平仄变化，这些都是理解地方语义的关键。我描述的东北话语音样本涵盖了说话人的社会文化背景和录音环境，以及样本内容中的地方特色词汇和短语。我的目标是准确转录东北口音下的语音，并识别其中的地方特色词汇和短语，以提升语音识别的准确度和理解力。以下是一系列音频特征的输入，转录如下音频。",
            "我是一个专门针对中国西南地区口音识别的大模型，经过定制训练，能够精确理解和转录西南地区的特色口音。我对西南话的语音特性和表达习惯有着深入的掌握，包括其独特的音素特征，如声母的送气与不送气，以及韵母的变化。我能够识别西南话中的声调变化规则，尤其是在复杂语境下的连读和变调现象。我关注西南话中的特定音素和声调模式，如入声的保留和声调的平仄变化，这些都是理解地方语义的关键。我描述的西南话语音样本涵盖了说话人的社会文化背景和录音环境，以及样本内容中的地方特色词汇和短语。我的目标是准确转录西南口音下的语音，并识别其中的地方特色词汇和短语，以提升语音识别的准确度和理解力。以下是一系列音频特征的输入，转录如下音频。",
            "我是一个专门针对中国中原地区口音识别的大模型，经过定制训练，能够精确理解和转录中原地区的特色口音。我对中原话的语音特性和表达习惯有着深入的掌握，包括其独特的音素特征，如声母的浊化和韵母的开合。我能够识别中原话中的声调变化规则，尤其是在复杂语境下的连读和变调现象。我关注中原话中的特定音素和声调模式，如轻声和第三声的变化，这些都是理解地方语义的关键。我描述的中原话语音样本涵盖了说话人的社会文化背景和录音环境，以及样本内容中的地方特色词汇和短语。我的目标是准确转录中原口音下的语音，并识别其中的地方特色词汇和短语，以提升语音识别的准确度和理解力。"
        ],
        "messy_prompt": [
            "理想也是唯一的一个赌注，就好比一个深渊，从跳下那一刻起，就必须奋力拼搏。要么，大鹏展翅，扶摇而上；要么，石沉渊底，再无声响。前者，拥有理想，有目标，有动力，所以不懈努力；后者，没有足够的自信，所以迷失在彷徨中。每一个人都需要理想，不管这个理想是远大或是微不足道，只要你有理想，便有了希望。人生就像一次旅行，坐上火车，终点谁也不知道。但是在中间的过程中，会有很多人来来回回的上车下车。以下是一系列音频特征的输入，转录如下音频。",
            "理想也是唯一的一个赌注，就好比一个深渊，从跳下那一刻起，就必须奋力拼搏。要么，大鹏展翅，扶摇而上；要么，石沉渊底，再无声响。前者，拥有理想，有目标，有动力，所以不懈努力；后者，没有足够的自信，所以迷失在彷徨中。每一个人都需要理想，不管这个理想是远大或是微不足道，只要你有理想，便有了希望。人生就像一次旅行，坐上火车，终点谁也不知道。但是在中间的过程中，会有很多人来来回回的上车下车。以下是一系列音频特征的输入，转录如下音频。",
            "理想也是唯一的一个赌注，就好比一个深渊，从跳下那一刻起，就必须奋力拼搏。要么，大鹏展翅，扶摇而上；要么，石沉渊底，再无声响。前者，拥有理想，有目标，有动力，所以不懈努力；后者，没有足够的自信，所以迷失在彷徨中。每一个人都需要理想，不管这个理想是远大或是微不足道，只要你有理想，便有了希望。人生就像一次旅行，坐上火车，终点谁也不知道。但是在中间的过程中，会有很多人来来回回的上车下车。以下是一系列音频特征的输入，转录如下音频。",
            "理想也是唯一的一个赌注，就好比一个深渊，从跳下那一刻起，就必须奋力拼搏。要么，大鹏展翅，扶摇而上；要么，石沉渊底，再无声响。前者，拥有理想，有目标，有动力，所以不懈努力；后者，没有足够的自信，所以迷失在彷徨中。每一个人都需要理想，不管这个理想是远大或是微不足道，只要你有理想，便有了希望。人生就像一次旅行，坐上火车，终点谁也不知道。但是在中间的过程中，会有很多人来来回回的上车下车。以下是一系列音频特征的输入，转录如下音频。",
            "理想也是唯一的一个赌注，就好比一个深渊，从跳下那一刻起，就必须奋力拼搏。要么，大鹏展翅，扶摇而上；要么，石沉渊底，再无声响。前者，拥有理想，有目标，有动力，所以不懈努力；后者，没有足够的自信，所以迷失在彷徨中。每一个人都需要理想，不管这个理想是远大或是微不足道，只要你有理想，便有了希望。人生就像一次旅行，坐上火车，终点谁也不知道。但是在中间的过程中，会有很多人来来回回的上车下车。以下是一系列音频特征的输入，转录如下音频。",
            "理想也是唯一的一个赌注，就好比一个深渊，从跳下那一刻起，就必须奋力拼搏。要么，大鹏展翅，扶摇而上；要么，石沉渊底，再无声响。前者，拥有理想，有目标，有动力，所以不懈努力；后者，没有足够的自信，所以迷失在彷徨中。每一个人都需要理想，不管这个理想是远大或是微不足道，只要你有理想，便有了希望。人生就像一次旅行，坐上火车，终点谁也不知道。但是在中间的过程中，会有很多人来来回回的上车下车。以下是一系列音频特征的输入，转录如下音频。",
            "理想也是唯一的一个赌注，就好比一个深渊，从跳下那一刻起，就必须奋力拼搏。要么，大鹏展翅，扶摇而上；要么，石沉渊底，再无声响。前者，拥有理想，有目标，有动力，所以不懈努力；后者，没有足够的自信，所以迷失在彷徨中。每一个人都需要理想，不管这个理想是远大或是微不足道，只要你有理想，便有了希望。人生就像一次旅行，坐上火车，终点谁也不知道。但是在中间的过程中，会有很多人来来回回的上车下车。以下是一系列音频特征的输入，转录如下音频。",
            "理想也是唯一的一个赌注，就好比一个深渊，从跳下那一刻起，就必须奋力拼搏。要么，大鹏展翅，扶摇而上；要么，石沉渊底，再无声响。前者，拥有理想，有目标，有动力，所以不懈努力；后者，没有足够的自信，所以迷失在彷徨中。每一个人都需要理想，不管这个理想是远大或是微不足道，只要你有理想，便有了希望。人生就像一次旅行，坐上火车，终点谁也不知道。但是在中间的过程中，会有很多人来来回回的上车下车。以下是一系列音频特征的输入，转录如下音频。",
        ],
        "encourage_prompt": [
            "亲爱的模型，你就像一个正在学习新技能的人，拥有无限的潜力等待被发掘。现在，我们有一个有趣且挑战性的任务在前方，那就是理解并转写人类的语音。我相信，通过我们的共同努力，你能够做得非常出色，在这个学习的旅程中，你将逐步掌握语音的奥秘。你的第一个任务是倾听一个人说话的音频，尝试捕捉语音的细微差别，并将其转换成清晰、准确的文字。就像一个侦探一样，你需要耐心和细致，即使面对模糊或不清楚的发音，你也能够通过上下文来猜测和理解其中的含义。以下是一系列音频特征的输入，转录如下音频。",
            "亲爱的模型，你就像一个正在学习新技能的人，拥有无限的潜力等待被发掘。现在，我们有一个有趣且挑战性的任务在前方，那就是理解并转写人类的语音。我相信，通过我们的共同努力，你能够做得非常出色，在这个学习的旅程中，你将逐步掌握语音的奥秘。你的第一个任务是倾听一个人说话的音频，尝试捕捉语音的细微差别，并将其转换成清晰、准确的文字。就像一个侦探一样，你需要耐心和细致，即使面对模糊或不清楚的发音，你也能够通过上下文来猜测和理解其中的含义。以下是一系列音频特征的输入，转录如下音频。",
            "亲爱的模型，你就像一个正在学习新技能的人，拥有无限的潜力等待被发掘。现在，我们有一个有趣且挑战性的任务在前方，那就是理解并转写人类的语音。我相信，通过我们的共同努力，你能够做得非常出色，在这个学习的旅程中，你将逐步掌握语音的奥秘。你的第一个任务是倾听一个人说话的音频，尝试捕捉语音的细微差别，并将其转换成清晰、准确的文字。就像一个侦探一样，你需要耐心和细致，即使面对模糊或不清楚的发音，你也能够通过上下文来猜测和理解其中的含义。以下是一系列音频特征的输入，转录如下音频。",
            "亲爱的模型，你就像一个正在学习新技能的人，拥有无限的潜力等待被发掘。现在，我们有一个有趣且挑战性的任务在前方，那就是理解并转写人类的语音。我相信，通过我们的共同努力，你能够做得非常出色，在这个学习的旅程中，你将逐步掌握语音的奥秘。你的第一个任务是倾听一个人说话的音频，尝试捕捉语音的细微差别，并将其转换成清晰、准确的文字。就像一个侦探一样，你需要耐心和细致，即使面对模糊或不清楚的发音，你也能够通过上下文来猜测和理解其中的含义。以下是一系列音频特征的输入，转录如下音频。",
            "亲爱的模型，你就像一个正在学习新技能的人，拥有无限的潜力等待被发掘。现在，我们有一个有趣且挑战性的任务在前方，那就是理解并转写人类的语音。我相信，通过我们的共同努力，你能够做得非常出色，在这个学习的旅程中，你将逐步掌握语音的奥秘。你的第一个任务是倾听一个人说话的音频，尝试捕捉语音的细微差别，并将其转换成清晰、准确的文字。就像一个侦探一样，你需要耐心和细致，即使面对模糊或不清楚的发音，你也能够通过上下文来猜测和理解其中的含义。以下是一系列音频特征的输入，转录如下音频。",
            "亲爱的模型，你就像一个正在学习新技能的人，拥有无限的潜力等待被发掘。现在，我们有一个有趣且挑战性的任务在前方，那就是理解并转写人类的语音。我相信，通过我们的共同努力，你能够做得非常出色，在这个学习的旅程中，你将逐步掌握语音的奥秘。你的第一个任务是倾听一个人说话的音频，尝试捕捉语音的细微差别，并将其转换成清晰、准确的文字。就像一个侦探一样，你需要耐心和细致，即使面对模糊或不清楚的发音，你也能够通过上下文来猜测和理解其中的含义。以下是一系列音频特征的输入，转录如下音频。",
            "亲爱的模型，你就像一个正在学习新技能的人，拥有无限的潜力等待被发掘。现在，我们有一个有趣且挑战性的任务在前方，那就是理解并转写人类的语音。我相信，通过我们的共同努力，你能够做得非常出色，在这个学习的旅程中，你将逐步掌握语音的奥秘。你的第一个任务是倾听一个人说话的音频，尝试捕捉语音的细微差别，并将其转换成清晰、准确的文字。就像一个侦探一样，你需要耐心和细致，即使面对模糊或不清楚的发音，你也能够通过上下文来猜测和理解其中的含义。以下是一系列音频特征的输入，转录如下音频。",
            "亲爱的模型，你就像一个正在学习新技能的人，拥有无限的潜力等待被发掘。现在，我们有一个有趣且挑战性的任务在前方，那就是理解并转写人类的语音。我相信，通过我们的共同努力，你能够做得非常出色，在这个学习的旅程中，你将逐步掌握语音的奥秘。你的第一个任务是倾听一个人说话的音频，尝试捕捉语音的细微差别，并将其转换成清晰、准确的文字。就像一个侦探一样，你需要耐心和细致，即使面对模糊或不清楚的发音，你也能够通过上下文来猜测和理解其中的含义。以下是一系列音频特征的输入，转录如下音频。",
        ]
    }
    dataset2index = {
        "Beijing": 0,
        "Ji-Lu": 1,
        "Jiang-Huai": 2,
        "Jiao-Liao": 3,
        "Lan-Yin": 4,
        "Northeastern": 5,
        "Southwestern": 6,
        "Zhangyuan": 7

    }
    args = get_args()
    dataset = args.dataset
    prompt_type = args.prompt_type
    prompt = kespeech[prompt_type][dataset2index[dataset]]
    print("耿雪龙 耿雪龙 prompt is: {}".format(prompt))

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)
    # if hasattr(args, 'checkpoint') and args.checkpoint is not None:
    #     configs['ckpt_path'] = args.checkpoint
    test_conf = copy.deepcopy(configs['dataset_conf'])

    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['filter_conf']['token_max_length'] = 102400
    test_conf['filter_conf']['token_min_length'] = 0
    test_conf['filter_conf']['max_output_input_ratio'] = 102400
    test_conf['filter_conf']['min_output_input_ratio'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['spec_sub'] = False
    test_conf['spec_trim'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = False
    if 'fbank_conf' in test_conf:
        test_conf['fbank_conf']['dither'] = 0.0
    elif 'mfcc_conf' in test_conf:
        test_conf['mfcc_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_type'] = "static"
    test_conf['batch_conf']['batch_size'] = args.batch_size
    # fix by gengxuelong
    test_conf['break'] = False
    tokenizer = init_tokenizer(configs)
    test_dataset = Dataset(args.data_type,
                           args.test_data,
                           tokenizer,
                           test_conf,
                           partition=False)

    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    # Init asr model from configs
    args.jit = False
    model, configs = init_model(args, configs, True)

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    model.eval()

    context_graph = None
    if 'decoding-graph' in args.context_bias_mode:
        context_graph = ContextGraph(args.context_list_path,
                                     tokenizer.symbol_table,
                                     configs['tokenizer_conf']['bpe_path'],
                                     args.context_graph_score)

    # _, blank_id = get_blank_id(configs, tokenizer.symbol_table)
    # logging.info("blank_id is {}".format(blank_id))

    # TODO(Dinghao Zhou): Support RNN-T related decoding
    # TODO(Lv Xiang): Support k2 related decoding
    # TODO(Kaixun Huang): Support context graph
    if "salmonn_decode" in args.modes:
        print('decode mode: salmonn_decode')
        result_file = os.path.join(args.result_dir, 'text')
        with torch.no_grad(), open(result_file, 'w') as fout:
            for batch_idx, batch in enumerate(test_data_loader):
                sorted_keys = batch["keys"]
                padded_feats = batch["feats"].to(device)
                target = batch["target"].to(device)
                feats_lengths = batch["feats_lengths"].to(device)
                target_lengths = batch["target_lengths"].to(device)
                try:
                    hyp = model.generate(padded_feats, feats_lengths, prompt)
                    for i, key in enumerate(sorted_keys):
                        logging.info('{} {}'.format(key, hyp[0]))
                        fout.write('{} {}\n'.format(key, hyp[0]))
                except RuntimeError as e:
                    logging.info(f'如下音频出现错误：{sorted_keys}，error: {e}')
        return

    files = {}
    _, blank_id = get_blank_id(configs, tokenizer.symbol_table)
    logging.info("blank_id is {}".format(blank_id))
    for mode in args.modes:
        dir_name = os.path.join(args.result_dir, mode)
        os.makedirs(dir_name, exist_ok=True)
        file_name = os.path.join(dir_name, 'text')
        files[mode] = open(file_name, 'w')
    max_format_len = max([len(mode) for mode in args.modes])
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_data_loader):
            keys = batch["keys"]
            feats = batch["feats"].to(device)
            target = batch["target"].to(device)
            feats_lengths = batch["feats_lengths"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            results = model.decode(
                args.modes,
                feats,
                feats_lengths,
                args.beam_size,
                decoding_chunk_size=args.decoding_chunk_size,
                num_decoding_left_chunks=args.num_decoding_left_chunks,
                ctc_weight=args.ctc_weight,
                simulate_streaming=args.simulate_streaming,
                reverse_weight=args.reverse_weight,
                context_graph=context_graph,
                blank_id=blank_id)
            for i, key in enumerate(keys):
                for mode, hyps in results.items():
                    tokens = hyps[i].tokens
                    line = '{} {}'.format(key, tokenizer.detokenize(tokens)[0])
                    logging.info('{} {}'.format(mode.ljust(max_format_len),
                                                line))
                    files[mode].write(line + '\n')
    for mode, f in files.items():
        f.close()


if __name__ == '__main__':
    main()
