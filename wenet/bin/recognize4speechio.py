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
            "这是涉及经济、货币、金融的演讲或会议内容，转录如下音频。",
            "这是涉及时政的新闻播报，转录如下音频。",
            "这是涉及名人工作、生活的访谈电视节目，转录如下音频。",
            "这是涉及足球解说的专题电视节目，转录如下音频。",
            "这是涉及社会、人文、商业的会场演讲，转录如下音频。",
        ],
        "repeat_prompt": [
            "这是涉及经济、货币、金融的演讲或会议内容，这是涉及经济、货币、金融的演讲或会议内容，这是涉及经济、货币、金融的演讲或会议内容，这是涉及经济、货币、金融的演讲或会议内容，这是涉及经济、货币、金融的演讲或会议内容，这是涉及经济、货币、金融的演讲或会议内容，这是涉及经济、货币、金融的演讲或会议内容，这是涉及经济、货币、金融的演讲或会议内容，这是涉及经济、货币、金融的演讲或会议内容，这是涉及经济、货币、金融的演讲或会议内容，这是涉及经济、货币、金融的演讲或会议内容，这是涉及经济、货币、金融的演讲或会议内容，这是涉及经济、货币、金融的演讲或会议内容，这是涉及经济、货币、金融的演讲或会议内容，转录如下音频。",
            "这是涉及时政的新闻播报，这是涉及时政的新闻播报，这是涉及时政的新闻播报，这是涉及时政的新闻播报，这是涉及时政的新闻播报，这是涉及时政的新闻播报，这是涉及时政的新闻播报，这是涉及时政的新闻播报，这是涉及时政的新闻播报，这是涉及时政的新闻播报，这是涉及时政的新闻播报，这是涉及时政的新闻播报，这是涉及时政的新闻播报，这是涉及时政的新闻播报，这是涉及时政的新闻播报，这是涉及时政的新闻播报，这是涉及时政的新闻播报，这是涉及时政的新闻播报，这是涉及时政的新闻播报，这是涉及时政的新闻播报，这是涉及时政的新闻播报，这是涉及时政的新闻播报，这是涉及时政的新闻播报，这是涉及时政的新闻播报，转录如下音频。",
            "这是涉及名人工作、生活的访谈电视节目，这是涉及名人工作、生活的访谈电视节目，这是涉及名人工作、生活的访谈电视节目，这是涉及名人工作、生活的访谈电视节目，这是涉及名人工作、生活的访谈电视节目，这是涉及名人工作、生活的访谈电视节目，这是涉及名人工作、生活的访谈电视节目，这是涉及名人工作、生活的访谈电视节目，这是涉及名人工作、生活的访谈电视节目，这是涉及名人工作、生活的访谈电视节目，这是涉及名人工作、生活的访谈电视节目，这是涉及名人工作、生活的访谈电视节目，这是涉及名人工作、生活的访谈电视节目，这是涉及名人工作、生活的访谈电视节目，这是涉及名人工作、生活的访谈电视节目，这是涉及名人工作、生活的访谈电视节目，转录如下音频。",
            "这是涉及足球解说的专题电视节目，这是涉及足球解说的专题电视节目，这是涉及足球解说的专题电视节目，这是涉及足球解说的专题电视节目，这是涉及足球解说的专题电视节目，这是涉及足球解说的专题电视节目，这是涉及足球解说的专题电视节目，这是涉及足球解说的专题电视节目，这是涉及足球解说的专题电视节目，这是涉及足球解说的专题电视节目，这是涉及足球解说的专题电视节目，这是涉及足球解说的专题电视节目，这是涉及足球解说的专题电视节目，这是涉及足球解说的专题电视节目，这是涉及足球解说的专题电视节目，这是涉及足球解说的专题电视节目，这是涉及足球解说的专题电视节目，这是涉及足球解说的专题电视节目，这是涉及足球解说的专题电视节目，转录如下音频。",
            "这是涉及社会、人文、商业的会场演讲，这是涉及社会、人文、商业的会场演讲，这是涉及社会、人文、商业的会场演讲，这是涉及社会、人文、商业的会场演讲，这是涉及社会、人文、商业的会场演讲，这是涉及社会、人文、商业的会场演讲，这是涉及社会、人文、商业的会场演讲，这是涉及社会、人文、商业的会场演讲，这是涉及社会、人文、商业的会场演讲，这是涉及社会、人文、商业的会场演讲，这是涉及社会、人文、商业的会场演讲，这是涉及社会、人文、商业的会场演讲，这是涉及社会、人文、商业的会场演讲，这是涉及社会、人文、商业的会场演讲，这是涉及社会、人文、商业的会场演讲，这是涉及社会、人文、商业的会场演讲，这是涉及社会、人文、商业的会场演讲，转录如下音频。",
        ],
        "long_prompt": [
            "我是一个专门针对经济、货币和金融领域内容的语音识别大模型，经过特别训练以精确理解和转录涉及这些主题的演讲和会议内容。我对金融术语和经济数据的表达方式有深入的理解，能够识别和转录与市场动态、货币政策和投资策略相关的专业词汇。我能够处理复杂的数学模型和统计数据，识别它们在口语中的表达方式，如百分比、财务报表和经济预测。我还能够理解和转录经济论述中的特定发音和语调变化，包括专业术语的正确发音和经济数据的口语表达。我能够识别语境下的语调变化，特别是在提问和陈述经济数据时的语调变化。我的目标是准确转录经济、货币和金融领域的语音，并识别其中的特定词汇和短语，以提高语音识别的准确性和理解能力。以下是一系列音频特征的输入，转录如下音频。",
            " 我是一个专门针对时政新闻播报内容的语音识别大模型，经过特别训练以精确理解和转录涉及国内外政治事件、政策变动和国际关系的新闻报道。我对政治术语和时事新闻的表达方式有深入的理解，能够识别和转录与政府声明、立法活动和外交互动相关的专业词汇。我能够处理复杂的政治论述和统计数据，识别它们在口语中的表达方式，如选举结果、民意调查和政策分析。我还能够理解和转录新闻播报中的特定发音和语调变化，包括专业术语的正确发音和时政数据的口语表达。我的目标是准确转录时政新闻播报的语音，并识别其中的特定词汇和短语，以提高语音识别的准确性和理解能力。以下是一系列音频特征的输入，转录如下音频。",
            "我是一个专门针对名人访谈电视节目内容的语音识别大模型，经过特别训练以精确理解和转录涉及名人个人经历、工作成就和公共形象的讨论。我对娱乐圈和公众人物的生活方式有深入的理解，能够识别和转录与艺术创作、演艺活动和社会贡献相关的专业词汇。我能够处理名人故事和个人见解，识别它们在口语中的表达方式，如个人趣事、职业挑战和生活哲学。我还能够理解和转录访谈节目中的特定发音和语调变化，包括个性化的表达方式和情感色彩的口语表达。我的目标是准确转录名人访谈节目的语音，并识别其中的特定词汇和短语，以提高语音识别的准确性和理解能力。以下是一系列音频特征的输入，转录如下音频。",
            " 我是一个专门针对足球解说专题电视节目内容的语音识别大模型，经过特别训练以精确理解和转录涉及足球比赛、球员表现和战术分析的评论。我对足球术语和比赛规则有深入的理解，能够识别和转录与比赛进程、球员动态和教练策略相关的专业词汇。我能够处理实时比赛的激动人心的描述和统计数据，识别它们在口语中的表达方式，如进球数、传球成功率和球队排名。我还能够理解和转录解说中的特定发音和语调变化，包括激情澎湃的表达方式和紧张刺激的口语表达。我的目标是准确转录足球解说专题电视节目的语音，并识别其中的特定词汇和短语，以提高语音识别的准确性和理解能力。以下是一系列音频特征的输入，转录如下音频。",
            " 我是一个专门针对社会、人文、商业领域会场演讲内容的语音识别大模型，经过特别训练以精确理解和转录涉及社会议题、文化交流和商业创新的演讲。我对社会科学术语和商业模式有深入的理解，能够识别和转录与社会变革、文化理念和商业战略相关的专业词汇。我能够处理深刻的社会论述和复杂的商业案例分析，识别它们在口语中的表达方式，如社会统计、文化差异和市场趋势。我还能够理解和转录演讲中的特定发音和语调变化，包括专业术语的正确发音和商业数据的口语表达。我的目标是准确转录社会、人文、商业领域的会场演讲的语音，并识别其中的特定词汇和短语，以提高语音识别的准确性和理解能力。",
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
        "speechio_0": 0,
        "speechio_1": 1,
        "speechio_2": 2,
        "speechio_3": 3,
        "speechio_4": 4,
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
