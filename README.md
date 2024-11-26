# wenet_LLM_from_ASLP

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python-Version](https://img.shields.io/badge/Python-3.7%7C3.8-brightgreen)](https://github.com/wenet-e2e/wenet)


## Install

### Install python package

``` sh
pip install https://github.com/gengxuelong/wenet_LLM_from_ASLP.git
cd wenet_LLM_from_ASLP
pip install -r requirements.txt
```

## paper 
[**Unveiling the Potential of LLM-Based ASR on Chinese Open-Source Datasets**](https://arxiv.org/abs/2405.02132)

This codebase is the concrete implementation of our paper "Unveiling the Potential of LLM-Based ASR on Chinese Open-Source Datasets."

## recognize for our best ckpt model in paper
the model can be downloaded from [here](https://pan.baidu.com/s/1-iz-xSdZwa0AFojWBU2l1g?pwd=n7ub) by **Baidu Cloud**.
```commandline
cd examples/ASLP_ASRLLM
ln -s ../../wenet .
ln -s ../../tools .
```
Next, modify the paths in conf/train_ASLP_ASRLLM.yaml, 
where the Baichuan2-7B-chat used for initialization can be obtained [here](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat). The original Chinese HuBERT can be obtained [here](https://cloud.tencent.com/developer/article/2017032). 
HuBERT needs to be converted to the format required by S3PRL. The specific steps are as follows:
```python
source_fairseq_path =""
s3qrl_path = ""
from s3prl.upstream.hubert.convert import load_and_convert_fairseq_ckpt
load_and_convert_fairseq_ckpt(source_fairseq_path, s3qrl_path)
```

Next, set the **decode_checkpoint** variable in recognize.sh to the path of the checkpoint you downloaded from Baidu Cloud.

The test set used for inference follows the same format as WeNet. The input is a data.list file in JSONL format, where each line is a JSON object containing the audio file path, corresponding text, and key value.

## Acknowledge

We borrowed a lot of code from [WeNet 2.0](https://github.com/wenet-e2e/wenet) for transformer based modeling.

## Citations

``` bibtex
@inproceedings{yao2021wenet,
  title={WeNet: Production oriented Streaming and Non-streaming End-to-End Speech Recognition Toolkit},
  author={Yao, Zhuoyuan and Wu, Di and Wang, Xiong and Zhang, Binbin and Yu, Fan and Yang, Chao and Peng, Zhendong and Chen, Xiaoyu and Xie, Lei and Lei, Xin},
  booktitle={Proc. Interspeech},
  year={2021},
  address={Brno, Czech Republic },
  organization={IEEE}
}

@article{zhang2022wenet,
  title={WeNet 2.0: More Productive End-to-End Speech Recognition Toolkit},
  author={Zhang, Binbin and Wu, Di and Peng, Zhendong and Song, Xingchen and Yao, Zhuoyuan and Lv, Hang and Xie, Lei and Yang, Chao and Pan, Fuping and Niu, Jianwei},
  journal={arXiv preprint arXiv:2203.15455},
  year={2022}
}
```
