import sys
sys.path.insert(0, '../../')
from wenet.transformer.hubert_encoder import S3prlFrontend
from gxl_ai_utils.utils import utils_file

configs = utils_file.load_dict_from_yaml("./test.yaml")
encoder = S3prlFrontend(**configs['encoder_conf'])
print(encoder)
