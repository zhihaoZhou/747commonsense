import random
import numpy as np
import torch.optim as optim
import sys

from model import *
from data_util import DataUtil
from train_util import TrainUtil
from lm_train_util import LMTrainUtil

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print("Use CUDA:", USE_CUDA)

seed = 1234
# Set the random seed manually for reproducibility.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class LMConfig:
    batch_size = 32
    bptt_len = 60
    embed_dim = 300
    # hidden_dim = 1024
    hidden_dim = 256
    dropout = 0.4
    lr = 10
    num_epochs = 30
    grad_clipping = 0.5
    data_dir = 'pretrain_data'
    train_f = 'lm.train'
    dev_f = 'lm.dev'
    file_path = 'pretrain_data'
    save_path = 'lm.pt'
    milestones = []
    gamma = 0.1
    is_train = False


class Config:
    num_epoch = 60
    batch_size_train = 32
    batch_size_eval = 256
    batch_size_test = 256
    embed_dim = 300
    embed_dim_pos = 12
    embed_dim_ner = 8
    embed_dim_value = 1
    embed_dim_rel = 10
    vectors = "glove.840B.300d"
    hidden_size = 96
    # hidden_size = 200
    num_layers = 1
    rnn_dropout_rate = 0
    embed_dropout_rate = 0.4
    rnn_output_dropout_rate = 0.4
    grad_clipping = 10
    lr = 2e-3
    milestones = [10, 15]
    gamma = 0.5
    lm_path = 'lm.pt'
    data_dir = 'preprocessed'
    train_fname = 'train-trial-combined-data-processed.json'
    dev_fname = 'dev-data-processed.json'
    test_fname = 'test-data-processed.json'


modes = ['train_lm', 'test_lm', 'train_tri-an']

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        print('add one of the following arguments:', modes)
    else:
        mode = sys.argv[1]
        print('mode is', mode)

        config = Config()
        lm_config = LMConfig()
        data_util = DataUtil(config, lm_config, device)

        # define language model
        lm = LM(data_util.vocab_size, lm_config.embed_dim, lm_config.hidden_dim, data_util.embedding,
                lm_config.dropout, device).to(device)

        lm_train_util = LMTrainUtil(data_util.lm_train_iter, data_util.lm_dev_iter, lm, device, lm_config,
                                    data_util.vocab_size, data_util.TEXT)

        if mode == 'train_lm':
            # train language model
            lm_train_util.train_model()
        elif mode == 'test_lm':
            lm_train_util.generate()
        elif mode == 'train_tri-an':
            # define tri-an model

            # model = TriAn(data_util.embedding, data_util.embedding_pos,
            #               data_util.embedding_ner, data_util.embedding_rel, config, device).to(device)

            # load trained lm
            lm_train_util.generate()
            model = TriAnWithLM(data_util.embedding, lm, data_util.embedding_pos,
                                data_util.embedding_ner, data_util.embedding_rel, config, lm_config, device).to(device)
            # model = TriAnWithLMMultiHop(data_util.embedding, lm, data_util.embedding_pos,
            #                     data_util.embedding_ner, data_util.embedding_rel, config, lm_config, device).to(device)

            # train tri-an model
            train_util = TrainUtil(data_util.train_iter, data_util.val_iter, model,
                                   device, config, data_util.TEXT)
            train_util.train_model()
