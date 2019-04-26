import random
import numpy as np
import torch.optim as optim

from model import *
from data_util import DataUtil
from train_util import TrainUtil

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
    hidden_dim = 1024
    dropout = 0.4
    lr = 10
    num_epochs = 40
    grad_clipping = 0.5
    data_dir = 'pretrain_data'
    train_f = 'lm.train'
    dev_f = 'lm.dev'
    file_path = 'pretrain_data'
    vectors = "glove.840B.300d"
    save_path = 'lm.pt'
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
    num_layers = 1
    rnn_dropout_rate = 0
    embed_dropout_rate = 0.4
    rnn_output_dropout_rate = 0.4
    grad_clipping = 10
    lr = 2e-3
    lm_path = 'lm.pt'
    data_dir = 'preprocessed'
    train_fname = 'train-trial-combined-data-processed.json'
    dev_fname = 'dev-data-processed.json'
    test_fname = 'test-data-processed.json'


if __name__ == '__main__':
    config = Config()
    lm_config = LMConfig()
    data_util = DataUtil(config, lm_config, device)
    print('train batches: %d, val batches: %d, test batches: %d' % (len(data_util.train_iter),
                                                                    len(data_util.val_iter),
                                                                    len(data_util.test_iter)))

    model = TriAn(data_util.embedding, data_util.embedding_pos,
                  data_util.embedding_ner, data_util.embedding_rel, config).to(device)

    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adamax(model.parameters(), lr=config.lr, weight_decay=0)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.5)

    train_util = TrainUtil(data_util.train_iter, data_util.val_iter, model,
                           optimizer, criterion, scheduler, device, config)

    train_util.train_model()



