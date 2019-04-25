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


def get_pretrained_lm():
    global config
    lm_vocab_size = 9689
    lm = LM(lm_vocab_size, config.embed_dim, 1024,
            nn.Embedding(lm_vocab_size, config.embed_dim), 0, device).to(device)
    lm.load_state_dict(torch.load(config.lm_path))
    print('loaded best model')
    lm.eval()
    lm.parameters().requires_grad = False
    return lm


if __name__ == '__main__':
    config = Config()

    data_dir = 'preprocessed'
    combined_fname = 'all-combined-data-processed.json'
    train_fname = 'train-trial-combined-data-processed.json'
    dev_fname = 'dev-data-processed.json'
    test_fname = 'test-data-processed.json'

    data_util = DataUtil(data_dir, combined_fname, train_fname, dev_fname,
                         test_fname, config, device)
    print('train batches: %d, val batches: %d, test batches: %d' % (len(data_util.train_iter),
                                                                    len(data_util.val_iter),
                                                                    len(data_util.test_iter)))

    lm = get_pretrained_lm()
    print('lm got!!')

    model = TriAn(data_util.embedding, data_util.embedding_pos,
                  data_util.embedding_ner, data_util.embedding_rel, config).to(device)

    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adamax(model.parameters(), lr=config.lr, weight_decay=0)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.5)

    train_util = TrainUtil(data_util.train_iter, data_util.val_iter, model,
                           optimizer, criterion, scheduler, device, config)

    train_util.train_model()



