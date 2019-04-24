import spacy
from spacy.symbols import ORTH
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext import data, datasets
import os
from model import LM

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print("Use CUDA:", USE_CUDA)


my_tok = spacy.load('en')
my_tok.tokenizer.add_special_case('<unk>', [{ORTH: '<unk>'}])


def spacy_tok(x):
    return [tok.text for tok in my_tok.tokenizer(x)]


class Config:
    batch_size = 32
    bptt_len = 60
    embed_dim = 300
    hidden_dim = 1024
    dropout = 0.5
    lr = 10
    num_epochs = 30
    grad_clipping = 10
    data_dir = 'pretrain_data'
    train_f = 'lm.train'
    dev_f = 'lm.dev'
    file_path = 'pretrain_data'
    vectors = "glove.840B.300d"


config = Config()
TEXT = data.Field(lower=True, tokenize=spacy_tok)
train = datasets.LanguageModelingDataset(os.path.join(config.file_path, config.train_f),
                                         TEXT, newline_eos=False)
dev = datasets.LanguageModelingDataset(os.path.join(config.file_path, config.dev_f),
                                       TEXT, newline_eos=False)


TEXT.build_vocab(train, vectors=config.vectors)
# TEXT.build_vocab(train)
train_iter = data.BPTTIterator(train, batch_size=config.batch_size, bptt_len=config.bptt_len, repeat=False)
dev_iter = data.BPTTIterator(dev, batch_size=config.batch_size, bptt_len=config.bptt_len, repeat=False)

print('train batch num: %d, dev batch num: %d' % (len(train_iter), len(dev_iter)))

# define model
vocab_size = len(TEXT.vocab)
embedding = nn.Embedding(vocab_size, config.embed_dim)
embedding.weight.data.copy_(TEXT.vocab.vectors)
embedding.weight.requires_grad = False
embedding = embedding

model = LM(vocab_size, config.embed_dim, config.hidden_dim, embedding, config.dropout, device)
model = model.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=config.lr)


def train_epoch():
    model.train()
    epoch_losses = []
    for batch in train_iter:
        x, y = batch.text.transpose(0, 1).contiguous().to(device), \
               batch.target.transpose(0, 1).contiguous().to(device)

        optimizer.zero_grad()
        decoded, _, _ = model(x)

        loss = criterion(decoded.view(-1, vocab_size), y.view(-1))
        loss.backward()
        _ = nn.utils.clip_grad_norm_(model.parameters(), config.grad_clipping)
        optimizer.step()

        epoch_losses.append(loss.item())
        # break
    return 2 ** np.mean(epoch_losses)


def eval_epoch():
    model.eval()
    epoch_losses = []
    for batch in dev_iter:
        x, y = batch.text.transpose(0, 1).contiguous().to(device), \
               batch.target.transpose(0, 1).contiguous().to(device)

        with torch.no_grad():
            decoded, _, _ = model(x)
            loss = criterion(decoded.contiguous().view(-1, vocab_size),
                             y.view(-1))
            epoch_losses.append(loss.item())
            # break
    return 2 ** np.mean(epoch_losses)


for epoch in range(config.num_epochs):
    train_perplex = train_epoch()
    dev_perplex = eval_epoch()
    print('epoch %d train_perplex %.4f, dev dev_perplex %.4f' %
          (epoch, train_perplex, dev_perplex))

