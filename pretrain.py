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
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30], gamma=0.1)


def train_epoch():
    scheduler.step()
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


if config.is_train:
    best_dev_perplex = float('inf')
    best_epoch = -1
    for epoch in range(config.num_epochs):
        cur_lr = optimizer.param_groups[0]['lr']
        train_perplex = train_epoch()
        dev_perplex = eval_epoch()

        if dev_perplex < best_dev_perplex:
            best_dev_perplex = dev_perplex
            best_epoch = epoch
            torch.save(model.state_dict(), config.save_path)
            print('saved best model')

        print('epoch %d, lr %.5f, train_perplex %.4f, dev dev_perplex %.4f' %
              (epoch, cur_lr, train_perplex, dev_perplex))
    print('best perplex %.4f, best epoch %d' % (best_dev_perplex, best_epoch))
# load best model
model.load_state_dict(torch.load(config.save_path))
print('loaded best model')

# see some generations
test_sentences = ['I went into my bedroom and flipped the light switch']
test_sentences = [spacy_tok(sent) for sent in test_sentences]
test_sentences = [[TEXT.vocab.stoi[tok] for tok in sent] for sent in test_sentences]
test_sentences = torch.LongTensor(test_sentences).to(device)

print(test_sentences)
print(test_sentences.shape)


def predict():
    with torch.no_grad():
        decoded, outputs, hidden = model(test_sentences)
        # we only care about the last decoded
        print(decoded.shape)
        last_decoded = decoded[:, -1, :]
        print(last_decoded.shape)

predict()

