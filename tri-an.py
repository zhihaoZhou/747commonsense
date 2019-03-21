
# coding: utf-8

# In[53]:


import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from torchtext import data
import spacy
import os
import time
import sys
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(USE_CUDA)


# # Hyperparameters

# In[54]:


num_epoch = 60
batch_size_train = 32
batch_size_eval = 256
embed_dim = 300
# embed_from = "glove.840B.300d"
hidden_size = 96
num_layers = 1
rnn_dropout_rate = 0
embed_dropout_rate = 0.4
rnn_output_dropout_rate = 0.4
grad_clipping = 10
lr = 2e-3


# # Load data
# refer to 
# 
# http://anie.me/On-Torchtext/
# 
# http://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/
# 
# http://mlexplained.com/2018/02/15/language-modeling-tutorial-in-torchtext-practical-torchtext-part-2/

# In[55]:


data_dir = 'preprocessed'
combined_fname = 'all-combined-data-processed.json'
train_fname = 'train-trial-combined-data-processed.json'
dev_fname = 'dev-data-processed.json'
test_fname = 'test-data-processed.json'


# we have keys: 'id', 'd_words', 'd_pos', 'd_ner', 'q_words', 'q_pos', 'c_words', 'label', 'in_q', 'in_c', 'lemma_in_q', 'tf', 'p_q_relation', 'p_c_relation'

# In[56]:


TEXT = data.ReversibleField(sequential=True, lower=False, include_lengths=True)

train, val, test = data.TabularDataset.splits(
    path=data_dir, train=train_fname,
    validation=dev_fname, test=test_fname, format='json',
    fields={'d_words': ('d_words', TEXT),
            'q_words': ('q_words', TEXT),
            'c_words': ('c_words', TEXT),
            'label': ('label', data.Field(sequential=False, use_vocab=False))
           })

print('train: %d, val: %d, test: %d' % (len(train), len(val), len(test)))


# In[57]:


# combined is only used for building vocabulary
combined = data.TabularDataset(
    path=os.path.join(data_dir, combined_fname), format='json',
    fields={'d_words': ('d_words', TEXT),
            'q_words': ('q_words', TEXT),
            'c_words': ('c_words', TEXT),
            'label': ('label', data.Field(sequential=False, use_vocab=False))
           })

# specify the path to the localy saved vectors
vec = torchtext.vocab.Vectors('glove.840B.300d.txt', data_dir)
# TEXT.build_vocab(combined, vectors=embed_from)
TEXT.build_vocab(combined, vectors=vec)
print('vocab size: %d' % len(TEXT.vocab))


# In[58]:


train_iter, val_iter, test_iter = data.Iterator.splits(
        (train, val, test), batch_sizes=(batch_size_train, batch_size_eval, batch_size_eval), \
    sort_key=lambda x: len(x.d_words), device=device, sort_within_batch=False, repeat=False)

print('train batches: %d, val batches: %d, test batches: %d' % (len(train_iter),                                                                 len(val_iter), len(test_iter)))


# # Create embedding

# In[59]:


embedding = nn.Embedding(len(TEXT.vocab), embed_dim)
embedding.weight.data.copy_(TEXT.vocab.vectors)
embedding.weight.requires_grad=False
embedding = embedding.to(device)


# In[60]:


embedding.weight.shape


# # Build model
# refer to
# 
# https://github.com/intfloat/commonsense-rc
# 
# https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
# 
# https://discuss.pytorch.org/t/solved-multiple-packedsequence-input-ordering/2106/23

# In[61]:


class BLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, rnn_output_dropout_rate):
        super(BLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.rnn_output_dropout = nn.Dropout(rnn_output_dropout_rate)
    
    def forward(self, inputs, lengths):
        """
        take inputs (embedded and padded), return outputs from lstm
        
        :param inputs: (batch_size, seq_len, embed_dim)
        :param lengths: (batch_size)
        :return: (batch_size, seq_len, hidden_size * 2)
        """
        lengths_sorted, sorted_idx = lengths.sort(descending=True)
        inputs_sorted = inputs[sorted_idx]
    
        inputs_packed = nn.utils.rnn.pack_padded_sequence(inputs_sorted, lengths_sorted.tolist(), batch_first=True)
        outputs_packed, _ = self.lstm(inputs_packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs_packed, batch_first=True)
        
        # Reverses sorting. 
        outputs = torch.zeros_like(outputs)            .scatter_(0, sorted_idx.unsqueeze(1).unsqueeze(1)
                      .expand(-1, outputs.shape[1], outputs.shape[2]), outputs)
        outputs = self.rnn_output_dropout(outputs)
        
        return outputs


# In[62]:


def lengths_to_mask(lengths, dtype=torch.uint8):
    """
    
    :param lengths: (batch_size)
    :param dtype: 
    :return: (batch_size, max_len)
    """
    
    lengths = lengths.cpu()
    
    max_len = lengths.max().item()
    mask = torch.arange(max_len,
                        dtype=lengths.dtype).expand(len(lengths), max_len) < lengths.unsqueeze(1)

    mask = torch.as_tensor(mask, dtype=dtype, device=device)
    mask = 1 - mask
    return mask


# In[63]:


class SeqAttnContext(nn.Module):
    def __init__(self, embed_dim):
        super(SeqAttnContext, self).__init__()
        
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )
        
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, y, y_mask):
        """
        calculate context vectors for x on y using attention on y
        
        :param x: (batch_size, x_seq_len, embed_dim)
        :param y: (batch_size, y_seq_len, embed_dim)
        :param y_lengths: (batch_size)
        :return: (batch_size, x_seq_len, embed_dim)
        """
        x_proj = self.proj(x)
        y_proj = self.proj(y)
        
        scores = x_proj.bmm(y_proj.transpose(2, 1))
        
        # mask scores
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        
        scores.data.masked_fill_(y_mask.data, -float('inf'))
        weights = self.softmax(scores)
        
        # Take weighted average
        contexts = weights.bmm(y)
        # here, instead of using y, maybe use another projection of y in the future
        
        return contexts


# In[64]:


class BilinearAttnEncoder(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(BilinearAttnEncoder, self).__init__()
        self.linear = nn.Linear(y_dim, x_dim)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x, y, x_mask):
        """
        summarize x into single vectors using bilinear attention on y
        
        :param x: (batch_size, seq_len, x_dim)
        :param y: (batch_size, y_dim)
        :param x_mask: (batch_size, seq_len)
        :return: 
        """
        y_proj = self.linear(y).unsqueeze(2)  # (batch_size, x_dim, 1)
        scores = x.bmm(y_proj).squeeze(2)
        
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        weights = self.softmax(scores) # (batch_size, seq_len)
        
        return weights.unsqueeze(1).bmm(x).squeeze(1)


# In[65]:


class SelfAttnEncoder(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttnEncoder, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, inputs, mask):
        """
        Summarize inputs into single vectors using self attention
        
        :param self: 
        :param inputs: (batch_size, seq_len, input_dim)
        :param mask: (batch_size, seq_len)
        :return: (batch_size, input_dim)
        """
        scores = self.linear(inputs).squeeze(2)
        scores.data.masked_fill_(mask.data, -float('inf'))
        weights = self.softmax(scores) # (batch_size, seq_len)
        
        return weights.unsqueeze(1).bmm(inputs).squeeze(1)


# In[66]:


class Bilinear(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Bilinear, self).__init__()
        self.linear = nn.Linear(x_dim, y_dim)
    
    def forward(self, x, y):
        """
        Calculate the biliear function x*W*y
        
        :param x: (batch_size, x_dim)
        :param y: (batch_size, y_dim)
        :return: (batch_size)
        """
        xW = self.linear(x)  # (batch_size, y_dim)
        return xW.unsqueeze(1).bmm(y.unsqueeze(2)).view(-1)


# In[67]:


class TriAn(nn.Module):
    def __init__(self, embedding):
        super(TriAn, self).__init__()
        self.embedding = embedding
        self.d_rnn = BLSTM(embed_dim * 2, hidden_size, num_layers, rnn_dropout_rate)
        self.q_rnn = BLSTM(embed_dim, hidden_size, num_layers, rnn_dropout_rate)
        self.c_rnn = BLSTM(embed_dim * 3, hidden_size, num_layers, rnn_dropout_rate)
        
        self.embed_dropout = nn.Dropout(embed_dropout_rate)
        
        self.d_on_q_attn = SeqAttnContext(embed_dim)
        self.c_on_q_attn = SeqAttnContext(embed_dim)
        self.c_on_d_attn = SeqAttnContext(embed_dim)
        
        self.d_on_q_encode = BilinearAttnEncoder(hidden_size * 2, hidden_size * 2)
        self.q_encode = SelfAttnEncoder(hidden_size * 2)
        self.c_encode = SelfAttnEncoder(hidden_size * 2)
        
        self.d_c_bilinear = Bilinear(hidden_size * 2, hidden_size * 2)
        self.q_c_bilinear = Bilinear(hidden_size * 2, hidden_size * 2)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, d_words, d_lengths, q_words, q_lengths, c_words, c_lengths):
        # embed inputs
        d_embed, q_embed, c_embed = self.embedding(d_words),             self.embedding(q_words), self.embedding(c_words)
        d_embed, q_embed, c_embed = self.embed_dropout(d_embed), self.embed_dropout(q_embed),            self.embed_dropout(c_embed)
        
        # get masks
        d_mask = lengths_to_mask(d_lengths)
        q_mask = lengths_to_mask(q_lengths)
        c_mask = lengths_to_mask(c_lengths)
        
        # get attention contexts
        d_on_q_contexts = self.embed_dropout(self.d_on_q_attn(d_embed, q_embed, q_mask))
        c_on_q_contexts = self.embed_dropout(self.c_on_q_attn(c_embed, q_embed, q_mask))
        c_on_d_contexts = self.embed_dropout(self.c_on_d_attn(c_embed, d_embed, d_mask))
        
        # form final inputs for rnns
        d_rnn_inputs = torch.cat([d_embed, d_on_q_contexts], dim=2)
        q_rnn_inputs = torch.cat([q_embed], dim=2)
        c_rnn_inputs = torch.cat([c_embed, c_on_q_contexts, c_on_d_contexts], dim=2)
        
        # calculate rnn outputs
        d_rnn_outputs = self.d_rnn(d_rnn_inputs, d_lengths)
        q_rnn_outputs = self.q_rnn(q_rnn_inputs, q_lengths)
        c_rnn_outputs = self.c_rnn(c_rnn_inputs, c_lengths)        
        
        # get final representations
        q_rep = self.q_encode(q_rnn_outputs, q_mask)
        c_rep = self.c_encode(c_rnn_outputs, c_mask)
        d_rep = self.d_on_q_encode(d_rnn_outputs, q_rep, d_mask)
        
        dWc = self.d_c_bilinear(d_rep, c_rep)
        qWc = self.q_c_bilinear(q_rep, c_rep)
        
        logits = dWc + qWc
        return self.sigmoid(logits)


# In[68]:


model = TriAn(embedding).to(device)

criterion = nn.BCELoss().to(device)
optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=0)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15], gamma=0.5)


# # Train and test model

# In[69]:


def get_accuaracy(outputs, labels):
    preds = (outputs > 0.5).float()
    acc = torch.mean((preds==labels).float())
    return acc


# In[70]:


def parse_batch(batch):
    d_words, d_lengths = batch.d_words
    q_words, q_lengths = batch.q_words
    c_words, c_lengths = batch.c_words

    d_words, d_lengths = torch.transpose(d_words, 0, 1), d_lengths
    q_words, q_lengths = torch.transpose(q_words, 0, 1), q_lengths
    c_words, c_lengths = torch.transpose(c_words, 0, 1), c_lengths

    labels = batch.label.float()
    
    return d_words, d_lengths, q_words, q_lengths, c_words, c_lengths, labels


# In[71]:


def train_epoch():
    scheduler.step()
    model.train()
    
    epoch_losses = []
    epoch_accus = []
    
    for i, batch in enumerate(train_iter):
        # get batch
        d_words, d_lengths, q_words, q_lengths,             c_words, c_lengths, labels = parse_batch(batch)
        
        # get outputs and loss
        optimizer.zero_grad()
        outputs = model(d_words, d_lengths, q_words, q_lengths, c_words, c_lengths)
        loss = criterion(outputs, labels)
        
        # update model
        loss.to(device)
        loss.backward()
        
        _ = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clipping)
        optimizer.step()
        
        # record losses and accuracies
        epoch_losses.append(loss.item())
        accu = get_accuaracy(outputs, labels)
        epoch_accus.append(accu.item())
#         break
    
    accu_avg = np.mean(np.array(epoch_accus))
    loss_avg = np.mean(np.array(epoch_losses))
    
    return accu_avg, loss_avg


# In[72]:


def eval_epoch():
    model.eval()
    
    epoch_losses = []
    epoch_accus = []
    
    for i, batch in enumerate(val_iter):
        # get batch
        d_words, d_lengths, q_words, q_lengths,             c_words, c_lengths, labels = parse_batch(batch)
        
        # eval
        with torch.no_grad():
            outputs = model(d_words, d_lengths, q_words, q_lengths, c_words, c_lengths)
            loss = criterion(outputs, labels)
            # record losses and accuracies
            epoch_losses.append(loss.item())
            accu = get_accuaracy(outputs, labels)
            epoch_accus.append(accu.item())
#             break
    
    accu_avg = np.mean(np.array(epoch_accus))
    loss_avg = np.mean(np.array(epoch_losses))
    
    return accu_avg, loss_avg


# In[ ]:


# training loop
for epoch in range(num_epoch):
    print('~' * 80)
        
    cur_lr = optimizer.param_groups[0]['lr']
    start = time.time()
    
    train_accu, train_loss = train_epoch()
    eval_accu, eval_loss = eval_epoch()
    
    end = time.time()
    print('%dth iteration took %.4f' % (epoch, end - start))
    print('learning rate: %.4f' % cur_lr)
    print('train_loss: %.4f' % train_loss)
    print('eval_loss: %.4f' % eval_loss)
    print('train_accuracy: %.4f' % train_accu)
    print('eval_accuracy: %.4f' % eval_accu)
    sys.stdout.flush()

