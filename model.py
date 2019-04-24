# refer to
#
# https://github.com/intfloat/commonsense-rc
#
# https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
#
# https://discuss.pytorch.org/t/solved-multiple-packedsequence-input-ordering/2106/23

# In[61]:
import torch
import torch.nn as nn
from torch.autograd import Variable


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
        outputs = torch.zeros_like(outputs).scatter_(0, sorted_idx.unsqueeze(1).unsqueeze(1) \
                                                     .expand(-1, outputs.shape[1], outputs.shape[2]), outputs)
        outputs = self.rnn_output_dropout(outputs)

        return outputs


def lengths_to_mask(lengths, dtype=torch.uint8):
    """

    :param lengths: (batch_size)
    :param dtype:
    :return: (batch_size, max_len)
    """
    device = lengths.get_device()
    lengths = lengths.cpu()

    max_len = lengths.max().item()
    mask = torch.arange(max_len,
                        dtype=lengths.dtype).expand(len(lengths), max_len) < lengths.unsqueeze(1)

    mask = torch.as_tensor(mask, dtype=dtype, device=device)
    mask = 1 - mask
    return mask


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
        weights = self.softmax(scores)  # (batch_size, seq_len)

        return weights.unsqueeze(1).bmm(x).squeeze(1)


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
        weights = self.softmax(scores)  # (batch_size, seq_len)

        return weights.unsqueeze(1).bmm(inputs).squeeze(1)


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


class TriAn(nn.Module):
    def __init__(self, embedding, embedding_pos, embedding_ner, embedding_rel, config):
        super(TriAn, self).__init__()
        self.embedding = embedding
        self.embedding_pos = embedding_pos
        self.embedding_ner = embedding_ner
        self.embedding_rel = embedding_rel

        self.d_rnn = BLSTM(config.embed_dim * 2 + config.embed_dim_pos + config.embed_dim_ner + config.
                           embed_dim_rel * 2 + config.embed_dim_value * 5, config.hidden_size,
                           config.num_layers, config.rnn_dropout_rate)
        self.q_rnn = BLSTM(config.embed_dim + config.embed_dim_pos, config.hidden_size, config.num_layers,
                           config.rnn_dropout_rate)
        self.c_rnn = BLSTM(config.embed_dim * 3, config.hidden_size, config.num_layers, config.rnn_dropout_rate)

        self.embed_dropout = nn.Dropout(config.embed_dropout_rate)

        self.d_on_q_attn = SeqAttnContext(config.embed_dim)
        self.c_on_q_attn = SeqAttnContext(config.embed_dim)
        self.c_on_d_attn = SeqAttnContext(config.embed_dim)

        self.d_on_q_encode = BilinearAttnEncoder(config.hidden_size * 2, config.hidden_size * 2)
        self.q_encode = SelfAttnEncoder(config.hidden_size * 2)
        self.c_encode = SelfAttnEncoder(config.hidden_size * 2)

        self.d_c_bilinear = Bilinear(config.hidden_size * 2, config.hidden_size * 2)
        self.q_c_bilinear = Bilinear(config.hidden_size * 2, config.hidden_size * 2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, d_words, d_pos, d_ner, d_lengths, q_words, q_pos, q_lengths, c_words, c_lengths, \
                in_q, in_c, lemma_in_q, lemma_in_c, tf, p_q_relation, p_c_relation):
        # embed inputs
        d_embed, q_embed, c_embed = self.embedding(d_words), self.embedding(q_words), self.embedding(c_words)
        d_embed, q_embed, c_embed = self.embed_dropout(d_embed), self.embed_dropout(q_embed), self.embed_dropout(
            c_embed)

        d_pos_embed, d_ner_embed, q_pos_embed = self.embedding_pos(d_pos), self.embedding_ner(
            d_ner), self.embedding_pos(q_pos)
        d_pos_embed, d_ner_embed, q_pos_embed = self.embed_dropout(d_pos_embed), self.embed_dropout(
            d_ner_embed), self.embed_dropout(q_pos_embed)

        p_q_rel_embed, p_c_rel_embed = self.embedding_rel(p_q_relation), self.embedding_rel(p_c_relation)
        p_q_rel_embed, p_c_rel_embed = self.embed_dropout(p_q_rel_embed), self.embed_dropout(p_c_rel_embed)

        # get masks
        d_mask = lengths_to_mask(d_lengths)
        q_mask = lengths_to_mask(q_lengths)
        c_mask = lengths_to_mask(c_lengths)

        # get attention contexts
        d_on_q_contexts = self.embed_dropout(self.d_on_q_attn(d_embed, q_embed, q_mask))
        c_on_q_contexts = self.embed_dropout(self.c_on_q_attn(c_embed, q_embed, q_mask))
        c_on_d_contexts = self.embed_dropout(self.c_on_d_attn(c_embed, d_embed, d_mask))

        # form final inputs for rnns
        d_rnn_inputs = torch.cat([d_embed, d_on_q_contexts, d_pos_embed, d_ner_embed, \
                                  p_q_rel_embed, p_c_rel_embed, in_q, in_c, lemma_in_q, lemma_in_c, tf], dim=2)
        q_rnn_inputs = torch.cat([q_embed, q_pos_embed], dim=2)
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


class LockedDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        """

        :param x: (batch_size, seq_len, x_dim)
        :return:
        """
        if not self.training or not self.p:
            return x

        mask = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p)
        mask = Variable(mask, requires_grad=False) / (1 - self.p)
        mask = mask.expand_as(x)
        masked = mask * x

        return masked


class LM(nn.Module):
    def __init__(self, ntoken, ninp, nhid, embedding, dropout, device):
        super(LM, self).__init__()
        self.device = device
        self.nhid = nhid
        self.encoder = embedding
        self.rnn = nn.LSTM(ninp, nhid, batch_first=True)
        self.decoder = nn.Linear(nhid, ntoken)
        self.embed_drop = LockedDropout(dropout)
        self.output_drop = LockedDropout(dropout)

        # # tie weights
        # self.decoder.weight = self.encoder.weight

    def forward(self, inputs, hidden=None):
        """

        :param inputs: (batch_size, max_len)
        :param hidden: ((1, batch_size, nhid), (1, batch_size, nhid))
        :return:
        """
        emb = self.embed_drop(self.encoder(inputs))
        if hidden:
            outputs, hidden = self.rnn(emb, hidden)
        else:
            outputs, hidden = self.rnn(emb)
        outputs = self.output_drop(outputs)
        decoded = self.decoder(outputs)
        return decoded, outputs, hidden
