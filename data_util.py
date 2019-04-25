import torchtext
from torchtext import data
import os
import torch.nn as nn


# Load data
# refer to
#
# http://anie.me/On-Torchtext/
#
# http://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/
#
# http://mlexplained.com/2018/02/15/language-modeling-tutorial-in-torchtext-practical-torchtext-part-2/

class DataUtil:
    @staticmethod
    def tokenizer(text):
        return text.split(" ")

    @staticmethod
    def to_numeric(tf_batch, tf_lens):
        for i in range(len(tf_batch)):
            for j in range(len(tf_batch[0])):
                if tf_batch[i][j] == '<pad>':
                    tf_batch[i][j] = float(0)
                else:
                    tf_batch[i][j] = float(tf_batch[i][j])

        return tf_batch

    def __init__(self, data_dir, combined_fname, train_fname, dev_fname, test_fname, config, device):
        TEXT = data.ReversibleField(sequential=True, tokenize=self.tokenizer, lower=False, include_lengths=True)
        POS = data.ReversibleField(sequential=True, lower=False, include_lengths=True)
        NER = data.ReversibleField(sequential=True, lower=False, include_lengths=True)
        LABEL = data.Field(sequential=False, use_vocab=False)
        IN_Q = data.Field(sequential=True, use_vocab=False, include_lengths=True, postprocessing=self.to_numeric)
        IN_C = data.Field(sequential=True, use_vocab=False, include_lengths=True, postprocessing=self.to_numeric)
        LEMMA_IN_Q = data.Field(sequential=True, use_vocab=False, include_lengths=True, postprocessing=self.to_numeric)
        LEMMA_IN_C = data.Field(sequential=True, use_vocab=False, include_lengths=True, postprocessing=self.to_numeric)
        TF = data.Field(sequential=True, use_vocab=False, include_lengths=True, postprocessing=self.to_numeric)
        REL = data.ReversibleField(sequential=True, lower=False, include_lengths=True)

        # we have keys: 'id', 'd_words', 'd_pos', 'd_ner', 'q_words', 'q_pos', 'c_words',
        #       'label', 'in_q', 'in_c', 'lemma_in_q', 'tf', 'p_q_relation', 'p_c_relation'
        train, val, test = data.TabularDataset.splits(
            path=data_dir, train=train_fname,
            validation=dev_fname, test=test_fname, format='json',
            fields={'d_words': ('d_words', TEXT),
                    'd_pos':   ('d_pos', POS),
                    'd_ner':   ('d_ner', NER),
                    'q_words': ('q_words', TEXT),
                    'q_pos':   ('q_pos', POS),
                    'c_words': ('c_words', TEXT),
                    'label': ('label', LABEL),
                    'in_q': ('in_q', IN_Q),
                    'in_c': ('in_c', IN_C),
                    'lemma_in_q': ('lemma_in_q', LEMMA_IN_Q),
                    'lemma_in_c': ('lemma_in_c', LEMMA_IN_C),
                    'tf': ('tf', TF),
                    'p_q_relation': ('p_q_relation', REL),
                    'p_c_relation': ('p_c_relation', REL)
                    })

        print('train: %d, val: %d, test: %d' % (len(train), len(val), len(test)))

        # combined is only used for building vocabulary
        combined = data.TabularDataset(
            path=os.path.join(data_dir, combined_fname), format='json',
            fields={'d_words': ('d_words', TEXT),
                    'd_pos':   ('d_pos', POS),
                    'd_ner':   ('d_ner', NER),
                    'q_words': ('q_words', TEXT),
                    'q_pos':   ('q_pos', POS),
                    'c_words': ('c_words', TEXT),
                    'label': ('label', LABEL),
                    'in_q': ('in_q', IN_Q),
                    'in_c': ('in_c', IN_C),
                    'lemma_in_q': ('lemma_in_q', LEMMA_IN_Q),
                    'lemma_in_c': ('lemma_in_c', LEMMA_IN_C),
                    'tf': ('tf', TF),
                    'p_q_relation': ('p_q_relation', REL),
                    'p_c_relation': ('p_c_relation', REL)
                    })

        TEXT.build_vocab(combined, vectors=config.vectors)
        POS.build_vocab(combined)
        NER.build_vocab(combined)
        REL.build_vocab(combined)

        print('vocab size: %d' % len(TEXT.vocab))
        print('pos size: %d' % len(POS.vocab))
        print('ner size: %d' % len(NER.vocab))
        print('rel size: %d' % len(REL.vocab))

        self.train_iter = data.BucketIterator(dataset=train, batch_size=config.batch_size_train,
                                              sort_key=lambda x: len(x.d_words), device=device, shuffle=True,
                                              sort_within_batch=False, repeat=False)

        self.val_iter = data.Iterator(dataset=val, batch_size=config.batch_size_eval,
                                      sort_key=lambda x: len(x.d_words),
                                      train=False, shuffle=False, sort_within_batch=False, device=device,
                                      repeat=False)

        self.test_iter = data.Iterator(dataset=test, batch_size=config.batch_size_test,
                                       sort_key=lambda x: len(x.d_words), train=False, shuffle=False,
                                       sort_within_batch=False, device=device, repeat=False)

        # # Create embeddings
        embedding = nn.Embedding(len(TEXT.vocab), config.embed_dim)
        embedding.weight.data.copy_(TEXT.vocab.vectors)
        embedding.weight.requires_grad = False
        self.embedding = embedding.to(device)

        embedding_pos = nn.Embedding(len(POS.vocab), config.embed_dim_pos)
        embedding_pos.weight.data.normal_(0, 0.1)
        self.embedding_pos = embedding_pos.to(device)

        embedding_ner = nn.Embedding(len(NER.vocab), config.embed_dim_ner)
        embedding_ner.weight.data.normal_(0, 0.1)
        self.embedding_ner = embedding_ner.to(device)

        embedding_rel = nn.Embedding(len(REL.vocab), config.embed_dim_rel)
        embedding_rel.weight.data.normal_(0, 0.1)
        self.embedding_rel = embedding_rel.to(device)

        print('embedding', self.embedding)
        print('embedding_pos', self.embedding_pos)
        print('embedding_ner', self.embedding_ner)
        print('embedding_rel', self.embedding_rel)

