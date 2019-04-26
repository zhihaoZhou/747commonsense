import torch.nn as nn
from torchtext import data, datasets
import os


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

    def __init__(self, config, lm_config, device):
        # define all fields
        TEXT = data.ReversibleField(sequential=True, tokenize=self.tokenizer,
                                    lower=False, include_lengths=False)
        POS = data.ReversibleField(sequential=True, lower=False, include_lengths=True)
        NER = data.ReversibleField(sequential=True, lower=False, include_lengths=True)
        LABEL = data.Field(sequential=False, use_vocab=False)
        IN_Q = data.Field(sequential=True, use_vocab=False, include_lengths=True,
                          postprocessing=self.to_numeric)
        IN_C = data.Field(sequential=True, use_vocab=False, include_lengths=True,
                          postprocessing=self.to_numeric)
        LEMMA_IN_Q = data.Field(sequential=True, use_vocab=False, include_lengths=True,
                                postprocessing=self.to_numeric)
        LEMMA_IN_C = data.Field(sequential=True, use_vocab=False, include_lengths=True,
                                postprocessing=self.to_numeric)
        TF = data.Field(sequential=True, use_vocab=False, include_lengths=True,
                        postprocessing=self.to_numeric)
        REL = data.ReversibleField(sequential=True, lower=False, include_lengths=True)

        # load lm data first
        lm_train = datasets.LanguageModelingDataset(os.path.join(lm_config.file_path, lm_config.train_f),
                                                    TEXT, newline_eos=False)
        lm_dev = datasets.LanguageModelingDataset(os.path.join(lm_config.file_path, lm_config.dev_f),
                                                  TEXT, newline_eos=False)

        # load actual data
        # we have keys: 'id', 'd_words', 'd_pos', 'd_ner', 'q_words', 'q_pos', 'c_words',
        #       'label', 'in_q', 'in_c', 'lemma_in_q', 'tf', 'p_q_relation', 'p_c_relation'
        train, val, test = data.TabularDataset.splits(
            path=config.data_dir, train=config.train_fname,
            validation=config.dev_fname, test=config.test_fname, format='json',
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

        # construct vocabulary
        TEXT.build_vocab(train, val, test, lm_train, lm_dev, vectors=config.vectors)
        POS.build_vocab(train, val, test)
        NER.build_vocab(train, val, test)
        REL.build_vocab(train, val, test)

        print('vocab size: %d' % len(TEXT.vocab))
        print('pos size: %d' % len(POS.vocab))
        print('ner size: %d' % len(NER.vocab))
        print('rel size: %d' % len(REL.vocab))

        self.TEXT = TEXT

        # iterators
        self.lm_train_iter = data.BPTTIterator(lm_train, batch_size=lm_config.batch_size,
                                               bptt_len=lm_config.bptt_len, repeat=False)
        self.lm_dev_iter = data.BPTTIterator(lm_dev, batch_size=lm_config.batch_size,
                                             bptt_len=lm_config.bptt_len, repeat=False)

        print('lm train batch num: %d, lm dev batch num: %d' %
              (len(self.lm_train_iter), len(self.lm_dev_iter)))

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

        print('train batch num: %d, dev batch num: %d' %
              (len(self.train_iter), len(self.val_iter)))

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

        self.vocab_size = len(TEXT.vocab)
        print('vocab_size is', self.vocab_size)

