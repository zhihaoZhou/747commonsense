import spacy
from spacy.symbols import ORTH
import torchtext
from torchtext import data, datasets
import os


my_tok = spacy.load('en')
my_tok.tokenizer.add_special_case('<unk>', [{ORTH: '<unk>'}])


def spacy_tok(x):
    return [tok.text for tok in my_tok.tokenizer(x)]


class Config:
    batch_size = 5
    bptt_len = 10
    data_dir = 'pretrain_data'
    train_f = 'lm.train'
    dev_f = 'lm.dev'
    file_path = 'pretrain_data'
    vectors = "glove.840B.300d"


if __name__ == '__main__':
    config = Config()
    TEXT = data.Field(lower=True, tokenize=spacy_tok)
    train = datasets.LanguageModelingDataset(os.path.join(config.file_path, config.train_f),
                                             TEXT, newline_eos=False)
    dev = datasets.LanguageModelingDataset(os.path.join(config.file_path, config.dev_f),
                                           TEXT, newline_eos=False)

    TEXT.build_vocab(train)
    # TEXT.build_vocab(train, vectors=config.vectors)
    train_iter = data.BPTTIterator(train, batch_size=config.batch_size, bptt_len=config.bptt_len)
    dev_iter = data.BPTTIterator(dev, batch_size=config.batch_size, bptt_len=config.bptt_len)

    print(len(train_iter), len(dev_iter))

    for batch in train_iter:
        x, y = batch.text.transpose(0, 1), batch.target.transpose(0, 1)
        print(vars(batch).keys())
        print('~' * 80)
        print(x)
        print('~'*80)
        print(y)
        break
