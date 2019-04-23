import spacy
from spacy.symbols import ORTH
import torchtext
from torchtext import data, datasets


my_tok = spacy.load('en')
my_tok.tokenizer.add_special_case('<unk>', [{ORTH: '<unk>'}])


def spacy_tok(x):
    return [tok.text for tok in my_tok.tokenizer(x)]


class Config:
    batch_size = 5
    bptt_len = 10
    file_path = 'pretrain_data/raw_stories.txt'


if __name__ == '__main__':
    config = Config()
    TEXT = data.Field(lower=True, tokenize=spacy_tok)
    train = datasets.LanguageModelingDataset(config.file_path, TEXT, newline_eos=False)
    print(len(train))

    TEXT.build_vocab(train)
    train_iter = data.BPTTIterator(train, batch_size=config.batch_size, bptt_len=config.bptt_len)

    for batch in train_iter:
        x, y = batch.text.transpose(0, 1), batch.target.transpose(0, 1)
        print(vars(batch).keys())
        print('~' * 80)
        print(x)
        print('~'*80)
        print(y)
        break
