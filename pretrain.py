import spacy
from spacy.symbols import ORTH
import torchtext
from torchtext import data, datasets


my_tok = spacy.load('en')
my_tok.tokenizer.add_special_case('<unk>', [{ORTH: '<unk>'}])


def spacy_tok(x):
    return [tok.text for tok in my_tok.tokenizer(x)]


if __name__ == '__main__':
    file_path = 'pretrain_data/raw_stories.txt'
    TEXT = data.Field(lower=True, tokenize=spacy_tok)
    train = datasets.LanguageModelingDataset(file_path, TEXT, newline_eos=False)
    print(len(train))

    TEXT.build_vocab(train)
    train_iter = data.BPTTIterator(train, batch_size=5, bptt_len=30)

    for batch in train_iter:
        print(batch)
        break
