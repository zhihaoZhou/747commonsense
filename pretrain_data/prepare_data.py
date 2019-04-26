import spacy
from spacy.symbols import ORTH


f_list = ['mcscript.raw.txt']

# join all raw txt to one str
train_raw_str = ''
dev_raw_str = ''
for f_name in f_list:
    with open(f_name) as f_tmp:
        for line in f_tmp:
            train_raw_str += line[:int(len(line) * 0.9)]
            dev_raw_str += line[int(len(line) * 0.9):]

lm_tok = spacy.load('en')
lm_tok.tokenizer.add_special_case('<unk>', [{ORTH: '<unk>'}])


def spacy_tok(x):
    return [tok.text for tok in lm_tok.tokenizer(x)]


# write 90% of the str to train, 10% to dev
with open('lm.train', 'w') as f:
    # tokenize then write
    train_raw_str = ' '.join(spacy_tok(train_raw_str))
    print(len(train_raw_str))
    f.write(train_raw_str)
with open('lm.dev', 'w') as f:
    # tokenize then write
    dev_raw_str = ' '.join(spacy_tok(dev_raw_str))
    print(len(dev_raw_str))
    f.write(dev_raw_str)
