import os

directory = '.'

# join all raw txt to one str
train_raw_str = ''
dev_raw_str = ''
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".raw.txt"):
        with open(os.path.join(directory, filename)) as f_tmp:
            for line in f_tmp:
                train_raw_str += line[:int(len(line) * 0.8)]
                dev_raw_str += line[int(len(line) * 0.8):]
print(len(train_raw_str))
print(len(dev_raw_str))

# write 80% of the str to train, 20% to dev
with open('lm.train', 'w') as f:
    f.write(train_raw_str)
with open('lm.dev', 'w') as f:
    f.write(dev_raw_str)
