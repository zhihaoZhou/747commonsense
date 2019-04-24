f_list = ['mcscript.raw.txt']

# join all raw txt to one str
train_raw_str = ''
dev_raw_str = ''
for f_name in f_list:
    with open(f_name) as f_tmp:
        for line in f_tmp:
            train_raw_str += line[:int(len(line) * 0.9)]
            dev_raw_str += line[int(len(line) * 0.9):]
print(len(train_raw_str))
print(len(dev_raw_str))

# write 90% of the str to train, 10% to dev
with open('lm.train', 'w') as f:
    f.write(train_raw_str)
with open('lm.dev', 'w') as f:
    f.write(dev_raw_str)
