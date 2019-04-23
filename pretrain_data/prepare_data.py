import os

directory = '.'

# join all raw txt to one str
all_raw_str = ''
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".raw.txt"):
        with open(os.path.join(directory, filename)) as f_tmp:
            for line in f_tmp:
                all_raw_str += line
print(len(all_raw_str))

# write 80% of the str to train, 20% to dev
with open('lm.train', 'w') as f:
    f.write(all_raw_str[:int(len(all_raw_str)*0.8)])
with open('lm.dev', 'w') as f:
    f.write(all_raw_str[int(len(all_raw_str) * 0.8):])
