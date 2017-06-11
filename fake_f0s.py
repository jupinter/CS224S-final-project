import os
import random

feats_dir = '../ATrampAbroad/feats'
f0_dir = '../ATrampAbroad/f0'

min_freq = 50.
max_freq = 300.

for feats in os.listdir(feats_dir):
    f0 = feats.split('.')[0] + '.txt'
    with open(os.path.join(feats_dir, feats)) as f, open(os.path.join(f0_dir, f0), 'w') as o:
        for line in f:
            for i in range(3):
                o.write('{:8f} '.format(random.uniform(min_freq, max_freq)))
            o.write('\n')
