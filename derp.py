import matplotlib.pyplot as plt
import os

feats_dir = '../ATrampAbroad/feats'

def num_lines(file):
    with open(file) as f:
        s = sum(1 for line in f if line != '\n') 
    return s

l = [num_lines(os.path.join(feats_dir, f)) for f in os.listdir(feats_dir)]

nums = list(set(l))
counts = {x : l.count(x) for x in nums}
pairs = zip(*counts.items())
plt.scatter(*pairs)
plt.title('Syllable Counts')
plt.ylim(ymin = 0)
plt.xlim(xmin = 0)
plt.show()