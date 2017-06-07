import matplotlib.pyplot as plt
import os

# feats_dir = '../ATrampAbroad/feats'

# def num_lines(file):
#     with open(file) as f:
#         s = sum(1 for line in f if line != '\n') 
#     return s

# l = [num_lines(os.path.join(feats_dir, f)) for f in os.listdir(feats_dir)]

# nums = list(set(l))
# counts = {x : l.count(x) for x in nums}
# pairs = zip(*counts.items())
# # plt.scatter(*pairs)

# plt.hist(nums, range(0, 130, 5))

# plt.title('Syllable Counts')
# plt.ylim(ymin = 0)
# plt.xlim(xmin = 0)
# plt.show()

name = ''

with open('lr_example.txt') as f:
    for line in f:
        if line[0] == '#':
            continue
        elif line == '\n':
            if name != '':
                print 'saving' + name
                plt.plot(train, label=name + ' train')
                plt.plot(dev, label=name + ' dev')
        elif line[:5] == 'lr = ':
            name = line.strip()[5:]
            train = []
            dev = []
            continue
        else:
            line = line.split(' ')
            train += [float(line[5])]
            dev += [float(line[9])]
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.show()
