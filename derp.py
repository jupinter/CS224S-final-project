import matplotlib.pyplot as plt
import os
import numpy as np


feats_dirs = ['../ATrampAbroad/feats_test', '../LifeOnTheMississippi/feats', '../TheAdventuresOfTomSawyer/feats', '../TheManThatCorruptedHadleyburg/feats']

# feats_dirs = ['../ATrampAbroad/feats_test']

def num_lines(file):
    with open(file) as f:
        s = sum(1 for line in f if line != '\n') 
    return s

# for feats_dir in ['../test/feats']:
#     for file in os.listdir(feats_dir):
#         if num_lines(os.path.join(feats_dir, file)) == 0:
#             print feats_dir, file

# part1 = ["Wanted, Chief Justice of the Massechusetts Supreme Court.",
#          "In April, the S.J.C.'s current leader Edward Hennessy reaches the mandentory retirement age of seventy, and a successor is expected to be named in March.",
#          "It may be the most important appointment Governor Michael Dukakis makes during the remainder of his administration and one of the toughest.",
#          "As WBUR's Margo Melnicove reports, Hennessy will be a hard act to follow."
#          ]
# part2 = ["In nineteen-seventy-six Democratic Governor Michael Dukakis fulfilled a campaign promise to de-politicize judicial appointments.",
#          "He named Republican Edward hennessy to head the State Supreme Judicial Court.",
#          "For Hennessy, it was another step along a distinguished career that began as a trial lawyer and led to an appointment as an associate Supreme Court Justice in nineteen-seventy-one.",
#          "That year Thomas Maffy, now president of the Massachusetts Bar association, was Hennessy's law clerk."]
# with open('../test/test_index.txt', 'w') as f:
#     for i in range(8):
#         for idx, sent in enumerate(part1):
#             f.write("chp{}1_{:05d}\t1\t{}/{}\t0\t0\t{}\n".format(i, idx + 1, idx + 1, len(part1), sent))
#         for idx, sent in enumerate(part2):
#             f.write("chp{}2_{:05d}\t2\t{}/{}\t0\t0\t{}\n".format(i, idx + 1, idx + 1, len(part2), sent))

# for file in os.listdir('../test/features'):
#     with open(os.path.join('../test/features', file)) as f:
#         with open(os.path.join('../test/features', file.split('.')[0] + '.txt'), 'w') as g:
#             for line in f:
#                 line = line.strip().split(' ')
#                 syl_duration = line[2]
#                 g.write(syl_duration + '\n')



l = [num_lines(os.path.join(feats_dir, f)) for feats_dir in feats_dirs for f in os.listdir(feats_dir) ]

nums = list(set(l))
counts = {x : l.count(x) for x in nums}
pairs = zip(*counts.items())
# plt.scatter(*pairs)

plt.hist(l, range(0, max(nums), 4))

print sum(1. for x in l if x <= 60) / len(l)


plt.title('Syllable Counts')
plt.ylim(ymin = 0)
plt.xlim(xmin = 0)
plt.xlabel('num syllables')
plt.ylabel('num sentences')
plt.show()

# def get_pitch_chps(file):
#     curr_file = ''
#     l = []
#     with open(file) as f:
#         f.next()
#         for line in f:
#             line = line.strip().split('\t')
#             new_file = line[0][-11:] 
#             if curr_file != new_file:
#                 curr_file = new_file
#                 l += [curr_file.split('.')[0]]
#     return l

# train_costs = []
# dev_costs = []
# param_norms = []
# with open('log.txt') as f:
#     for line in f:
#         if line == 'randomized training order\n':
#             break
#     for line in f:
#         # if line == 'randomized training order\n':
#         #     break
#         line = line.strip().split('|')
#         if line[0][:5] != 'Epoch':
#             continue
#         train_costs += [float(line[1].strip().split(' ')[2])]
#         dev_costs += [float(line[2].strip().split(' ')[2])]
#         param_norms += [float(line[3].strip().split(' ')[2])]
# plt.plot(train_costs, label = 'train')
# plt.plot(dev_costs, label = 'dev')
# plt.legend()
# plt.ylabel('Cost)')
# # plt.plot(param_norms)
# # plt.ylabel('Param norm')
# plt.xlabel('Epoch')
# plt.show()



# with open('lr_example.txt') as f:
#     name = ''
#     for line in f:
#         if line[0] == '#':
#             continue
#         elif line == '\n':
#             if name != '':
#                 print 'saving', name
#                 plt.plot(train, label=name + ' train')
#                 plt.plot(dev, label=name + ' dev')
#         elif line[:5] == 'lr = ':
#             name = line.strip()[5:]
#             train = []
#             dev = []
#             continue
#         else:
#             line = line.split(' ')
#             train += [float(line[5])]
#             dev += [float(line[9])]
# plt.legend(bbox_to_anchor=(1, 1),
#            bbox_transform=plt.gcf().transFigure)
# plt.xlabel('Epoch')
# plt.ylabel('Error')
# plt.show()

# stuff = [((50, 64), (150937.599, 152141.602)),
#         ((80, 64), (157095.956, 145916.255)),
#         ((100, 64), (157203.146, 153239.398)),
        # ((50, 128), (149995.017, 159700.656)),
        # ((80, 128), (1,1)),
        # ((100, 128), (1,1)),
        # ]
# stuff = [(50, (150937.599, 152141.602)),
#         (80, (157095.956, 145916.255)),
#         (100, (157203.146, 153239.398)),]
# stuff = [('numeric', (157095.956, 145916.255)),
#         ('all', (154197.210, 181491.502))
#         ]
# x_names, ys = zip(*stuff)
# ys = zip(*ys)
# x = range(len(stuff))
# plt.xticks(x, x_names)
# plt.scatter(x, ys[0], label = 'train', color = 'blue')
# plt.scatter(x, ys[1], label = 'dev', color = 'red')
# plt.xlabel('features')
# plt.ylabel('error')
# plt.title('Adding non-numeric features')
# # plt.ylim(ymin = 0)
# plt.legend(loc = 2)
# plt.show()