import matplotlib.pyplot as plt
import os
import numpy as np


with open('dump.txt') as f:
    stack = np.vstack([line.strip().split(' ') for line in f])
num_feats = stack.shape[1]
for i in range(num_feats):
    possible = set(stack[:, i])
    try:
        [float(x) for x in possible]
        print len(possible), 'numbers', (possible if len(possible) < 10 else '')
    except ValueError:
        print possible



'''
R:SylStructure.daughter1.name segment_names
R:SylStructure.daughtern.name segment_names
R:SylStructure.daughter1.seg_coda_fric 0,1
R:SylStructure.daughtern.seg_coda_fric 0,1
R:SylStructure.daughter1.seg_onset_stop 0,1
R:SylStructure.daughtern.seg_onset_stop 0,1
R:SylStructure.daughter1.segment_duration float
R:SylStructure.daughtern.segment_duration float
R:SylStructure.daughter1.syl_final 0, 1
accented: 0
pos_in_word some ints
position_type set(['single', 'initial', 'mid', 'final'])
stress 0, 1
syl_accent set(['L+H*', 'NONE', 'L-L%', '!H*', 'multi', 'H*', 'L-H%', 'H-H%'])
syl_break 0 to 4
syl_coda_type set(['+S', '-V', '+V-S'])
syl_numphones numbers
syl_codasize numbers 
syl_onset_type set(['+S', '-V', '+V-S'])
syllable_duration floats
tobi_accent accents
tobi_endtone accents
R:SylStructure.parent.blevel 0 to 4
R:SylStructure.parent.cap 0, 1
R:SylStructure.parent.contentp 0, 1
R:SylStructure.parent.gpos set(['pps', 'md', 'cc', 'wp', 'det', 'content', 'to', 'in', 'aux'])
R:SylStructure.parent.pbreak set(['NB', 'B', 'BB'])
R:SylStructure.parent.pbreak_score 0?
R:SylStructure.parent.pos set(['vb', 'cc', 'jjs', 'jjr', 'cd', 'prp', 'in', 'nns', 'nnp', 'nnps', 'wrb', 'nn', '1', 'to', '2', 'ls', 'rb', 'rbr', 'fw', 'punc', 'sym', 'pos', 'jj', 'wp', 'rp', 'dt', 'md', 'vbg', 'vbd', 'of', 'pdt', 'rbs', 'vbn', 'vbp', 'wdt', 'uh', 'vbz', 'ex'])
R:SylStructure.parent.pos_score 0?
R:SylStructure.parent.word_break skip?
R:SylStructure.parent.word_duration float
R:SylStructure.parent.word_numsyls count
R:SylStructure.parent.R:Token.parent.prepunctuation set(['"\'', '""', "'", '(', "('", "''", '"', "['", '\'"', '{', '['])
R:SylStructure.parent.R:Token.parent.punc set([');', '!"', "!'", ':]', "';", '),', ').', ".'", "'.]}", '!', ",'", '"', '.,', "'", '!...', ')', '."', ',', '.', '0', '.;', "'.'", ".']}", ';', ':', '?', '...', '""', '.)', ';"', ";'", ',)', "?'", ',"', '!{', '."\'', '?"', '.]}', '!"\'', "',", ']}', '!....', '.]'])
'''


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

# plt.hist(l, range(0, 140, 5))

# plt.title('Syllable Counts')
# plt.ylim(ymin = 0)
# plt.xlim(xmin = 0)
# plt.xlabel('num syllables')
# plt.ylabel('num sentences')
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