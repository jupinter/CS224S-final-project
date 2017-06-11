import os
import numpy as np
import sys
from tqdm import tqdm


# position_type
position_types = ['single', 'initial', 'mid', 'final']
# syl_coda_type, syl_onset_type
coda_types = ['+S', '-V', '+V-S'] 
# syl_accent, tobi_accent, tobi_endtone
tone_types = ['NONE', 'H*', '!H*', 'L-L%', 'L-H%', 'H-H%', 'L+H*', 'multi']
# pbreak
break_types = ['NB', 'B', 'BB', 'mB']
# gpos
gpos_types = ['pps', 'md', 'cc', 'wp', 'det', 'content', 'to', 'in', 'aux']
# pos
pos_types = ['vb', 'cc', 'jjs', 'jjr', 'cd', 'prp', 'in', 'nns', 'nnp', 'nnps', 'wrb', 'nn', '1', 'to', '2', 'ls', 'rb', 'rbr', 'fw', 'punc', 'sym', 'pos', 'jj', 'wp', 'rp', 'dt', 'md', 'vbg', 'vbd', 'of', 'pdt', 'rbs', 'vbn', 'vbp', 'wdt', 'uh', 'vbz', 'ex']
# daughter1.name, daughtern.name
segment_names = ['aa', 'iy', 'ch', 'ae', 'eh', 'ah', 'ao', 'ih', 'ey', 'aw', 'ay', 'ax', 'er', 'r', 'th', 'zh', 'oy', 'dh', 'ow', 'hh', 'jh', 'b', 'd', 'g', 'f', 'k', 'm', 'l', 'n', 'p', 's', 'sh', 't', 'w', 'v', 'y', 'z', 'uw']
# prepunctuation
prepunctuation_bins = ['\'', '(']
# punctuation
punctuation_bins = ['!', '?', '.', ',', ';', ')', '\'']

num_numeric_features = 19
alpha_features = [position_types] + [coda_types] * 2 + [tone_types] * 3 + [break_types] + [gpos_types] + [pos_types] + [segment_names] * 2 + [prepunctuation_bins] + [punctuation_bins]
num_alpha_features = sum(len(x) for x in alpha_features)
num_total_features = num_numeric_features + num_alpha_features
print '{} Numeric features, {} alpha features, for a feature size of {}'.format(num_numeric_features, len(alpha_features), num_total_features)

def process_feats(line):
    # return process_line(line)


    line = line.strip().split(' ')
    feats = np.zeros(num_total_features)
    offset = num_numeric_features
    for idx, x in enumerate(line):
        if idx < num_numeric_features:
            feats[idx] = float(x)
        else:
            # one hot shennanigans
            alpha_idx = idx - num_numeric_features
            alpha_feats = alpha_features[alpha_idx]
            if alpha_idx == len(alpha_features) - 1:
                # ['\'', '(']
                feats[offset] = 1 if '\'' in x or '"' in x else 0
                feats[offset + 1] = 1 if ')' in x or '{' in x or '[' in x else 0
            elif alpha_idx == len(alpha_features) - 2:
                # ['!', '?', '.', ',', ';', ')', '\'']
                feats[offset] = 1 if '!' in x else 0
                feats[offset + 1] = 1 if '?' in x else 0
                feats[offset + 2] = 1 if '.' in x else 0
                feats[offset + 3] = 1 if ',' in x else 0
                feats[offset + 4] = 1 if ';' in x else 0
                feats[offset + 5] = 1 if ')' in x or '{' in x or '[' in x else 0
                feats[offset + 6] = 1 if '\'' in x or '"' in x else 0
            else:
                try:
                    feat_idx = alpha_feats.index(x)
                    feats[feat_idx + offset] = 1
                except ValueError:
                    # not there, just keep going
                    pass
            offset += len(alpha_feats)
    return feats


if __name__ == '__main__':
    if len(sys.argv) < 3 or not os.path.isdir(sys.argv[1]) or not os.path.isdir(sys.argv[2]):
        print 'ERROR: provide input directory and output directory'
        print ''
        print 'File Args: python process_feats.py sentence_index.txt outputdir'
        print 'Usage example: python process_feats.py ../ATrampAbroad/feats ../ATrampAbroad/feats_processed'
        exit()
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    for file in tqdm(os.listdir(input_dir)):
        in_path = os.path.join(input_dir, file)
        out_path = os.path.join(output_dir, file)
        with open(in_path) as f, open(out_path, 'w') as g:
            for line in f:
                feats = process_feats(line)
                g.write(' '.join(str(x) for x in feats) + '\n')


