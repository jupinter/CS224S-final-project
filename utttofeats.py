import os
import sys
import subprocess

converter = '../festival/examples/dumpfeats'
feats_list = ['name']
feats = "'(" + ' '.join(feats_list) + ")'"

if __name__ == '__main__':
    if len(sys.argv) < 3 or not os.path.isdir(sys.argv[1]) or not os.path.isdir(sys.argv[2]):
        print 'ERROR: provide input utt dir and output feats directory'
        print ''
        print 'File Args: python uttofeats.py utt_dir feats_dir'
        print 'Usage example: python indextowav.py ../ATrampAbroad/utt ../ATrampAbroad/feats'
        exit()


    in_path = sys.argv[1]
    output_dir = sys.argv[2]
    output = os.path.join(output_dir, '%s.feats')
    fromfile = 'utts.txt'
    
    with open(fromfile, 'w') as f:
        for utt in os.listdir(in_path):
            f.write(os.path.join(in_path, utt) + '\n')

    os.system('{} -relation Syllable -feats {} -from_file {} -output {}'.format(converter, feats, fromfile, output))