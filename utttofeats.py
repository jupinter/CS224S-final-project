import os
import sys
import subprocess

converter = '../festival/examples/dumpfeats'

if __name__ == '__main__':
    if len(sys.argv) < 4 or not os.path.isdir(sys.argv[1]) or os.path.isdir(sys.argv[2]) or not os.path.isdir(sys.argv[3]):
        print 'ERROR: provide input utt dir and output feats directory'
        print ''
        print 'File Args: python utttofeats.py utt_dir feats_file output_feats_dir'
        print 'Usage example: python utttofeats.py ../ATrampAbroad/utt feats.txt ../ATrampAbroad/feats'
        exit()


    utt_path = sys.argv[1]
    feats = sys.argv[2]
    output_dir = sys.argv[3]
    output = os.path.join(output_dir, '%s.feats')
    fromfile = 'utts.txt'
    
    with open(fromfile, 'w') as f:
        for utt in os.listdir(utt_path):
            f.write(os.path.join(utt_path, utt) + '\n')

    command = '{} -relation Syllable -feats {} -from_file {} -output {}'.format(converter, feats, fromfile, output)
    print command
    os.system(command)