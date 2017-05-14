import numpy as np
import sys

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'PROVIDE INPUT AND OUTPUT FILES'
        print 'Args: python labtogrid.py infile.lab outfile.TextGrid'
        print 'Usage example: python labtogrid.py ../ATrampAbroad/lab/chp01_00001.lab ../ATrampAbroad/TextGrid/chp01_00001.TextGrid'
    filename = sys.argv[1]
    phones = []
    syllables = []
    words = []
    poss = []
    with open(filename) as f:
        phone_start = 0
        syl_start = 0
        word_start = 0
        for line in f:
            elems = line.strip().split('; ')
            e1 = elems[0].split(' ')
            if len(e1) == 1: # first line
                continue 
            end_time = float(e1[0])
            phone = e1[2]
            phones.append([phone_start, end_time, phone])
            phone_start = end_time
            if len(elems) > 1:
                syllable = elems[1] # just a $
                syllables.append([syl_start, end_time, syllable])
                syl_start = end_time
            if len(elems) > 2:
                word = elems[2][6:-1]
                words.append([word_start, end_time, word])
                pos = elems[3][5:-1]
                poss.append([word_start, end_time, pos])
                word_start = end_time
    namesplit = filename.split('/')
    outfile = '/'.join(namesplit[:-2]) + '/TextGrid/' + namesplit[-1][:-4] + '.TextGrid'
    with open(outfile, 'w') as f:
        f.write('File type = "ooTextFile"\n')
        f.write('Object class = "TextGrid"\n')
        f.write('\n')
        f.write('xmin = 0\n')
        f.write('xmax = {}\n'.format(end_time))
        f.write('tiers? <exists>\n')
        f.write('size = 3\n')
        f.write('item []:\n')
        def print_list(l, name, index):
            f.write('\titem [{}]:\n'.format(index))
            f.write('\t\tclass = "IntervalTier"\n')
            f.write('\t\tname = "{}"\n'.format(name))
            f.write('\t\txmin = 0\n')
            f.write('\t\txmax = {}\n'.format(end_time))
            f.write('\t\tintervals: size = {}\n'.format(len(l)))
            for i, (start, end, text) in enumerate(l):
                f.write('\t\tintervals [{}]\n'.format(i + 1))
                f.write('\t\t\txmin = {}\n'.format(start))
                f.write('\t\t\txmax = {}\n'.format(end))
                f.write('\t\t\ttext = "{}"\n'.format(text))
        print_list(phones, 'phones', 1)
        print_list(syllables, 'syllables', 2)
        print_list(words, 'words', 3)
        print_list(poss, 'pos', 3)
