import os
import sys

def read_lab(file):
    phones = []
    syllables = []
    words = []
    poss = []
    with open(file) as f:
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
    return phones, syllables, words, poss, end_time
def write_textgrid(file, phones, syllables, words, poss, end_time):
    with open(file, 'w') as f:
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

def process(inpath, outpath):
    phones, syllables, words, poss, end_time = read_lab(inpath)
    write_textgrid(outpath, phones, syllables, words, poss, end_time)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'ERROR: provide input and output paths/directories'
        print ''
        print 'File Args: python labtogrid.py infile.lab outfile.TextGrid'
        print 'Usage example: python labtogrid.py ../ATrampAbroad/lab/chp01_00001.lab ../ATrampAbroad/TextGrid/chp01_00001.TextGrid'
        print ''
        print 'Dir Args: python labtogrid.py indir outdir'
        print 'Usage example: python labtogrid.py ../ATrampAbroad/lab ../ATrampAbroad/TextGrid'
    if os.path.isdir(sys.argv[1]) and os.path.isdir(sys.argv[2]):
        indir = sys.argv[1]
        outdir = sys.argv[2]
        curr_chapter = 0
        for infile in os.listdir(indir):
            outfile = infile[:-4] + '.TextGrid'
            chapter = int(infile[3:5])
            if chapter != curr_chapter:
                print 'Processing chapter', chapter
                curr_chapter = chapter
            inpath = os.path.join(indir, infile)
            outpath = os.path.join(outdir, outfile)
            process(inpath, outpath)
    elif not os.path.isdir(sys.argv[1]) and not os.path.isdir(sys.argv[2]):
        process(sys.argv[1], sys.argv[2])
    else:
        print 'ERROR: Both args should be either both files or both directories'
