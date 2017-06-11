import os
import sys
import subprocess

converter = '../festival/examples/text2utt'

def line_to_feats(line, outpath):
    with open('temp.txt', 'w') as f:
        f.write(line)
    # print line
    # print converter
    # print outpath
    # print '{} -o {} {}'.format(converter, outpath, 'temp.txt')
    line = line.strip('-')
    os.system('{} {} > {}'.format(converter, 'temp.txt', outpath))

if __name__ == '__main__':
    if len(sys.argv) < 3 or os.path.isdir(sys.argv[1]) or not os.path.isdir(sys.argv[2]):
        print 'ERROR: provide input index file and output directory'
        print ''
        print 'File Args: python indextowav.py sentence_index.txt outputdir'
        print 'Usage example: python indextowav.py ../ATrampAbroad/sentence_index.txt ../ATrampAbroad/utt'
        print ''
        print 'Optional Arg: last chapter to process: does 1-n'
        print 'default is all'
        exit()


    index = sys.argv[1]
    output_dir = sys.argv[2]

    min_chapters = None
    max_chapters = int(sys.argv[3]) if len(sys.argv) > 3 else None

    curr_chapter = 0
    with open(index) as f:
        for line in f:
            if line[0] == '#':
                continue
            line = line.split('\t')
            name = line[0]
            # 5_SENT_TXT_BOOK  6_SENT_TXT_REC  7_SENT_TXT_LAB
            sentence = line[5]
            if len(name) < 3 or name[:3] != 'chp': # no file associated
                continue
            chapter = int(name[3:5])
            if min_chapters != None and chapter < min_chapters:
                continue
            if max_chapters != None and chapter > max_chapters:
                break

            if chapter != curr_chapter:
                print 'Processing chapter', chapter
                curr_chapter = chapter
            outpath = os.path.join(output_dir, name + '.utt')
            line_to_feats(sentence, outpath)