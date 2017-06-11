import os
import sys
import subprocess

converter = '../festival/bin/text2wave'

def line_to_wav(line, path):
    with open('temp.txt', 'w') as f:
        f.write(line)
    # subprocess supposedly nicer, but it doesn't terminate.
    # subprocess.call([converter, '-o', path, 'temp.txt'], shell=True)

    # This eats ctrl + c, so need to manually kill python to terminate
    os.system('{} -o {} {}'.format(converter, path, 'temp.txt'))

'''
1 indexed descrpiton of lines:
# 1_SENT_FILE_NAME  2_PARAGRAPH_NUMBER  3_SENT_IN_PARAGR  4_CONFIDENCE_VALUE_IN_PERCENT  5_SENT_NUMBER_IN_BOOK 6_SENT_TXT_BOOK  7_SENT_TXT_REC  8_SENT_TXT_LAB
'''

if __name__ == '__main__':
    if len(sys.argv) < 3 or os.path.isdir(sys.argv[1]) or not os.path.isdir(sys.argv[2]):
        print 'ERROR: provide input index file and output directory'
        print ''
        print 'File Args: python indextowav.py sentence_index.txt outputdir'
        print 'Usage example: python indextowav.py ../ATrampAbroad/sentence_index.txt ../ATrampAbroad/TTS'
        print ''
        print 'Optional Arg: last chapter to process: does 1-n'
        print 'default is all'
        exit()
    index = sys.argv[1]
    output_dir = sys.argv[2]

    min_chapters = None
    max_chapters = int(sys.argv[3]) if len(sys.argv) > 3 else None

    'Converting to utterences'
    curr_chapter = 0
    with open(index) as f:
        for line in f:
            if line[0] == '#':
                continue
            line = line.split('\t')
            name = line[0]
            sentence = line[5]
            if len(name) < 3 or name[:3] != 'chp': # no file associated
                continue
            chapter = int(name[3:5])
            if min_chapters and chapter < min_chapters:
                continue
            if max_chapters and chapter > max_chapters:
                exit()

            if chapter != curr_chapter:
                print 'Processing chapter', chapter
                curr_chapter = chapter
            outpath = os.path.join(output_dir, name + '.wav')
            line_to_wav(sentence, outpath)