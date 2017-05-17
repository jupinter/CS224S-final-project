import os
import sys
import shutil

text2wave = '../festival/bin/text2wave'

def get_path(name):
    path = os.path.join('..', 'boston-univ-radio-corpus', 'BU_RADIO_' + name[1], name[0:3])
    if name[3] == 's':
        path = os.path.join(path, 'radio', name[3:6])
    else:
        path = os.path.join(path, 'labnews', name[3], 'radio' if name[4:6] == 'rl' else 'oldradio')
    return path

# for files of format end_time, junk, BLAH
def get_entry(path, name, extension, text_index):
    file = os.path.join(path, name + extension)
    ret = []
    start_time = 0.
    with open(file) as f:
        for line in f:
            if line == '#\n':
                break
        for line in f:
            if line == '\n':
                continue
            elems = line.strip().split(' ')
            end_time = float(elems[0])
            if end_time == 0 or str(end_time) == 'nan':
                continue
            text = elems[text_index]
            ret.append([start_time, end_time, text])
            start_time = end_time
    return ret

def get_pos(path, name, words):
    file = os.path.join(path, name + '.pos')
    idx = 0
    pos = []
    with open(file) as f:
        for line in f:
            if line == '\n':
                continue
            elems = line.strip().split(' ')
            # word in words[idx] should be word in elems up to extra punctuation in elems
            if words[idx][2].lower() not in elems[0].lower():
                print 'word/pos mismatch: ', words[idx][2], elems[0]
            pos.append([words[idx][0], words[idx][1], elems[1]])
            idx += 1
    return pos

def write_textgrid(out_path, name, layers, names, end_time):
    out_file = os.path.join(out_path, name + '.TextGrid')
    with open(out_file, 'w') as f:
        f.write('File type = "ooTextFile"\n')
        f.write('Object class = "TextGrid"\n')
        f.write('\n')
        f.write('xmin = 0\n')
        f.write('xmax = {}\n'.format(end_time))
        f.write('tiers? <exists>\n')
        f.write('size = {}\n'.format(len(names)))
        f.write('item []:\n')
        for index, layer in enumerate(layers):
            f.write('\titem [{}]:\n'.format(index + 1))
            f.write('\t\tclass = "IntervalTier"\n')
            f.write('\t\tname = "{}"\n'.format(names[index]))
            f.write('\t\txmin = 0\n')
            f.write('\t\txmax = {}\n'.format(layer[-1][1]))
            f.write('\t\tintervals: size = {}\n'.format(len(layer)))
            for i, (start, end, text) in enumerate(layer):
                f.write('\t\tintervals [{}]\n'.format(i + 1))
                f.write('\t\t\txmin = {}\n'.format(start))
                f.write('\t\t\txmax = {}\n'.format(end))
                f.write('\t\t\ttext = "{}"\n'.format(text))

def TTS(name, path, out_path):
    in_file = os.path.join(path, name + '.txt')
    out_file = os.path.join(out_path, name + '.wav')
    temp_file = 'temp.txt'
    string = ''
    with open(in_file) as orig:
        for line in orig:
            string = string + line
    string = string.replace('brth', '').replace('\n', ' ').replace('  ', ' ').replace('  ', ' ')
    with open(temp_file, 'w') as temp:
        temp.write(string)
    os.system('{} -o {} {}'.format(text2wave, out_file, temp_file))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Provide name of data to be processed, e.g. f2bs01p1'
        print 'puts stuff into ../boston-samples'
        print 'python bostontogrid.py f2bs01p1'
        exit()
    name = sys.argv[1]

    path = get_path(name)
    out_path = os.path.join('..', 'boston-samples')
    # make sure out_path exists
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # build text grid.
    # described in filedesc.doc

    tones = get_entry(path, name, '.ton', 3)
    labels = get_entry(path, name, '.lbl', 4)
    words = get_entry(path, name, '.wrd', 4)
    pos = get_pos(path, name, words)
    breaks = get_entry(path, name, '.brk', 4)

    end_time = labels[-1][1]

    write_textgrid(out_path, name, [breaks, tones, words, pos], ['breaks', 'tones', 'words', 'pos'], end_time)

    # copy original sound
    shutil.copy2(os.path.join(path, name + '.sph'), out_path)

    # text to speech, with no breath
    TTS(name, path, out_path)
