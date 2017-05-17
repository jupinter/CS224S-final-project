import os
import sys

def get_folder(name):
    folder = os.path.join('..', 'boston-univ-radio-corpus', 'BU_RADIO_' + name[1], name[0:3])
    if name[3] == 's':
        folder = os.path.join(folder, 'radio', name[3:6])
    else:
        folder = os.path.join(folder, 'labnews', name[3], 'radio' if name[4:6] == 'rl' else 'nonradio')
    return folder

def get_tones(folder, name):
    file = os.path.join(folder, name + '.ton')
    tones = []
    start_time = 0.
    with open(file) as f:
        for line in f:
            if line == '#\n':
                break
        for line in f:
            elems = line.strip().split(' ')
            end_time = float(elems[0])
            text = elems[3]
            tones.append([start_time, end_time, text])
            start_time = end_time
    return tones

def get_labels(folder, name):
    file = os.path.join(folder, name + '.lbl')
    labels = []
    start_time = 0.
    with open(file) as f:
        for line in f:
            if line == '#\n':
                break
        for line in f:
            elems = line.strip().split(' ')
            end_time = float(elems[0])
            if end_time == 0 or str(end_time) == 'nan':
                continue
            text = elems[4]
            labels.append([start_time, end_time, text])
            start_time = end_time
    return labels

# for files of format end_time, junk, BLAH
def get_entry(folder, name, extension, text_index):
    file = os.path.join(folder, name + extension)
    ret = []
    start_time = 0.
    with open(file) as f:
        for line in f:
            if line == '#\n':
                break
        for line in f:
            elems = line.strip().split(' ')
            end_time = float(elems[0])
            if end_time == 0 or str(end_time) == 'nan':
                continue
            text = elems[text_index]
            ret.append([start_time, end_time, text])
            start_time = end_time
    return ret

def get_pos(folder, name, words):
    file = os.path.join(folder, name + '.pos')
    idx = 0
    pos = []
    with open(file) as f:
        for line in f:
            if line == '\n':
                continue
            elems = line.strip().split(' ')
            # word in words[idx] should be word in elems up to extra punctuation in elems
            if words[idx][2].lower() not in elems[0].lower():
                print 'mismatch: ', words[idx][2], elems[0]
            pos.append([words[idx][0], words[idx][1], elems[1]])
            idx += 1
    return pos

def get_aln(folder, name, labels):
    

def write_textgrid(layers, names, end_time):
    file = 'out.TextGrid'
    with open(file, 'w') as f:
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

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'give name of thingy to be converted'
        print 'e.g. f1as01p1 or f1ajrlp1'
        exit()
    name = sys.argv[1]

    folder = get_folder(name)

    tones = get_entry(folder, name, '.ton', 3)
    labels = get_entry(folder, name, '.lbl', 4)
    words = get_entry(folder, name, '.wrd', 4)
    pos = get_pos(folder, name, words)

    end_time = labels[-1][1]

    write_textgrid([tones, words, pos, labels], ['tones', 'words', 'pos', 'labels'], end_time)
