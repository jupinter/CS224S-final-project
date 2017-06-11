from __future__ import division
import matplotlib.pyplot as plt
import os
import numpy as np

predict_dir = '../ATrampAbroad/predictions'
true_file = '../ATrampAbroad/pitches_test.txt'

true_starts = []
true_middles = []
true_ends = []
pred_starts = []
pred_middles = []
pred_ends = []
with open(true_file) as true:
    true.readline()
    curr_file = ''
    for true_line in true:
        true_line = true_line.strip().split('\t')
        new_file = true_line[0][-11:]
        if curr_file != new_file:
            if curr_file != '':
                pred.close() #end pred early, if true is actually shorter
            curr_file = new_file
            pred_file = os.path.join(predict_dir, curr_file + '.txt')
            try:
                pred = open(pred_file)
            except IOError:
                # all done, since batching
                break
        pred_line = pred.readline() 
        if pred_line != '':
            pred_vals = pred_line.strip().split('\t')
            true_vals = true_line[5:8]
            try:
                t = float(true_vals[0])
                p = float(pred_vals[0])
                true_starts += [t]
                pred_starts += [p]
            except ValueError: # value error: --undefined-- in true
                pass
            try:
                t = float(true_vals[1])
                p = float(pred_vals[1])
                true_middles += [t]
                pred_middles += [p]
            except ValueError: # value error: --undefined-- in true
                pass
            try:
                t = float(true_vals[2])
                p = float(pred_vals[2])
                true_ends += [t]
                pred_ends += [p]
            except ValueError: # value error: --undefined-- in true
                pass
        else:
            pass

true_starts = np.array(true_starts)
true_middles = np.array(true_middles)
true_ends = np.array(true_ends)
pred_starts = np.array(pred_starts)
pred_middles = np.array(pred_middles)
pred_ends = np.array(pred_ends)

start_mean = np.mean(true_starts)
start_std = np.std(true_starts)
mid_mean = np.mean(true_middles)
mid_std = np.std(true_middles)
end_mean = np.mean(true_ends)
end_std = np.std(true_ends)

num_start = len(true_starts)
num_mid = len(true_middles)
num_end = len(true_ends)

print 'num start:', num_start
print 'num   mid:', num_mid
print 'num   end:', num_end

print 'start mean:', start_mean
print 'start  std:', start_std
print 'mid   mean:', mid_mean
print 'mid    std:', mid_std
print 'end   mean:', end_mean
print 'end    std:', end_std

start_diff = true_starts - pred_starts
mid_diff = true_middles - pred_middles
end_diff = true_ends - pred_ends
print 'start true - pred range:', min(start_diff), max(start_diff)
print 'mid   true - pred range:', min(mid_diff), max(mid_diff)
print 'end   true - pred range:', min(mid_diff), max(mid_diff)

for percent in [.02, .05, .1, .15, .25]:
    print 'Within {} standard deviations:'.format(percent)
    s = sum(1 for x in pred_starts if abs(x - start_mean) < percent * start_std)
    m = sum(1 for x in pred_middles if abs(x - mid_mean) < percent * mid_std)
    e = sum(1 for x in pred_ends if abs(x - end_mean) < percent * mid_std)
    print ' start: {:.2f}% ({} / {})'.format(s / num_start * 100, s, num_start)
    print '   mid: {:.2f}% ({} / {})'.format(m / num_mid * 100, m, num_mid)
    print '   end: {:.2f}% ({} / {})'.format(e / num_end * 100, e, num_end)


# test cost 37590.058 | test param 1374103.360
