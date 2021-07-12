import os
import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',
                      help='preprocessing mode',
                      default='txt', choices=['csv', 'txt'], type=str)
    parser.add_argument('--text_path',
                      help='path of text file',
                      required=True, type=str)
    parser.add_argument('--label_path',
                      help='path of label file',
                      required=True, type=str)
    parser.add_argument('--save_path',
                      help='path of merged file',
                      required=True, type=str)
    parser.add_argument('--save_label_path',
                      help='path of label file',
                      default=None, type=str)
    args = parser.parse_args()
    return args

def count_line_num(fpath):
    line_num = sum(1 for _ in open(fpath))
    return line_num

def main():
    args = parse_args()
    tpath = args.text_path
    lpath = args.label_path
    spath = args.save_path
    assert count_line_num(tpath) == count_line_num(lpath)

    if args.mode == 'txt':
        with open(tpath) as text, open(lpath) as label, open(spath, 'w') as sf:
            for tline in text:
                tline = tline.strip()
                lline = label.readline().strip()
                sf.write('%s\t%s\n'%(lline, tline))
    else:
        assert args.save_label_path is not None
        slpath = args.save_label_path

        labels = set()
        with open(lpath) as label:
            for line in label:
                line = line.strip().split('\t')[1]
                labels.update(line.split())
            labels = sorted(list(labels))

        l2d = dict()
        with open(slpath, 'w') as slf:
            for i, l in enumerate(labels):
                l2d[l] = i
                slf.write('\t%d\n'%i)

        with open(tpath) as text, open(lpath) as label, open(spath, 'w') as sf:
            for tline in text:
                tline = tline.strip().split('\t')[-1]
                lline = label.readline().strip().split('\t')[1]
                ts = ['%s'%t for t in tline.split()]
                ts = ','.join(ts)
                ls = ['%d'%l2d[l] for l in lline.split()]
                ls = ','.join(ls)
                sf.write('%s\t%s\n'%(ls, ts))


if __name__ == "__main__":
    main()
