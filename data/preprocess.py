import os
import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_path',
                      help='path of text file',
                      required=True, type=str)
    parser.add_argument('--label_path',
                      help='path of label file',
                      required=True, type=str)
    parser.add_argument('--save_path',
                      help='path of merged file',
                      required=True, type=str)
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

    with open(tpath) as text, open(lpath) as label, open(spath, 'w') as sf:
        for tline in text:
            tline = tline.strip()
            lline = label.readline().strip()
            sf.write('%s\t%s\n'%(lline, tline))

if __name__ == "__main__":
    main()
