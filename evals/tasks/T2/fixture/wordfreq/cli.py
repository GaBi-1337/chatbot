import argparse
import sys

from .core import count_words, top_words


def build_parser():
    p = argparse.ArgumentParser(prog="wordfreq", description="Count word frequencies in a text file.")
    p.add_argument("file", help="text file to read, or - for stdin")
    p.add_argument("--top", type=int, default=None, help="only show the N most frequent words")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    text = sys.stdin.read() if args.file == "-" else open(args.file, encoding="utf-8").read()
    for word, n in top_words(count_words(text), args.top):
        print(f"{word} {n}")
    return 0
