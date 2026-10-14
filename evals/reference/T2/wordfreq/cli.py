import argparse
import sys

from .core import count_words, load_exclusions, top_words


def build_parser():
    p = argparse.ArgumentParser(prog="wordfreq", description="Count word frequencies in a text file.")
    p.add_argument("file", help="text file to read, or - for stdin")
    p.add_argument("--top", type=int, default=None, help="only show the N most frequent words")
    p.add_argument("--exclude-file", default=None, help="file of words to ignore, one per line")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    exclude = None
    if args.exclude_file:
        try:
            exclude = load_exclusions(args.exclude_file)
        except OSError as e:
            print(f"wordfreq: cannot read exclude file: {e}", file=sys.stderr)
            return 2
    text = sys.stdin.read() if args.file == "-" else open(args.file, encoding="utf-8").read()
    for word, n in top_words(count_words(text, exclude), args.top):
        print(f"{word} {n}")
    return 0
