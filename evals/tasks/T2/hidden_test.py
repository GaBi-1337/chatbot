import os
import subprocess
import sys

W = os.environ["WORKDIR"]


def run(*args, stdin=None):
    return subprocess.run([sys.executable, "-m", "wordfreq", *args], input=stdin, cwd=W,
                          capture_output=True, text=True)


def test_basic_exclude(tmp_path):
    ex = tmp_path / "e.txt"
    ex.write_text("the\nof\n")
    r = run("-", "--exclude-file", str(ex), stdin="The cat of the hat. Of course cat")
    assert r.returncode == 0
    assert r.stdout == "cat 2\ncourse 1\nhat 1\n"


def test_comments_blanks_whitespace_case(tmp_path):
    ex = tmp_path / "e.txt"
    ex.write_text("# comment\n\n   THE  \n  # not a word\nCat\n")
    r = run("-", "--exclude-file", str(ex), stdin="the cat the dog dog bird")
    assert r.stdout == "dog 2\nbird 1\n"


def test_top_applies_after_exclusion(tmp_path):
    ex = tmp_path / "e.txt"
    ex.write_text("a\n")
    r = run("-", "--top", "2", "--exclude-file", str(ex), stdin="a a a a b b c d")
    assert r.stdout == "b 2\nc 1\n"


def test_file_input_and_empty_exclude(tmp_path):
    ex = tmp_path / "e.txt"
    ex.write_text("")
    t = tmp_path / "t.txt"
    t.write_text("x y x")
    r = run(str(t), "--exclude-file", str(ex))
    assert r.stdout == "x 2\ny 1\n"


def test_missing_exclude_file(tmp_path):
    r = run("-", "--exclude-file", str(tmp_path / "nope.txt"), stdin="a b")
    assert r.returncode == 2
    assert r.stderr.strip() != ""
    assert r.stdout == ""


def test_without_flag_unchanged():
    r = run("-", stdin="b a b")
    assert r.stdout == "b 2\na 1\n"
