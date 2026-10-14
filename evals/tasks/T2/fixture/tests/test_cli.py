import subprocess
import sys


def run(*args, stdin=None):
    return subprocess.run([sys.executable, "-m", "wordfreq", *args], input=stdin,
                          capture_output=True, text=True)


def test_stdin():
    r = run("-", stdin="x y x")
    assert r.returncode == 0
    assert r.stdout == "x 2\ny 1\n"


def test_top():
    r = run("-", "--top", "1", stdin="x y x")
    assert r.stdout == "x 2\n"
