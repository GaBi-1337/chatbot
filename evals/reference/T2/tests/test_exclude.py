import subprocess
import sys


def test_exclude_file(tmp_path):
    ex = tmp_path / "ex.txt"
    ex.write_text("# stop\nThe\n\n")
    r = subprocess.run([sys.executable, "-m", "wordfreq", "-", "--exclude-file", str(ex)],
                       input="the a the b", capture_output=True, text=True)
    assert r.stdout == "a 1\nb 1\n"
