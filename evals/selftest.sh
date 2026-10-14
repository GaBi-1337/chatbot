#!/usr/bin/env bash
# For each task: check must fail (exit 1) on untouched fixture and pass (exit 0) on fixture+reference.
# Reference dir: files are overlaid on the fixture copy; if apply.sh exists it is run in the copy afterwards.
HERE="$(cd "$(dirname "$0")" && pwd)"
S="$HERE/results/selftest"; rm -rf "$S"; mkdir -p "$S"
rc=0
for t in T1 T2 T3 T4 T5 T6 T7; do
  for mode in fixture reference; do
    w="$S/$t-$mode/work"; mkdir -p "$w"; cp -a "$HERE/tasks/$t/fixture/." "$w/"
    if [ $mode = reference ]; then
      for f in "$HERE/reference/$t"/*; do [ "$(basename "$f")" = apply.sh ] || cp -a "$f" "$w/"; done
      [ -x "$HERE/reference/$t/apply.sh" ] && (cd "$w" && "$HERE/reference/$t/apply.sh")
      want=0
    else want=1; fi
    msg=$("$HERE/tasks/$t/check.sh" "$w"); got=$?
    if [ $got = $want ]; then r=OK; else r=BAD; rc=1; fi
    printf '%s %-9s %-4s exit=%s (want %s) %s\n' "$t" "$mode" "$r" "$got" "$want" "$msg"
  done
done
rm -rf "$S"
[ $rc = 0 ] && echo "SELFTEST PASS" || echo "SELFTEST FAIL"
exit $rc
