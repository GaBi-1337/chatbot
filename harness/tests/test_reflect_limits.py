"""Limits of `muse reflect` (harness/reflect.py), fed the call shapes Hermes v0.21.7 really sends.

skill_manage advertises ONLY an operations array (tools/skill_manager_tool.py SKILL_MANAGE_SCHEMA), even for one
change; memory batches the same way. The flat shape is legacy. A guard tested on the flat shape alone once refused
every real call.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reflect  # noqa: E402


def op(action, name, **kw):
    return {"action": action, "name": name, **kw}


def test_operations_batch_over_the_limit_is_refused():
    err, names = reflect.skill_call_check({"operations": [op("create", "a"), op("create", "b")]}, set(), 1)
    assert err and "1 left" in err and names == {"a", "b"}


def test_single_operation_within_the_limit_passes():
    assert reflect.skill_call_check({"operations": [op("create", "a")]}, set(), 1) == (None, {"a"})


def test_changing_an_already_changed_skill_does_not_count_again():
    err, _ = reflect.skill_call_check({"operations": [op("patch", "a", old_string="x", new_string="y")]}, {"a"}, 1)
    assert err is None


def test_deletes_are_refused_in_any_shape():
    assert reflect.skill_call_check({"operations": [op("delete", "a")]}, set(), 3)[0]
    assert reflect.skill_call_check({"operations": [op("remove_file", "a", file_path="x.md")]}, set(), 3)[0]
    assert reflect.skill_call_check({"action": "delete", "name": "a"}, set(), 3)[0]


def test_legacy_flat_shape_still_counts():
    assert reflect.skill_call_check({"action": "create", "name": "a"}, {"b"}, 1)[0]
    assert reflect.skill_call_check({"action": "create", "name": "a"}, set(), 1) == (None, {"a"})


def test_memory_batch_counts_every_entry():
    batch = {"operations": [{"action": "add", "target": "memory", "content": c} for c in "xyz"]}
    err, n = reflect.memory_call_check(batch, 0, 2)
    assert err and n == 3
    assert reflect.memory_call_check(batch, 0, 3) == (None, 3)


def test_memory_single_write_and_reads():
    assert reflect.memory_call_check({"action": "add", "target": "user", "content": "x"}, 4, 5) == (None, 1)
    assert reflect.memory_call_check({"action": "add", "target": "user", "content": "x"}, 5, 5)[0]
    assert reflect.memory_call_check({"action": "read"}, 5, 5) == (None, 0)
