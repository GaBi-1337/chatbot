"""Build-time check for hermes-tool-whitelist.patch: a fork's tool whitelist must hold on the worker threads that
run its tools, not only on the fork's own thread. Fails on unpatched Hermes v0.21.7, where it was thread-local."""
import threading

from hermes_cli.plugins import clear_thread_tool_whitelist, get_pre_tool_call_directive, set_thread_tool_whitelist
from tools.daemon_pool import DaemonThreadPoolExecutor
from tools.thread_context import propagate_context_to_thread

seen = {}


def fork():
    # As in agent/background_review.py: set on the fork's thread, then tools run on pool workers.
    set_thread_tool_whitelist({"read_file"}, deny_msg_fmt="denied {tool_name}")
    try:
        seen["fork thread"] = get_pre_tool_call_directive("write_file", {})
        pool = DaemonThreadPoolExecutor(max_workers=1)
        seen["worker thread"] = pool.submit(propagate_context_to_thread(get_pre_tool_call_directive),
                                            "write_file", {}).result(timeout=60)
        pool.shutdown()
    finally:
        clear_thread_tool_whitelist()


t = threading.Thread(target=fork)
t.start()
t.join(120)
want = ("block", "denied write_file")
assert seen == {"fork thread": want, "worker thread": want}, f"tool whitelist not enforced: {seen}"
print("tool whitelist holds on worker threads")
