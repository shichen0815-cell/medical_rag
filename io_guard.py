# io_guard.py
import sys
import threading

class StdoutToStderr:
    """
    将 stdout 重定向到 stderr，避免破坏 input 光标
    """
    def __init__(self):
        self._stderr = sys.stderr
        self._lock = threading.Lock()

    def write(self, message):
        with self._lock:
            self._stderr.write(message)
            self._stderr.flush()

    def flush(self):
        self._stderr.flush()


def hijack_stdout():
    """
    项目级 stdout 劫持
    """
    sys.stdout = StdoutToStderr()
