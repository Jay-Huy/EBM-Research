"""System-wide logger, used instead of print.

- `Logger`: log.info / log.warning / log.error / log.debug.
  - Always prints to the terminal.
  - With `log_file`: also writes to that file.
  - With `config`: starts with a header (run info + git commit + config).
- `TestLogger`: subclass for tests; writes to tests/logs/<name>.log and always has a header.

Example:
    log = Logger("train", log_file="results/ET-sudoku/train.log", config=cfg)
    log.info("step %d  loss %.4f", step, loss)
"""
import datetime
import json
import logging
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_LOG_DIR = REPO_ROOT / "tests" / "logs"
FORMAT = "%(asctime)s %(levelname)-7s %(message)s"


def _git(*args):
    """Run a git command.

    Returns:
        (ok, stdout) on success, or (False, first line of the error).
    """
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
    except FileNotFoundError:
        return False, "git command not found"
    if out.returncode == 0:
        return True, out.stdout.strip()
    return False, (out.stderr.strip().splitlines() or ["git error"])[0]


def git_commit():
    """Current git commit.

    Returns one of:
        - the commit hash, with " + uncommitted changes" if the working tree is dirty;
        - "no commit yet";
        - "cannot read git: <first line of the git error>".
    """
    ok, msg = _git("rev-parse", "--is-inside-work-tree")
    if not ok:
        return f"cannot read git: {msg}"
    ok, head = _git("rev-parse", "--verify", "-q", "HEAD")
    if not ok:
        return "no commit yet"
    _, dirty = _git("status", "--porcelain")
    return head + (" + uncommitted changes" if dirty else "")


def run_info(name, config):
    """Header of one run: name, time, host, platform, python, git commit, config."""
    return {
        "name": name,
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "git_commit": git_commit(),
        "config": config,
    }


class Logger:
    """Logger for the whole system.

    - Always prints to the terminal.
    - `log_file` given: also writes to that file (overwritten on each run; mode="a" appends, e.g. resume).
    - `config` given: starts with a header (`run_info`).
    """

    def __init__(self, name, log_file=None, config=None, level=logging.INFO, mode="w"):
        self._log = logging.getLogger(name)
        self._log.setLevel(level)
        self._log.propagate = False
        self.close()                                  # drop old handlers if re-created with the same name
        handlers = [logging.StreamHandler(sys.stdout)]
        if log_file is not None:
            self.path = Path(log_file)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(self.path, mode=mode, encoding="utf-8"))
        for h in handlers:
            h.setFormatter(logging.Formatter(FORMAT, datefmt="%H:%M:%S"))
            self._log.addHandler(h)
        if config is not None:
            self.header(run_info(name, config))

    def header(self, info):
        self.info("=" * 72)
        for k, v in info.items():
            self.info(f"{k:<11}: {json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v}")
        self.info("=" * 72)

    def kv(self, key, value):
        """One line 'key        : value' (setup blocks of train.log / inference.log, keys aligned)."""
        self.info(f"{key:<10} : {value}")

    def debug(self, msg, *args):   self._log.debug(msg, *args)
    def info(self, msg, *args):    self._log.info(msg, *args)
    def warning(self, msg, *args): self._log.warning(msg, *args)
    def error(self, msg, *args):   self._log.error(msg, *args)

    def close(self):
        for h in list(self._log.handlers):
            h.close()
            self._log.removeHandler(h)


class TestLogger(Logger):
    """Logger for tests.

    - Writes to tests/logs/<name>.log.
    - Always starts with a header.
    """

    def __init__(self, name, config=None):
        super().__init__(name, log_file=TEST_LOG_DIR / f"{name}.log", config=config or {})
