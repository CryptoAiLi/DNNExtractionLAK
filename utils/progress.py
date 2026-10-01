"""Share single-line progress displays and noninteractive stage summaries."""
import math
import os
import sys
import time
from datetime import datetime

from tqdm import tqdm


def log_progress(message):
    """Log a stage message while safely redrawing any active progress bar."""
    tqdm.write(f'[{datetime.now():%H:%M:%S}] {message}', file=sys.stdout)


class Progress:
    """Report absolute completed counts with context-managed cleanup.

    mode selects auto, bar, log, or off. Auto detects supported terminals,
    including PyCharm; use log when carriage-return updates are unsupported.
    interval throttles refreshes, completion always refreshes, and ETA is stage-local.
    """

    def __init__(self, label, total, interval=5.0, *, mode='auto', stream=None):
        if mode not in ('auto', 'bar', 'log', 'off'):
            raise ValueError('Unknown progress mode')
        if total < 0 or not math.isfinite(interval) or interval <= 0:
            raise ValueError('Expected nonnegative total and finite positive interval')
        self.label, self.total, self.interval = label, total, interval
        self.stream = sys.stdout if stream is None else stream
        interactive = self.stream.isatty() or (
            bool(os.environ.get('PYCHARM_HOSTED')) and self.stream is sys.__stdout__)
        self.mode = ('bar' if interactive else 'log') if mode == 'auto' else mode
        self.completed, self.detail, self.closed = 0, '', False
        self.started = self.last_report = time.perf_counter()
        self.bar = None
        if self.mode == 'bar':
            self.bar = tqdm(total=total, desc=label, file=self.stream,
                            dynamic_ncols=True, ascii=True, mininterval=interval,
                            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} '
                                       '[{elapsed}<{remaining}, {rate_fmt}]{postfix}')
        elif self.mode == 'log':
            self._write(f'{label}: start, total={total}')

    def _write(self, message):
        print(f'[{datetime.now():%H:%M:%S}] {message}', file=self.stream, flush=True)

    def update(self, completed, detail='', *, force=False):
        if self.closed:
            return
        self.completed, self.detail = completed, detail
        now = time.perf_counter()
        if self.bar is not None:
            # Update state without refreshing; throttle display updates in one place.
            self.bar.n = completed
            self.bar.set_postfix_str(detail, refresh=False)
            if force or completed == self.total or now - self.last_report >= self.interval:
                self.bar.refresh()
                self.last_report = now

    def close(self, *, interrupted=False):
        if self.closed:
            return
        self.closed = True
        status = 'interrupted' if interrupted else (
            'done' if self.completed == self.total else 'stopped')
        if self.bar is not None:
            self.bar.set_postfix_str(f'{status} {self.detail}', refresh=False)
            self.bar.close()
        elif self.mode == 'log':
            self._write(f'{self.label}: {status}, {self.completed}/{self.total} '
                        f'elapsed={time.perf_counter() - self.started:.1f}s {self.detail}')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close(interrupted=exc_type is not None)
