"""Readable durations and recent-completion ETA without per-step dashboard work."""
import math


def duration(seconds):
    if seconds is None or not math.isfinite(seconds) or seconds<0:return 'Calculating…'
    seconds=int(seconds)
    days,seconds=divmod(seconds,86400);hours,seconds=divmod(seconds,3600);minutes,seconds=divmod(seconds,60)
    if days:return f'{days}d {hours:02d}h {minutes:02d}m'
    if hours:return f'{hours}h {minutes:02d}m {seconds:02d}s'
    if minutes:return f'{minutes}m {seconds:02d}s'
    return f'{seconds}s'


class CompletionRate:
    def __init__(self):self.previous=None;self.rate=None;self.count=0

    def update(self,completed,active_seconds):
        if self.previous is not None:
            delta=completed-self.previous[0];elapsed=active_seconds-self.previous[1]
            if delta>0 and elapsed>0:
                sample=delta/elapsed
                self.rate=sample if self.rate is None else .25*sample+.75*self.rate
                self.count+=delta
        self.previous=(completed,active_seconds)

    def remaining(self,episodes):return episodes/self.rate if self.count>=5 and self.rate else None
