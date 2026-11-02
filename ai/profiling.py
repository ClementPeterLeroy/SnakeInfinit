"""Opt-in exclusive timers; no graph/math modifications or tensor hooks."""
from contextlib import contextmanager
import os
import sys
import time


class Profiler:
    def __init__(self,enabled=False):
        self.enabled=enabled
        self.seconds={}
        self.calls={}
        self.stack=[]

    @contextmanager
    def measure(self,name):
        if not self.enabled:
            yield
            return
        frame=[time.perf_counter(),0.]
        self.stack.append(frame)
        try:yield
        finally:
            elapsed=time.perf_counter()-frame[0]
            self.stack.pop()
            if self.stack:self.stack[-1][1]+=elapsed
            self.seconds[name]=self.seconds.get(name,0.)+max(0.,elapsed-frame[1])
            self.calls[name]=self.calls.get(name,0)+1

    def wrap(self,instance,method,name):
        original=getattr(instance,method)
        def measured(*args,**kwargs):
            with self.measure(name):return original(*args,**kwargs)
        setattr(instance,method,measured)

    def attach(self,trainer):
        if not self.enabled:return
        choosing=False
        original_choose=trainer.choose_action
        original_forward=trainer.online.forward
        def forward(*args,**kwargs):
            if choosing:
                with self.measure('neural_inference'):return original_forward(*args,**kwargs)
            return original_forward(*args,**kwargs)
        def choose(*args,**kwargs):
            nonlocal choosing
            choosing=True
            try:
                with self.measure('action_selection'):return original_choose(*args,**kwargs)
            finally:choosing=False
        trainer.online.forward=forward
        trainer.choose_action=choose
        for instance,method,name in ((trainer.environment,'observation','state_encoding'),
            (trainer.environment,'step','environment_step'),
            (trainer.replay,'sample','replay_sampling'),(trainer,'optimize','forward_backward'),
            (trainer.optimizer,'step','optimizer_update')):
            self.wrap(instance,method,name)

    def report(self):
        return {name:dict(seconds=self.seconds.get(name,0.),calls=self.calls.get(name,0)) for name in
            ('environment_step','state_encoding','action_selection','neural_inference','replay_sampling','forward_backward',
             'optimizer_update','metrics_aggregation','checkpoint_serialization','ui_synchronization')}


def process_usage():
    """Process CPU seconds and working-set bytes; missing measurements stay None."""
    cpu=os.times().user+os.times().system
    current=peak=None
    if sys.platform=='win32':
        try:
            import ctypes
            from ctypes import wintypes
            class Counters(ctypes.Structure):
                _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in
                    ('peak','working','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile')]
            info=Counters();info.cb=ctypes.sizeof(info)
            kernel=ctypes.WinDLL('kernel32',use_last_error=True)
            kernel.GetCurrentProcess.restype=wintypes.HANDLE
            psapi=ctypes.WinDLL('psapi',use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
            if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(info),info.cb):
                current,peak=info.working,info.peak
        except (OSError,AttributeError):pass
    else:
        try:
            import resource
            peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        except ImportError:pass
    return dict(cpu_seconds=cpu,ram_bytes=current,peak_ram_bytes=peak)
