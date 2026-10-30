"""Bounded worker-thread MP4 decoding; presentation and event handling stay live."""
import logging
from queue import Empty, Full, Queue
from threading import Event, Thread
import pygame
from ui.components import BG


class StartupVideo:
    def __init__(self, path):
        self.path = path
        self.frames = Queue(maxsize=3)
        self.stop = Event()
        self.done = Event()
        self.error = None
        self.duration = 6.0
        self.elapsed = 0.0
        self.started = False
        self.current = None
        self.pending = None
        self.thread = Thread(target=self._decode, name="startup-video", daemon=True)
        self.thread.start()

    def _decode(self):
        try:
            import av
            with av.open(str(self.path)) as container:
                if not container.streams.video:
                    raise ValueError("Startup file has no video stream")
                if container.duration:
                    self.duration = container.duration / av.time_base
                if container.streams.audio:
                    logging.warning("Startup video contains audio; embedded audio is not supported by this player")
                stream = container.streams.video[0]
                rate = float(stream.average_rate or 30)
                first_time = None
                for index, frame in enumerate(container.decode(video=0)):
                    if self.stop.is_set():
                        break
                    pts = float(frame.time) if frame.time is not None else index / rate
                    if first_time is None:
                        first_time = pts
                    rgb = frame.reformat(format="rgb24")
                    plane = rgb.planes[0]
                    raw = bytes(plane)
                    stride, row_bytes = plane.line_size, rgb.width * 3
                    packed = raw if stride == row_bytes else b"".join(
                        raw[y * stride:y * stride + row_bytes] for y in range(rgb.height))
                    item = (pts - first_time, packed, (rgb.width, rgb.height))
                    while not self.stop.is_set():
                        try:
                            self.frames.put(item, timeout=0.05)
                            break
                        except Full:
                            pass
        except Exception as error:
            self.error = str(error)
            logging.warning("Startup video unavailable (%s): %s", self.path, error)
        finally:
            self.done.set()

    def update(self, dt):
        if self.started:
            self.elapsed += dt
        while True:
            if self.pending is None:
                try:
                    self.pending = self.frames.get_nowait()
                except Empty:
                    break
            pts, raw, size = self.pending
            if self.started and pts > self.elapsed:
                break
            self.current = pygame.image.frombuffer(raw, size, "RGB").copy()
            self.started = True
            self.pending = None
        return (self.error is not None and self.done.is_set()
                or self.started and self.elapsed >= self.duration and self.done.is_set()
                and self.frames.empty() and self.pending is None
                or self.done.is_set() and not self.started and self.frames.empty())

    def draw(self, surface):
        surface.fill(BG)
        if self.current is not None:
            rect = self.current.get_rect().fit(surface.get_rect())
            surface.blit(pygame.transform.smoothscale(self.current, rect.size), rect)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1.0)
