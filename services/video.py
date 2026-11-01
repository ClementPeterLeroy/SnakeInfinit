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
        self.decoded_frames = 0
        self.displayed_frames = 0
        self.waiting_elapsed = 0.0
        self.final_hold = 0.12
        self.thread = Thread(target=self._decode, name="startup-video", daemon=True)
        self.thread.start()

    def _decode(self):
        try:
            import av
            logging.info("Startup decoder PyAV=%s opening=%s", av.__version__, self.path)
            with av.open(str(self.path)) as container:
                if not container.streams.video:
                    raise ValueError("Startup file has no video stream")
                if container.streams.audio:
                    logging.warning("Startup video contains audio; embedded audio is not supported by this player")
                stream = container.streams.video[0]
                if stream.duration is not None and stream.time_base is not None:
                    self.duration = float(stream.duration * stream.time_base)
                elif container.duration:
                    self.duration = container.duration / av.time_base
                rate = float(stream.average_rate or 30)
                logging.info("Startup stream codec=%s size=%sx%s nominal_rate=%s duration=%.3fs audio_streams=%s",
                             stream.codec_context.name, stream.width, stream.height, rate,
                             self.duration, len(container.streams.audio))
                first_time = None
                for index, frame in enumerate(container.decode(video=0)):
                    if self.stop.is_set():
                        break
                    pts = float(frame.time) if frame.time is not None else index / rate
                    if first_time is None:
                        first_time = pts
                    self.duration = max(self.duration, pts - first_time + 1 / rate)
                    rgb = frame.reformat(format="rgb24")
                    plane = rgb.planes[0]
                    raw = bytes(plane)
                    stride, row_bytes = plane.line_size, rgb.width * 3
                    packed = raw if stride == row_bytes else b"".join(
                        raw[y * stride:y * stride + row_bytes] for y in range(rgb.height))
                    item = (pts - first_time, packed, (rgb.width, rgb.height))
                    self.decoded_frames += 1
                    while not self.stop.is_set():
                        try:
                            self.frames.put(item, timeout=0.05)
                            break
                        except Full:
                            pass
                if self.decoded_frames == 0 and not self.stop.is_set():
                    raise ValueError("Startup video produced no decodable frames")
        except Exception as error:
            self.error = str(error)
            logging.exception("Startup video unavailable (%s): %s. Continuing to menu; verify installed requirements and bundled assets.", self.path, error)
        finally:
            self.done.set()

    def update(self, dt):
        if self.started:
            self.elapsed += dt
        else:
            self.waiting_elapsed += dt
            if self.waiting_elapsed >= 10 and not self.done.is_set():
                self.error = "Startup decoder did not produce a first frame within 10 seconds"
                logging.error(self.error)
                self.stop.set()
                return True
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
            if not self.started:
                logging.info("Startup first frame displayed at video timestamp %.3fs", pts)
            self.displayed_frames += 1
            self.started = True
            self.pending = None
        return (self.error is not None and self.done.is_set()
                or self.started and self.elapsed >= self.duration + self.final_hold and self.done.is_set()
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
        logging.info("Startup decoder closed: decoded=%s displayed=%s elapsed=%.3fs error=%s thread_alive=%s",
                     self.decoded_frames, self.displayed_frames, self.elapsed, self.error, self.thread.is_alive())
