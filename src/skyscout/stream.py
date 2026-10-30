"""GStreamer RTSP server fed with BGR frames from the detection worker."""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

LOG = logging.getLogger(__name__)


class RtspStreamer:
    def __init__(self, width: int, height: int, fps: int, port: int, mount: str,
                 bitrate_kbps: int, encoder_pipeline: str | None = None):
        try:
            import gi
            gi.require_version("Gst", "1.0")
            gi.require_version("GstRtspServer", "1.0")
            from gi.repository import GLib, Gst, GstRtspServer
        except (ImportError, ValueError) as exc:
            raise RuntimeError("GStreamer Python bindings are missing; install python3-gi, gir1.2-gstreamer-1.0, and gir1.2-gst-rtsp-server-1.0") from exc

        Gst.init(None)
        self.GLib, self.Gst = GLib, Gst
        self.width, self.height, self.fps = width, height, fps
        self._lock = threading.Lock()
        self._appsrc = None
        self._start_ns = time.monotonic_ns()
        self._last_pts = -1
        self.loop = GLib.MainLoop()
        self.server = GstRtspServer.RTSPServer()
        self.server.set_service(str(port))

        if encoder_pipeline is None:
            encoder_pipeline = (f"x264enc tune=zerolatency speed-preset=ultrafast "
                                f"bitrate={bitrate_kbps} key-int-max={fps * 2} bframes=0")
        launch = (
            f"( appsrc name=source is-live=true format=time block=false "
            f"caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 "
            f"! queue max-size-buffers=2 leaky=downstream ! videoconvert "
            f"! {encoder_pipeline} ! h264parse config-interval=-1 "
            f"! rtph264pay name=pay0 pt=96 config-interval=1 )"
        )
        # Fail at startup if the selected encoder or its pipeline is unavailable.
        pipeline = Gst.parse_launch(launch)
        pipeline.set_state(Gst.State.NULL)
        factory = GstRtspServer.RTSPMediaFactory()
        factory.set_launch(launch)
        factory.set_shared(True)
        factory.connect("media-configure", self._on_media_configure)
        self.server.get_mount_points().add_factory(mount, factory)
        if not self.server.attach(None):
            raise RuntimeError(f"failed to start RTSP server on port {port}")
        LOG.info("RTSP ready at rtsp://<device-ip>:%d%s", port, mount)

    def _on_media_configure(self, _factory, media):
        appsrc = media.get_element().get_by_name("source")
        if appsrc is None:
            raise RuntimeError("RTSP pipeline has no appsrc")
        appsrc.set_property("max-bytes", self.width * self.height * 3 * 2)
        if appsrc.find_property("leaky-type"):
            appsrc.set_property("leaky-type", 2)  # downstream: retain newer frames
        with self._lock:
            # A new RTSP media session needs timestamps starting near its own clock origin.
            self._start_ns = time.monotonic_ns()
            self._last_pts = -1
            self._appsrc = appsrc
        media.connect("unprepared", self._on_media_unprepared, appsrc)

    def _on_media_unprepared(self, _media, appsrc):
        with self._lock:
            if self._appsrc is appsrc:
                self._appsrc = None

    def push(self, frame: np.ndarray) -> None:
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise ValueError(f"RTSP frame must be BGR uint8 {self.width}x{self.height}, got {frame.shape} {frame.dtype}")
        with self._lock:
            appsrc = self._appsrc
            if appsrc is None:
                return
            raw = np.ascontiguousarray(frame).tobytes()
            buffer = self.Gst.Buffer.new_allocate(None, len(raw), None)
            buffer.fill(0, raw)
            pts = max(time.monotonic_ns() - self._start_ns, self._last_pts + 1)
            self._last_pts = pts
            buffer.pts = pts
            buffer.duration = self.Gst.CLOCK_TIME_NONE
            result = appsrc.emit("push-buffer", buffer)
            if result != self.Gst.FlowReturn.OK:
                LOG.debug("RTSP frame dropped: %s", result)

    def run(self) -> None:
        self.loop.run()

    def stop(self) -> None:
        self.GLib.idle_add(self.loop.quit)
