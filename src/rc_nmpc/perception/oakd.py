"""
OAK-D Pro adapter using MobileNetSpatialDetectionNetwork + ObjectTracker.
Outputs VisionObject with (x_b, y_b, z_b) in *meters* in the camera/body frame.

Runtime features:
- start(): build pipeline and queues
- start_background(rate_hz): non-blocking reader thread that keeps latest VisionObject
- get_latest(max_age_ms): thread-safe snapshot with staleness check
- poll_object(): single non-blocking read
- start_display()/stop_display(): optional OpenCV viewer in its own thread (bbox overlay)
- get_display_packet(max_age_ms): returns (frame_bgr, bbox_xyxy|None, conf|None)
- close(): clean shutdown

Used by run_vehicle.py:
- Press B for 'simple' mode (bearing-only steering + constant accel)
"""
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union
import time
import threading

import depthai as dai

try:
    import cv2 as cv
    _HAS_CV = True
except Exception:
    _HAS_CV = False

from rc_nmpc.utils.messages import VisionObject
from rc_nmpc.utils.config import load_yaml

# Default label index for "person" in MobileNetSSD VOC model
PERSON_LABEL = 15

@dataclass
class OakdParams:
    nn_path: str
    full_frame: bool = False        # Track on full RGB frame if True
    conf_thresh: float = 0.5
    depth_lower_mm: int = 100       # 0.1 m
    depth_upper_mm: int = 5000      # 5.0 m
    preview_w: int = 300
    preview_h: int = 300
    mono_res: str = "THE_400_P"     # Mono camera resolution
    rgb_res: str  = "THE_1080_P"    # RGB camera resolution
    track_labels: List[int] = None  # e.g., [15] (person)

class OakdTracker:
    """
    Lifecycle:
      - start(): creates pipeline, starts device, prepares queues
      - start_background(): spins a reader thread to keep a fresh snapshot
      - get_latest(): returns last VisionObject if fresh enough
      - poll_object(): single non-blocking read (no thread required)
      - start_display()/stop_display(): show camera window with bbox in its own thread
      - get_display_packet(): get (frame, bbox, conf) without drawing
      - close(): releases resources
    """
    def __init__(self, config_path: str):
        cfg = load_yaml(config_path)
        nn_path = cfg.get("nnPath")
        if not nn_path:
            raise ValueError("perception.yaml must define nnPath pointing to mobilenet-ssd blob")

        self.params = OakdParams(
            nn_path=nn_path,
            full_frame=bool(cfg.get("full_frame", False)),
            conf_thresh=float(cfg.get("confidence", 0.5)),
            depth_lower_mm=int(cfg.get("depth_lower_mm", 100)),
            depth_upper_mm=int(cfg.get("depth_upper_mm", 5000)),
            preview_w=int(cfg.get("preview_w", 300)),
            preview_h=int(cfg.get("preview_h", 300)),
            track_labels=list(cfg.get("track_labels", [PERSON_LABEL])),
        )

        self._device: Optional[dai.Device] = None
        self._q_tracklets: Optional[dai.DataOutputQueue] = None
        self._q_preview: Optional[dai.DataOutputQueue] = None  # NEW: frames for display

        # Background snapshot machinery (object)
        self._latest: Optional[VisionObject] = None
        self._lock = threading.Lock()

        # Display support
        self._disp_stop_evt = threading.Event()
        self._disp_thread: Optional[threading.Thread] = None
        self._window_name: str = "OAK-D"

        # Generic stop for bg vision thread
        self._stop_evt = threading.Event()
        self._bg_thread: Optional[threading.Thread] = None

    # ---- Device / pipeline bring-up
    def start(self):
        p = dai.Pipeline()

        # Nodes
        camRgb = p.create(dai.node.ColorCamera)
        monoLeft = p.create(dai.node.MonoCamera)
        monoRight = p.create(dai.node.MonoCamera)
        stereo = p.create(dai.node.StereoDepth)
        ssd = p.create(dai.node.MobileNetSpatialDetectionNetwork)
        tracker = p.create(dai.node.ObjectTracker)

        # Outputs
        xoutTrack = p.create(dai.node.XLinkOut)
        xoutTrack.setStreamName("tracklets")

        # NEW: preview frames for display (synced to SSD passthrough)
        xoutPrev = p.create(dai.node.XLinkOut)
        xoutPrev.setStreamName("preview")

        # Camera props
        camRgb.setPreviewSize(self.params.preview_w, self.params.preview_h)
        camRgb.setResolution(getattr(dai.ColorCameraProperties.SensorResolution, self.params.rgb_res))
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        monoLeft.setResolution(getattr(dai.MonoCameraProperties.SensorResolution, self.params.mono_res))
        monoLeft.setCamera("left")
        monoRight.setResolution(getattr(dai.MonoCameraProperties.SensorResolution, self.params.mono_res))
        monoRight.setCamera("right")

        # Stereo props
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

        # Spatial SSD props
        ssd.setBlobPath(self.params.nn_path)
        ssd.setConfidenceThreshold(self.params.conf_thresh)
        ssd.input.setBlocking(False)
        ssd.setBoundingBoxScaleFactor(0.5)
        ssd.setDepthLowerThreshold(self.params.depth_lower_mm)
        ssd.setDepthUpperThreshold(self.params.depth_upper_mm)

        # Tracker props
        tracker.setDetectionLabelsToTrack(self.params.track_labels)  # e.g., [15] for person
        tracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
        tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

        # Links
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        camRgb.preview.link(ssd.input)
        stereo.depth.link(ssd.inputDepth)

        if self.params.full_frame:
            camRgb.setPreviewKeepAspectRatio(False)
            camRgb.video.link(tracker.inputTrackerFrame)
            tracker.inputTrackerFrame.setBlocking(False)
            tracker.inputTrackerFrame.setQueueSize(2)
        else:
            ssd.passthrough.link(tracker.inputTrackerFrame)

        ssd.passthrough.link(tracker.inputDetectionFrame)
        ssd.out.link(tracker.inputDetections)

        tracker.out.link(xoutTrack.input)

        # NEW: also send passthrough frames out for GUI
        ssd.passthrough.link(xoutPrev.input)

        # Start device
        self._device = dai.Device(p)
        self._q_tracklets = self._device.getOutputQueue("tracklets", maxSize=4, blocking=False)
        self._q_preview   = self._device.getOutputQueue("preview",   maxSize=4, blocking=False)

    # ---- Background reader thread (VisionObject)
    def start_background(self, rate_hz: float = 60.0):
        """Continuously poll and cache the latest VisionObject."""
        if self._bg_thread and self._bg_thread.is_alive():
            return
        self._stop_evt.clear()
        period = 1.0 / max(1.0, rate_hz)

        def _loop():
            while not self._stop_evt.is_set():
                try:
                    vo = self.poll_object()
                    if vo is not None:
                        with self._lock:
                            self._latest = vo
                except Exception:
                    pass
                time.sleep(period)

        self._bg_thread = threading.Thread(target=_loop, daemon=True)
        self._bg_thread.start()

    def get_latest(self, max_age_ms: int = 200) -> Optional[VisionObject]:
        """Return the freshest VisionObject if newer than max_age_ms; else None."""
        with self._lock:
            vo = self._latest
        if not vo:
            return None
        age_ms = (time.monotonic_ns() - vo.t_cam) / 1e6
        return vo if age_ms <= max_age_ms else None

    # ---- Single non-blocking read (no thread required)
    def poll_object(self) -> Optional[VisionObject]:
        """
        Non-blocking: returns VisionObject with spatialCoordinates (meters)
        from the tracker, or None if no new data.
        """
        if self._q_tracklets is None:
            return None

        track = self._q_tracklets.tryGet()
        if track is None or not track.tracklets:
            return None

        # Choose first/primary tracklet (policy: SMALLEST_ID)
        t = track.tracklets[0]
        sc = t.spatialCoordinates  # millimeters
        x_m = float(sc.x) / 1000.0
        y_m = float(sc.y) / 1000.0
        z_m = float(sc.z) / 1000.0

        # DepthAI spatial frame: X right (+), Y down (+), Z forward (+)
        return VisionObject(
            t_cam=time.monotonic_ns(),
            x_b=x_m, y_b=y_m, z_b=z_m,
            conf=1.0  # tracker doesn't expose class confidence; use 1.0
        )

    # ---- Display packet (frame + bbox) for external rendering
    def get_display_packet(self, max_age_ms: int = 200) -> Optional[Tuple["np.ndarray", Optional[Tuple[int,int,int,int]], Optional[float]]]:
        """
        Returns (frame_bgr, bbox_xyxy|None, conf|None), or None if no fresh frame.
        - frame is BGR np.ndarray from ssd.passthrough (same resolution as preview_w/h).
        - bbox is computed from the *latest* tracklet's ROI projected into pixel coords.
        This call is non-blocking; it uses the most recent items from the queues.
        """
        if self._q_preview is None:
            return None

        # Try to get a frame (most recent)
        img_msg = self._q_preview.tryGet()
        if img_msg is None:
            return None

        frame = img_msg.getCvFrame()  # BGR
        H, W = frame.shape[:2]

        bbox_xyxy: Optional[Tuple[int,int,int,int]] = None
        conf: Optional[float] = None

        # Try to pull a tracklet and make a bbox (optional)
        if self._q_tracklets is not None:
            track = self._q_tracklets.tryGet()
            if track is not None and track.tracklets:
                t = track.tracklets[0]  # smallest ID policy
                # roi is normalized rect in [0..1]
                if hasattr(t, "roi"):
                    r = t.roi
                    # Top-left & bottom-right normalized points
                    tl = r.topLeft()
                    br = r.bottomRight()
                    x1 = max(0, min(int(tl.x * W), W - 1))
                    y1 = max(0, min(int(tl.y * H), H - 1))
                    x2 = max(0, min(int(br.x * W), W - 1))
                    y2 = max(0, min(int(br.y * H), H - 1))
                    # Ensure proper ordering
                    if x2 > x1 and y2 > y1:
                        bbox_xyxy = (x1, y1, x2, y2)
                        conf = 1.0  # tracker doesn't expose confidence

        return (frame, bbox_xyxy, conf)

    # ---- Built-in display thread (OpenCV window)
    def start_display(self, window_name: str = "OAK-D", rate_hz: float = 60.0):
        """
        Starts a separate thread that shows a live window with bbox overlay.
        Safe to call multiple times; it won't spawn duplicates.
        """
        if not _HAS_CV:
            print("[oakd] OpenCV not available; display disabled.")
            return
        if self._disp_thread and self._disp_thread.is_alive():
            return

        self._window_name = window_name
        self._disp_stop_evt.clear()
        period = 1.0 / max(1.0, rate_hz)

        def _loop():
            try:
                cv.namedWindow(self._window_name, cv.WINDOW_NORMAL)
                cv.resizeWindow(self._window_name, 960, 540)
            except Exception:
                pass

            while not self._disp_stop_evt.is_set():
                pkt = None
                try:
                    pkt = self.get_display_packet(max_age_ms=200)
                except Exception:
                    pkt = None

                if pkt is None:
                    time.sleep(0.01)
                    continue

                frame, bbox, conf = pkt
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    try:
                        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        if conf is not None:
                            cv.putText(frame, f"{conf:.2f}", (x1, max(0, y1 - 6)),
                                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
                    except Exception:
                        pass

                try:
                    cv.imshow(self._window_name, frame)
                    # make window responsive, allow 'q' to close viewer only
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        self._disp_stop_evt.set()
                        break
                except Exception:
                    # if no display (e.g. headless), just stop
                    self._disp_stop_evt.set()
                    break

                time.sleep(period)

            # Cleanup window
            try:
                cv.destroyWindow(self._window_name)
            except Exception:
                pass

        self._disp_thread = threading.Thread(target=_loop, daemon=True)
        self._disp_thread.start()

    def stop_display(self):
        """Stops the display thread and closes the window."""
        try:
            self._disp_stop_evt.set()
            if self._disp_thread and self._disp_thread.is_alive():
                self._disp_thread.join(timeout=1.0)
        except Exception:
            pass
        self._disp_thread = None

    # ---- Shutdown
    def close(self):
        # stop background threads
        try:
            self._stop_evt.set()
            if self._bg_thread:
                self._bg_thread.join(timeout=1.0)
        except Exception:
            pass

        # stop display thread
        self.stop_display()

        # release device/queues
        try:
            if self._device is not None:
                self._device.close()
        except Exception:
            pass

        self._device = None
        self._q_tracklets = None
        self._q_preview = None
        self._bg_thread = None
        self._latest = None
