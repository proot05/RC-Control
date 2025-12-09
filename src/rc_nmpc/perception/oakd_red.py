"""
OAK-D Pro adapter using SpatialDetectionNetwork (e.g., YOLOv6-nano) + ObjectTracker.
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
from typing import Optional, List, Tuple
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

# Default label index for "person" in COCO/YOLO models
PERSON_LABEL = 0

@dataclass
class OakdParams:
    # Prefer a model *name* like "yolov6-nano" (builder API). If not provided,
    # nn_path can point to a compiled blob (fallback path).
    nn_name: Optional[str]

    full_frame: bool = False        # Track on full RGB frame if True
    conf_thresh: float = 0.5
    depth_lower_mm: int = 100       # 0.1 m
    depth_upper_mm: int = 5000      # 5.0 m
    track_labels: List[int] = None  # e.g., [0] (person)

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

        # Config compatibility:
        # - Preferred: cfg["nnName"] like "yolov6-nano" (as in object_tracker.py)
        # - Fallback: cfg["MBN_nnPath"] or cfg["Yolo_nnPath"] blob path
        nn_name = cfg.get("nnName") or cfg.get("nnModelName")  # allow either key
        #nn_path = cfg.get("Yolo_nnPath") or cfg.get("MBN_nnPath") or cfg.get("nnPath")

        # If neither provided, default to yolov6-nano builder name
        if not nn_name:
            nn_name = "yolov6-nano"

        self.params = OakdParams(
            nn_name=nn_name,
            full_frame=bool(cfg.get("full_frame", False)),
            conf_thresh=float(cfg.get("confidence", 0.5)),
            depth_lower_mm=int(cfg.get("depth_lower_mm", 100)),
            depth_upper_mm=int(cfg.get("depth_upper_mm", 5000)),
            track_labels=list(cfg.get("track_labels", [PERSON_LABEL])),
        )

        self._q_tracklets: Optional[dai.DataOutputQueue] = None
        self._q_preview: Optional[dai.DataOutputQueue] = None  # frames for display (tracker passthrough)

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

        self.fps: float = 0
        self.counter: int = 0
        self.startTime: float = 0
        
        # added 11/17
        self._pipeline: Optional[dai.Pipeline] = None

    # ---- Device / pipeline bring-up
    def start(self):
        pipeline = dai.Pipeline()

        # Nodes
        camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        
        stereo = pipeline.create(dai.node.StereoDepth)
        leftOutput = monoLeft.requestOutput((640, 400))
        rightOutput = monoRight.requestOutput((640, 400))
        leftOutput.link(stereo.left)
        rightOutput.link(stereo.right)

        spatialDetectionNetwork = pipeline.create(dai.node.SpatialDetectionNetwork)
        # Try the "builder" pattern first (like object_tracker.py). If unavailable, fall back to manual links.
        used_builder = False
        try:
            if self.params.nn_name:
                # Newer API: attach camera+stereo within the node build
                spatialDetectionNetwork.build(camRgb, stereo, self.params.nn_name)  # e.g., "yolov6-nano"
                used_builder = True
        except Exception:
            used_builder = False

        if not used_builder:
            # Fallback: use blob + manual links
            if not self.params.nn_path:
                raise ValueError(
                    "No nnName (e.g., 'yolov6-nano') and no nnPath provided. "
                    "Set 'nnName' in perception.yaml or provide a compiled blob path."
                )
        
        objectTracker = pipeline.create(dai.node.ObjectTracker)

        # SpatialDetectionNetwork config (align with object_tracker.py)
        spatialDetectionNetwork.setConfidenceThreshold(self.params.conf_thresh)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(self.params.depth_lower_mm)
        spatialDetectionNetwork.setDepthUpperThreshold(self.params.depth_upper_mm)
        labelMap = spatialDetectionNetwork.getClasses()

        # Tracker props
        objectTracker.setDetectionLabelsToTrack(self.params.track_labels)  # track only person
        # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
        objectTracker.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS)
        # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
        objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)
        
        # input arguments below may be incorrect, TESTTTTTT
        
        self._q_preview = objectTracker.passthroughTrackerFrame.createOutputQueue(maxSize=8, blocking=False)
        self._q_tracklets = objectTracker.out.createOutputQueue(maxSize=8, blocking=False)

        if self.params.full_frame:
            camRgb.requestFullResolutionOutput().link(objectTracker.inputTrackerFrame)
            # do not block the pipeline if it's too slow on full frame
            objectTracker.inputTrackerFrame.setBlocking(False)
            objectTracker.inputTrackerFrame.setMaxSize(1)
        else:
            spatialDetectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)

        spatialDetectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
        spatialDetectionNetwork.out.link(objectTracker.inputDetections)

        pipeline.start()
        self._pipeline = pipeline # added 11/17

    # ---- Background reader thread (VisionObject)
    def start_background(self, rate_hz: float = 60.0):
        """Continuously poll and cache the latest VisionObject."""
        if self._bg_thread and self._bg_thread.is_alive():
            return
        self._stop_evt.clear()
        period = 1.0 / max(1.0, rate_hz)

        self.startTime = time.monotonic()

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

        track = self._q_tracklets.get()
        assert isinstance(track, dai.Tracklets), "Expected Tracklets"

        self.counter += 1
        current_time = time.monotonic()
        if (current_time - self.startTime) > 1:
            self.fps = self.counter / (current_time - self.startTime)
            self.counter = 0
            self.startTime = current_time

        # ---- NEW: pick nearest tracklet (nearest person) ----
        nearest = None
        nearest_z = 1e9  # very large initial value

        for t in track.tracklets:
            sc = t.spatialCoordinates
            z_m = float(sc.z) / 1000.0
            # Only consider realistic distances
            if 0.1 < z_m < nearest_z:
                nearest = t
                nearest_z = z_m

        if nearest is None:
            return None

        sc = nearest.spatialCoordinates
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
        - frame is BGR np.ndarray from tracker's passthrough (same resolution as preview_w/h if preview path is used).
        - bbox is computed from the *latest* tracklet's ROI projected into pixel coords.
        This call is non-blocking; it uses the most recent items from the queues.
        """
        if self._q_preview is None:
            return None

        img_msg = self._q_preview.get()
        assert isinstance(img_msg, dai.ImgFrame), "Expected ImgFrame"
        #if img_msg is None:
            #return None

        frame = img_msg.getCvFrame()  # BGR
        H, W = frame.shape[:2]

        bbox_xyxy: Optional[Tuple[int,int,int,int]] = None
        conf: Optional[float] = None

        if self._q_tracklets is not None:
            track = self._q_tracklets.get()
            assert isinstance(track, dai.Tracklets), "Expected Tracklets"
            if track is not None and track.tracklets:
                t = track.tracklets[0]  # smallest ID policy
                if hasattr(t, "roi"):
                    r = t.roi
                    tl = r.topLeft()
                    br = r.bottomRight()
                    x1 = max(0, min(int(tl.x * W), W - 1))
                    y1 = max(0, min(int(tl.y * H), H - 1))
                    x2 = max(0, min(int(br.x * W), W - 1))
                    y2 = max(0, min(int(br.y * H), H - 1))
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
                    cv.putText(frame, "NN fps: {:.2f}".format(self.fps), (2, frame.shape[0] - 4),
                               cv.FONT_HERSHEY_TRIPLEX, 0.4, (255, 255, 255))
                    cv.imshow(self._window_name, frame)
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        self._disp_stop_evt.set()
                        break
                except Exception as e:
                    print(e)
                    self._disp_stop_evt.set()
                    break

                time.sleep(period)

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
        
        # added 11/17
        try:
            if getattr(self, "_pipeline", None) is not None and self._pipeline.isRunning():
                self._pipeline.stop()
        except Exception:
            pass

        self._q_tracklets = None
        self._q_preview = None
        self._bg_thread = None
        self._latest = None
