"""ROS 2 (Jazzy) node: coloured pegs + sockets in the ROBOT frame (ROBOTIS `link1`).

Subscribes (time-synchronised):
  /camera/camera/color/image_raw                   (rgb8)
  /camera/camera/aligned_depth_to_color/image_raw  (16UC1, mm, aligned to colour)
  /camera/camera/color/camera_info                 (intrinsics)

Publishes (robot frame = robot_frame.frame_id, default link1, metres):
  ~/scene               std_msgs/String  JSON with every peg and socket  <- use this in the IK node
  ~/peg/<colour>        geometry_msgs/PointStamped  peg TOP-FACE centre (z = top of the peg)
  ~/socket/<shape>      geometry_msgs/PointStamped  socket centre on the board surface
  ~/status              std_msgs/String  calibration diagnostics (JSON, for omx_calibration)
  ~/annotated_image     sensor_msgs/Image
  ~/debug_mask          sensor_msgs/Image
  ~/markers             visualization_msgs/MarkerArray
  TF  link1 -> camera_link              (frozen calibration; RViz shows the cloud on the arm)
  TF  camera_color_optical_frame -> robot_base_marker
Services:
  ~/recalibrate         std_srvs/Trigger  forget the frozen camera pose and measure it again

Pipeline per frame:
  1. ArUco marker -> PnP pose -> refined with the depth plane around it -> averaged over
     `marker.freeze_frames` frames and FROZEN (the arm may cover the marker later).
  2. camera -> robot transform  p_r = A p_c + b  (marker pose, marker placement on the drawn
     axes, then drawn axes -> link1 from `robot_frame.*`, found with `frame_calibration`).
  3. Pegs: colour blob -> every pixel back-projected -> top face = highest points -> area
     centroid. Sockets: board plane from depth -> outline rays hit the plane -> area centroid.
  4. Median over the last few frames, then publish.
"""
import json
import os
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PointStamped, TransformStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from . import geometry3d as g3
from .colour_detection import ColourSpec, DetectionParams, detect_colours
from .coordinate_estimation import DepthParams, Intrinsics, deproject, depth_at_patch, robust_depth
from .image_conversion import bgr_to_image_msg, colour_msg_to_bgr, depth_msg_to_metres
from .marker_detection import detect_markers, estimate_pose, rotation_to_quaternion
from .shape_classification import classify_shape
from .socket_detection import SocketParams, detect_sockets

# Defaults; normally overridden by config/detection_params.yaml.
DEFAULT_COLOURS = {
    #  name     hsv_lower        hsv_upper        draw BGR          expected shape
    'yellow': ([18, 120, 120], [33, 255, 255], [0, 255, 255], 'triangle'),
    'green':  ([34, 80, 60],   [75, 255, 255], [0, 200, 0],   'circle'),
    'blue':   ([98, 120, 50],  [130, 255, 255], [255, 120, 0], 'square'),
    'orange': ([3, 130, 120],  [17, 255, 255], [0, 128, 255], 'hexagon'),
}


class PegDetector(Node):
    def __init__(self):
        super().__init__('peg_detector')
        self._declare_parameters()
        self._load_config()
        self.add_on_set_parameters_callback(self._on_param_change)
        self._reload = False

        self.intr: Optional[Intrinsics] = None
        self.dist = np.zeros(5)
        self._last_print = 0.0
        self._frame_times: List[float] = []
        self._warned_size = False
        self._last_masks: Dict[str, np.ndarray] = {}

        # camera pose (marker in camera frame), frozen once stable
        self.cal_frozen: Optional[dict] = None
        self.cal_buf: List[tuple] = []
        self.cal_state = 'none'
        self.cal_drift = None
        self.filters: Dict[str, g3.MedianFilter] = {}
        if self.get_parameter('marker.load_saved').value:
            self._load_saved_calibration()

        img_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=5,
                             durability=DurabilityPolicy.VOLATILE)
        info_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST, depth=1)
        gp = self.get_parameter
        self.create_subscription(CameraInfo, gp('camera_info_topic').value,
                                 self._on_camera_info, info_qos)
        colour_sub = Subscriber(self, Image, gp('color_topic').value, qos_profile=img_qos)
        depth_sub = Subscriber(self, Image, gp('depth_topic').value, qos_profile=img_qos)
        self.sync = ApproximateTimeSynchronizer(
            [colour_sub, depth_sub], queue_size=int(gp('sync_queue_size').value),
            slop=float(gp('sync_slop_s').value))
        self.sync.registerCallback(self._on_frames)

        self.pub_annotated = self.create_publisher(Image, '~/annotated_image', 2)
        self.pub_mask = self.create_publisher(Image, '~/debug_mask', 2)
        self.pub_markers = self.create_publisher(MarkerArray, '~/markers', 2)
        self.pub_scene = self.create_publisher(String, '~/scene', 10)
        self.pub_status = self.create_publisher(String, '~/status', 10)
        self.pub_points: Dict[str, object] = {}
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._cam_tf_warned = False
        self.create_service(Trigger, '~/recalibrate', self._on_recalibrate)

        self.get_logger().info(
            f"Waiting for {gp('color_topic').value}, {gp('depth_topic').value} "
            f"and {gp('camera_info_topic').value} ...")

    # ------------------------------------------------------------------ params
    def _declare_parameters(self):
        d = self.declare_parameter
        d('color_topic', '/camera/camera/color/image_raw')
        d('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        d('camera_info_topic', '/camera/camera/color/camera_info')
        d('sync_slop_s', 0.05)
        d('sync_queue_size', 10)
        d('print_period_s', 1.0)
        d('publish_debug_mask', False)
        d('smoothing_frames', 7)

        d('depth_scale', 0.001)
        d('depth_min_m', 0.15)
        d('depth_max_m', 2.0)
        d('depth_erode_px', 3)
        d('depth_min_valid_px', 30)
        d('depth_min_valid_fraction', 0.2)

        d('blur_kernel', 5)
        d('morph_kernel', 5)
        d('min_area_px', 300.0)
        d('max_area_px', 60000.0)
        d('min_fill_ratio', 0.6)
        d('max_per_colour', 1)
        d('overlap_iou', 0.5)
        d('shape_epsilon', 0.03)

        d('colour_names', list(DEFAULT_COLOURS.keys()))
        for name, (lo, hi, bgr, shape) in DEFAULT_COLOURS.items():
            d(f'colours.{name}.hsv_lower', lo)
            d(f'colours.{name}.hsv_upper', hi)
            d(f'colours.{name}.draw_bgr', bgr)
            d(f'colours.{name}.expected_shape', shape)

        # Peg top-face extraction
        d('peg_top.erode_px', 2)
        d('peg_top.top_percentile', 95.0)
        d('peg_top.top_band_m', 0.006)
        d('peg_top.min_top_px', 25)

        # Sockets: red outlines on the shape board. Red wraps around H=0 -> two ranges.
        d('sockets.enabled', True)
        d('sockets.hsv_lower1', [0, 90, 60])
        d('sockets.hsv_upper1', [10, 255, 255])
        d('sockets.hsv_lower2', [170, 90, 60])
        d('sockets.hsv_upper2', [179, 255, 255])
        d('sockets.morph_kernel', 3)
        d('sockets.min_hole_area_px', 400.0)
        d('sockets.max_hole_area_px', 40000.0)
        d('sockets.surface_dilate_px', 6)
        d('sockets.group_px', 6)
        d('sockets.min_coverage', 0.75)
        d('sockets.max_interior_red', 0.10)
        d('sockets.circle_fallback', True)
        d('sockets.circle_min_radius_px', 15)
        d('sockets.circle_max_radius_px', 120)
        d('sockets.circle_min_coverage', 0.45)
        d('sockets.max_sockets', 4)

        d('marker.enabled', True)
        d('marker.dictionary', 'DICT_4X4_50')
        d('marker.id', 0)
        d('marker.length_m', 0.05)
        d('marker.position_in_base_m', [0.0, 0.0, 0.0])
        d('marker.yaw_in_base_deg', 0.0)
        d('marker.frame_id', 'robot_base_marker')
        d('marker.publish_tf', True)
        d('marker.refine_with_depth', True)
        # Known peg sizes (side length; cylinder: diameter) for the known-shape top-face fit, m
        # A peg must stand on the table and be within reach (filters pens, stickers, the arm...)
        d('peg_filter.z_top_min_m', 0.03)
        d('peg_filter.z_top_max_m', 0.20)
        d('peg_filter.max_radius_m', 0.32)
        d('peg_size_m.square', 0.040)
        d('peg_size_m.hexagon', 0.020)
        d('peg_size_m.triangle', 0.030)
        d('marker.refine_mode', 'table')       # table = whole-table plane, local = patch round marker
        d('marker.max_reproj_px', 2.0)
        d('marker.freeze', True)
        d('marker.freeze_frames', 30)
        d('marker.drift_warn_mm', 8.0)
        d('marker.save_file', '~/omx_calibration/camera_pose.yaml')
        d('marker.load_saved', False)

        # Drawn axes on the foam -> ROBOTIS link1 (found with `ros2 run peg_detection frame_calibration`)
        d('robot_frame.frame_id', 'link1')
        d('robot_frame.yaw_deg', 0.0)
        d('robot_frame.offset_m', [0.0, 0.0, 0.0])
        # Final correction fitted from measured ground-truth points (ros2 run omx_calibration fit_camera):
        #   p = R * p + t, as [r11 r12 r13 r21 r22 r23 r31 r32 r33 tx ty tz]. Identity = no correction.
        d('robot_frame.correction', [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        d('robot_frame.publish_camera_tf', True)
        d('robot_frame.camera_root_frame', 'camera_link')

    def _on_param_change(self, _params):
        self._reload = True
        return SetParametersResult(successful=True)

    def _load_config(self):
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.det_p = DetectionParams(int(g('blur_kernel')), int(g('morph_kernel')),
                                     float(g('min_area_px')), float(g('max_area_px')),
                                     float(g('min_fill_ratio')), int(g('max_per_colour')),
                                     float(g('overlap_iou')))
        self.depth_p = DepthParams(float(g('depth_min_m')), float(g('depth_max_m')),
                                   int(g('depth_erode_px')), int(g('depth_min_valid_px')),
                                   float(g('depth_min_valid_fraction')))
        self.top_p = g3.PegTopParams(int(g('peg_top.erode_px')), float(g('depth_min_m')),
                                     float(g('depth_max_m')), float(g('peg_top.top_percentile')),
                                     float(g('peg_top.top_band_m')), int(g('peg_top.min_top_px')))
        self.specs: List[ColourSpec] = []
        for name in g('colour_names'):
            if not self.has_parameter(f'colours.{name}.hsv_lower'):
                self.declare_parameter(f'colours.{name}.hsv_lower', [0, 0, 0])
                self.declare_parameter(f'colours.{name}.hsv_upper', [0, 0, 0])
                self.declare_parameter(f'colours.{name}.draw_bgr', [255, 255, 255])
                self.declare_parameter(f'colours.{name}.expected_shape', '')
            lo = np.array(g(f'colours.{name}.hsv_lower'), dtype=np.uint8)
            hi = np.array(g(f'colours.{name}.hsv_upper'), dtype=np.uint8)
            self.specs.append(ColourSpec(name, lo, hi,
                                         tuple(int(x) for x in g(f'colours.{name}.draw_bgr')),
                                         str(g(f'colours.{name}.expected_shape'))))

    # --------------------------------------------------------------- callbacks
    def _on_camera_info(self, msg: CameraInfo):
        first = self.intr is None
        self.intr = Intrinsics.from_camera_info(msg)
        self.dist = np.array(msg.d, dtype=np.float64) if len(msg.d) else np.zeros(5)
        if first:
            i = self.intr
            self.get_logger().info(
                f'Intrinsics from camera_info ({i.frame_id}, {i.width}x{i.height}): '
                f'fx={i.fx:.2f} fy={i.fy:.2f} cx={i.cx:.2f} cy={i.cy:.2f}')

    def _on_recalibrate(self, _req, res):
        self.cal_frozen, self.cal_buf, self.cal_drift = None, [], None
        self.filters.clear()
        res.success = True
        res.message = (f"Camera pose cleared; re-measuring over "
                       f"{self.get_parameter('marker.freeze_frames').value} frames. "
                       'Keep the marker visible and the arm away from it.')
        self.get_logger().info(res.message)
        return res

    def _on_frames(self, colour_msg: Image, depth_msg: Image):
        if self.intr is None:
            self.get_logger().warn('No camera_info yet; skipping frame.', throttle_duration_sec=2.0)
            return
        if self._reload:
            self._load_config()
            self._reload = False
            self.get_logger().info('Parameters reloaded.')

        if not (colour_msg.width == depth_msg.width == self.intr.width and
                colour_msg.height == depth_msg.height == self.intr.height):
            if not self._warned_size:
                self.get_logger().error(
                    f'Size mismatch: colour {colour_msg.width}x{colour_msg.height}, depth '
                    f'{depth_msg.width}x{depth_msg.height}, camera_info {self.intr.width}x'
                    f'{self.intr.height}. Is align_depth.enable:=true?')
                self._warned_size = True
            return
        if depth_msg.header.frame_id != colour_msg.header.frame_id:
            self.get_logger().warn(
                f'Depth frame {depth_msg.header.frame_id} != colour frame '
                f'{colour_msg.header.frame_id}: depth is NOT aligned to colour.',
                throttle_duration_sec=5.0)

        now = time.monotonic()
        self._frame_times = [t for t in self._frame_times if now - t < 2.0] + [now]
        stamp_dt_ms = abs(_stamp_s(colour_msg) - _stamp_s(depth_msg)) * 1000.0
        bgr = colour_msg_to_bgr(colour_msg)
        depth = depth_msg_to_metres(depth_msg, float(self.get_parameter('depth_scale').value))

        marker = self._detect_base_marker(bgr, depth) if \
            self.get_parameter('marker.enabled').value else None
        T = self._camera_to_robot(marker, depth)
        objects = self._detect_objects(bgr, depth, T)
        sockets = self._detect_sockets(bgr, depth, T) if \
            self.get_parameter('sockets.enabled').value else []

        header = colour_msg.header
        if marker is not None and marker.get('pose_ok') and \
                self.get_parameter('marker.publish_tf').value:
            self._publish_marker_tf(header, marker)
        if T is not None:
            self._publish_robot_outputs(header, objects, sockets)
            if self.get_parameter('robot_frame.publish_camera_tf').value:
                self._publish_camera_tf(header, T)

        self._publish_status(marker, sockets)
        self.pub_annotated.publish(bgr_to_image_msg(self._annotate(bgr, objects, marker, sockets, T),
                                                    header))
        if self.get_parameter('publish_debug_mask').value:
            self.pub_mask.publish(bgr_to_image_msg(self._debug_mask(bgr), header))
        self._publish_rviz_markers(header, objects, sockets, T)

        if now - self._last_print >= float(self.get_parameter('print_period_s').value):
            self._last_print = now
            rate = (len(self._frame_times) - 1) / max(now - self._frame_times[0], 1e-6) \
                if len(self._frame_times) > 1 else 0.0
            self._print_report(objects, marker, sockets, rate, stamp_dt_ms)

    # ------------------------------------------------------------ calibration
    def _camera_to_robot(self, marker, depth) -> Optional[g3.RigidTransform]:
        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        live = None
        if marker is not None and marker.get('pose_ok'):
            if marker['reproj_px'] > float(gp('marker.max_reproj_px')):
                marker['rejected'] = f"reprojection {marker['reproj_px']:.2f} px too high"
            else:
                R, t = marker['R'], marker['t']
                if gp('marker.refine_with_depth'):
                    if str(gp('marker.refine_mode')) == 'table':
                        ref = g3.refine_marker_with_table(R, t, marker['corners'], depth, self.intr,
                                                          min_m=self.depth_p.min_m,
                                                          max_m=self.depth_p.max_m)
                    else:
                        ref = g3.refine_marker_with_depth(R, t, marker['corners'], depth, self.intr,
                                                          min_m=self.depth_p.min_m,
                                                          max_m=self.depth_p.max_m)
                    marker['refine'] = ref
                    if ref.ok:
                        R, t = ref.R, ref.t
                marker['R_used'], marker['t_used'] = R, t
                live = (R, t)

        use = None
        if gp('marker.freeze'):
            if self.cal_frozen is None:
                if live is not None:
                    self.cal_buf.append(live)
                    if len(self.cal_buf) >= int(gp('marker.freeze_frames')):
                        R, t, n = g3.average_poses([r for r, _ in self.cal_buf],
                                                   [t for _, t in self.cal_buf])
                        spread = float(np.std([t for _, t in self.cal_buf], axis=0).max() * 1000)
                        self.cal_frozen = {'R': R, 't': t, 'n': n, 'spread_mm': spread,
                                           'source': 'measured'}
                        self.cal_buf = []
                        self._save_calibration()
                        self.get_logger().info(
                            f'Camera pose FROZEN from {n} frames (translation jitter {spread:.1f} mm).'
                            ' The arm may now cover the marker. Call ~/recalibrate if the camera moves.')
                use = live
                self.cal_state = 'live' if live is not None else 'none'
            if self.cal_frozen is not None:
                use = (self.cal_frozen['R'], self.cal_frozen['t'])
                self.cal_state = 'frozen'
                if live is not None:
                    dt = float(np.linalg.norm(live[1] - self.cal_frozen['t']) * 1000)
                    dr = g3.rotation_angle_deg(live[0], self.cal_frozen['R'])
                    self.cal_drift = (dt, dr)
                    if dt > float(gp('marker.drift_warn_mm')) or dr > 1.5:
                        self.get_logger().warn(
                            f'Live marker differs from frozen pose by {dt:.1f} mm / {dr:.1f} deg: '
                            'camera or marker moved? -> ros2 service call /peg_detector/recalibrate '
                            'std_srvs/srv/Trigger', throttle_duration_sec=5.0)
                else:
                    self.cal_drift = None
        else:
            use = live
            self.cal_state = 'live' if live is not None else 'none'
        if use is None:
            return None
        T = g3.camera_to_robot(
            use[0], use[1], np.deg2rad(float(gp('marker.yaw_in_base_deg'))),
            gp('marker.position_in_base_m'), np.deg2rad(float(gp('robot_frame.yaw_deg'))),
            gp('robot_frame.offset_m'))
        c = np.array(gp('robot_frame.correction'), float)
        if c.size == 12:
            Rc, tc = c[:9].reshape(3, 3), c[9:]
            T = g3.RigidTransform(Rc @ T.A, Rc @ T.b + tc)
        return T

    def _save_calibration(self):
        path = os.path.expanduser(str(self.get_parameter('marker.save_file').value))
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                yaml.safe_dump({'R_cam_marker': self.cal_frozen['R'].tolist(),
                                't_cam_marker': self.cal_frozen['t'].tolist(),
                                'frames': self.cal_frozen['n'],
                                'saved': time.strftime('%Y-%m-%d %H:%M:%S')}, f)
        except OSError as e:
            self.get_logger().warn(f'Could not save camera pose to {path}: {e}')

    def _load_saved_calibration(self):
        path = os.path.expanduser(str(self.get_parameter('marker.save_file').value))
        try:
            with open(path) as f:
                c = yaml.safe_load(f)
            self.cal_frozen = {'R': np.array(c['R_cam_marker']), 't': np.array(c['t_cam_marker']),
                               'n': c.get('frames', 0), 'spread_mm': float('nan'),
                               'source': f"loaded ({c.get('saved', '?')})"}
            self.get_logger().info(f'Loaded saved camera pose from {path} ({c.get("saved", "?")}).')
        except (OSError, KeyError, TypeError) as e:
            self.get_logger().warn(f'marker.load_saved: could not load {path}: {e}')

    # ---------------------------------------------------------------- pipeline
    def _smooth(self, key: str, p: np.ndarray) -> np.ndarray:
        n = int(self.get_parameter('smoothing_frames').value)
        if n <= 1:
            return p
        f = self.filters.setdefault(key, g3.MedianFilter(n=n))
        f.n = n
        return f.update(p)

    def _detect_objects(self, bgr, depth, T) -> List[Dict]:
        blobs_by_colour, self._last_masks = detect_colours(bgr, self.specs, self.det_p)
        eps = float(self.get_parameter('shape_epsilon').value)
        objects = []
        for spec in self.specs:
            blobs = blobs_by_colour.get(spec.name, [])
            if not blobs:
                objects.append({'colour': spec.name, 'spec': spec, 'detected': False})
                continue
            for b in blobs:
                shape, conf, feats = classify_shape(b.contour, eps)
                obj = {'colour': spec.name, 'spec': spec, 'detected': True, 'blob': b,
                       'shape_geom': shape, 'shape_conf': conf, 'shape_feats': feats,
                       'xyz': None, 'xyz_base': None, 'top': None, 'old_base': None,
                       'depth': robust_depth(depth, b.mask, self.depth_p)}
                if T is not None:
                    top = g3.peg_top_center(depth, b.mask, self.intr, T, self.top_p)
                    obj['top'] = top
                    obj['old_base'] = g3.silhouette_point(depth, b.mask, b.centroid, self.intr, T)
                    if top.valid:
                        centre = np.array(top.center, float)
                        shp = spec.expected_shape
                        if shp == 'circle' and top.top_outline_xy is not None:
                            cxy = g3.circle_center_xy(top.top_outline_xy)   # round peg: fit a circle
                            if cxy is not None:
                                centre[:2] = cxy
                        elif shp in g3.SYMMETRY and top.top_outline_xy is not None:
                            fit = g3.polygon_center_xy(top.top_outline_xy, g3.SYMMETRY[shp],
                                                       float(self.get_parameter(f'peg_size_m.{shp}').value))
                            if fit is not None:                          # known-size polygon fit
                                centre[:2] = fit[0]
                                obj['fit_yaw'] = fit[1]
                        obj['xyz_base'] = self._smooth('peg:' + spec.name, centre)
                        obj['xyz'] = T.inverse_apply(obj['xyz_base'])
                elif obj['depth'].valid:      # no calibration yet: camera frame, old method
                    obj['xyz'] = deproject(b.centroid[0], b.centroid[1], obj['depth'].z_m,
                                           self.intr)
                objects.append(obj)
        return self._one_per_colour(objects)

    def _one_per_colour(self, objects: List[Dict]) -> List[Dict]:
        """Several blobs of one colour (a pen, a sticker, the peg's own reflection...): keep the
        one that looks like a real peg - standing on the table at a plausible height and within
        reach - and the biggest of those."""
        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        zmin, zmax, rmax = float(gp('peg_filter.z_top_min_m')), float(gp('peg_filter.z_top_max_m')), \
            float(gp('peg_filter.max_radius_m'))
        best = {}
        for obj in objects:
            c = obj['colour']
            score = -1.0
            if obj.get('detected'):
                score = 0.0
                p = obj.get('xyz_base')
                if p is not None:
                    plausible = zmin <= p[2] <= zmax and float(np.hypot(p[0], p[1])) <= rmax
                    obj['plausible'] = plausible
                    score = 1.0 if plausible else 0.5
                score += obj['blob'].area_px * 1e-7       # tie-break: the bigger blob
            if c not in best or score > best[c][0]:
                best[c] = (score, obj)
        out = []
        for obj in objects:
            keep = best[obj['colour']][1] is obj
            if keep and obj.get('detected') and obj.get('plausible') is False:
                obj['xyz_base'] = None                    # implausible -> don't publish it
                obj['rejected'] = 'implausible height/position'
            if keep:
                out.append(obj)
        return out

    def _detect_sockets(self, bgr, depth, T) -> List[Dict]:
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        u8 = lambda n: np.array(g(n), dtype=np.uint8)  # noqa: E731
        sp = SocketParams(
            hsv_ranges=[(u8('sockets.hsv_lower1'), u8('sockets.hsv_upper1')),
                        (u8('sockets.hsv_lower2'), u8('sockets.hsv_upper2'))],
            morph_kernel=int(g('sockets.morph_kernel')),
            min_hole_area_px=float(g('sockets.min_hole_area_px')),
            max_hole_area_px=float(g('sockets.max_hole_area_px')),
            surface_dilate_px=int(g('sockets.surface_dilate_px')),
            group_px=int(g('sockets.group_px')),
            min_coverage=float(g('sockets.min_coverage')),
            max_interior_red=float(g('sockets.max_interior_red')),
            circle_fallback=bool(g('sockets.circle_fallback')),
            circle_min_radius_px=int(g('sockets.circle_min_radius_px')),
            circle_max_radius_px=int(g('sockets.circle_max_radius_px')),
            circle_min_coverage=float(g('sockets.circle_min_coverage')),
            max_sockets=int(g('sockets.max_sockets')))
        found, self._last_socket_mask = detect_sockets(bgr, sp)
        dp = DepthParams(self.depth_p.min_m, self.depth_p.max_m, 0,
                         self.depth_p.min_valid_px, self.depth_p.min_valid_fraction)
        shape_to_colour = {s.expected_shape: s.name for s in self.specs if s.expected_shape}
        out = []
        for s in found:
            sk = {'socket': s, 'shape': s.shape, 'confidence': s.confidence,
                  'fits': shape_to_colour.get(s.shape, ''), 'xyz': None, 'xyz_base': None,
                  'plane': None, 'uv': s.centroid,
                  'depth': robust_depth(depth, s.surface_mask, dp)}
            if T is not None:
                sc = g3.socket_center_on_plane(depth, s.surface_mask, g3.dense_outline(s.hole_contour),
                                               self.intr, T, self.depth_p.min_m, self.depth_p.max_m)
                sk['plane'] = sc
                if sc.valid:
                    sk['xyz_base'] = self._smooth('sock:' + s.shape, sc.center)
                    sk['xyz'] = T.inverse_apply(sk['xyz_base'])
                    sk['uv'] = g3.project(sk['xyz'], self.intr)
            elif sk['depth'].valid:
                sk['xyz'] = deproject(s.centroid[0], s.centroid[1], sk['depth'].z_m, self.intr)
            out.append(sk)
        return out

    def _detect_base_marker(self, bgr, depth) -> Optional[Dict]:
        gp = self.get_parameter
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        wanted = int(gp('marker.id').value)
        found = [c for i, c in detect_markers(gray, gp('marker.dictionary').value) if i == wanted]
        if not found:
            return {'detected': False}
        corners = found[0]
        u, v = corners.mean(axis=0)
        result = {'detected': True, 'corners': corners, 'uv': (float(u), float(v)),
                  'pose_ok': False}
        pose = estimate_pose(corners, float(gp('marker.length_m').value),
                             self.intr.as_matrix(), self.dist)
        if pose is not None:
            R, t, rvec, err = pose
            result.update(pose_ok=True, R=R, t=t, rvec=rvec, reproj_px=err)
        result['z_depth'] = depth_at_patch(depth, u, v, 4, self.depth_p)
        return result

    # ----------------------------------------------------------------- outputs
    def _point_pub(self, topic):
        if topic not in self.pub_points:
            self.pub_points[topic] = self.create_publisher(PointStamped, topic, 10)
        return self.pub_points[topic]

    def _publish_robot_outputs(self, header, objects, sockets):
        frame = self.get_parameter('robot_frame.frame_id').value
        r4 = lambda x: round(float(x), 4)  # noqa: E731
        scene = {'stamp': header.stamp.sec + header.stamp.nanosec * 1e-9, 'frame_id': frame, 'calibration': self.cal_state,
                 'units': 'm', 'pegs': {}, 'sockets': {}}
        for obj in objects:
            if obj.get('xyz_base') is None:
                continue
            x, y, z = obj['xyz_base']
            scene['pegs'][obj['colour']] = {
                'x': r4(x), 'y': r4(y), 'z_top': r4(z),
                'target_socket': obj['spec'].expected_shape, 'top_px': obj['top'].n_top}
            n_sym = g3.SYMMETRY.get(obj['spec'].expected_shape)
            if n_sym and obj['top'].top_outline_xy is not None:
                yaw, strength = g3.polygon_yaw_deg(obj['top'].top_outline_xy, n_sym)
                if obj.get('fit_yaw') is not None:
                    yaw = obj['fit_yaw']
                scene['pegs'][obj['colour']].update(yaw_deg=round(yaw, 1), yaw_sym_deg=360 // n_sym,
                                                    yaw_strength=round(strength, 2))
            self._publish_point(f"~/peg/{obj['colour']}", header, frame, obj['xyz_base'])
        for sk in sockets:
            if sk.get('xyz_base') is None:
                continue
            x, y, z = sk['xyz_base']
            scene['sockets'][sk['shape']] = {'x': r4(x), 'y': r4(y), 'z': r4(z),
                                             'for_peg': sk['fits'], 'confidence': sk['confidence']}
            n_sym = g3.SYMMETRY.get(sk['shape'])
            if n_sym and sk['plane'] is not None and sk['plane'].outline_xy is not None:
                yaw, strength = g3.polygon_yaw_deg(sk['plane'].outline_xy, n_sym)
                scene['sockets'][sk['shape']].update(yaw_deg=round(yaw, 1), yaw_sym_deg=360 // n_sym,
                                                     yaw_strength=round(strength, 2))
            self._publish_point(f"~/socket/{sk['shape']}", header, frame, sk['xyz_base'])
        self.pub_scene.publish(String(data=json.dumps(scene)))

    def _publish_status(self, marker, sockets):
        """Calibration diagnostics for the omx_calibration wizard (JSON)."""
        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        yaw = float(gp('robot_frame.yaw_deg'))
        off = np.array(gp('robot_frame.offset_m'), float)
        Rinv = g3.rot_z(-np.deg2rad(yaw))
        st = {'state': self.cal_state, 'marker_length_m': float(gp('marker.length_m')),
              'robot_frame': {'yaw_deg': yaw, 'offset_m': off.tolist()},
              'frozen': None, 'drift_mm': None, 'drift_deg': None,
              'marker': None, 'socket_z_drawn': {}}
        if self.cal_frozen is not None:
            c = self.cal_frozen
            st['frozen'] = {'n': int(c['n']), 'jitter_mm': float(c['spread_mm']),
                            'source': c['source']}
        if self.cal_drift is not None:
            st['drift_mm'], st['drift_deg'] = float(self.cal_drift[0]), float(self.cal_drift[1])
        if marker is not None and marker.get('detected') and marker.get('pose_ok'):
            m = {'visible': True, 'reproj_px': float(marker['reproj_px'])}
            ref = marker.get('refine')
            if ref is not None and ref.ok:
                m.update(tilt_correction_deg=float(ref.normal_change_deg),
                         side_mm=float(ref.side_m * 1000), plane_rms_mm=float(ref.plane_rms_mm))
            st['marker'] = m
        for sk in sockets:
            if sk.get('xyz_base') is not None:
                st['socket_z_drawn'][sk['shape']] = float((Rinv @ (sk['xyz_base'] - off))[2])
        self.pub_status.publish(String(data=json.dumps(st)))

    def _publish_point(self, topic, header, frame, p):
        msg = PointStamped()
        msg.header.stamp = header.stamp
        msg.header.frame_id = frame
        msg.point.x, msg.point.y, msg.point.z = (float(v) for v in p)
        self._point_pub(topic).publish(msg)

    def _publish_camera_tf(self, header, T: g3.RigidTransform):
        """robot frame -> camera_link, so RViz can show the camera's cloud on the arm model."""
        root = self.get_parameter('robot_frame.camera_root_frame').value
        try:
            tr = self.tf_buffer.lookup_transform(self.intr.frame_id, root, Time())
        except Exception as e:  # noqa: BLE001 (tf2 raises several types)
            if not self._cam_tf_warned:
                self.get_logger().info(f'No TF {root} -> {self.intr.frame_id} yet ({e}); '
                                       'camera TF to the robot will appear once the driver publishes it.')
                self._cam_tf_warned = True
            return
        q = tr.transform.rotation
        R1 = _quat_to_rot(q.x, q.y, q.z, q.w)
        t1 = np.array([tr.transform.translation.x, tr.transform.translation.y,
                       tr.transform.translation.z])
        R = T.A @ R1
        t = T.A @ t1 + T.b
        msg = TransformStamped()
        msg.header.stamp = header.stamp
        msg.header.frame_id = self.get_parameter('robot_frame.frame_id').value
        msg.child_frame_id = root
        msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = \
            (float(v) for v in t)
        qx, qy, qz, qw = rotation_to_quaternion(R)
        msg.transform.rotation.x, msg.transform.rotation.y = float(qx), float(qy)
        msg.transform.rotation.z, msg.transform.rotation.w = float(qz), float(qw)
        self.tf_broadcaster.sendTransform(msg)

    def _publish_marker_tf(self, header, marker):
        R = marker.get('R_used', marker['R'])
        t = marker.get('t_used', marker['t'])
        tf = TransformStamped()
        tf.header.stamp = header.stamp
        tf.header.frame_id = self.intr.frame_id or header.frame_id
        tf.child_frame_id = self.get_parameter('marker.frame_id').value
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = \
            (float(x) for x in t)
        qx, qy, qz, qw = rotation_to_quaternion(R)
        tf.transform.rotation.x, tf.transform.rotation.y = float(qx), float(qy)
        tf.transform.rotation.z, tf.transform.rotation.w = float(qz), float(qw)
        self.tf_broadcaster.sendTransform(tf)

    def _annotate(self, bgr, objects, marker, sockets, T):
        img = bgr.copy()
        put = _put_text
        for sk in sockets:
            s = sk['socket']
            u, v = (int(round(c)) for c in sk['uv'])
            cv2.drawContours(img, [s.hole_contour], -1, (255, 255, 255), 2)
            cv2.drawMarker(img, (u, v), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
            if sk.get('xyz_base') is not None:
                bx, by, _ = (100.0 * c for c in sk['xyz_base'])
                text = f"{s.shape} X{bx:+.1f} Y{by:+.1f}"
            else:
                text = s.shape
            put(img, text, (max(u - 45, 0), v + 22), 0.42)
        for obj in objects:
            spec = obj['spec']
            if not obj['detected']:
                continue
            b = obj['blob']
            cv2.drawContours(img, [b.contour], -1, spec.draw_bgr, 2)
            su, sv = (int(round(c)) for c in b.centroid)
            cv2.drawMarker(img, (su, sv), (150, 150, 150), cv2.MARKER_CROSS, 8, 1)  # old centre
            top = obj.get('top')
            if top is not None and top.valid and T is not None:
                ring = np.column_stack([top.top_outline_xy,
                                        np.full(len(top.top_outline_xy), top.center[2])])
                pc = T.inverse_apply(ring)
                pts = np.array([g3.project(p, self.intr) for p in pc], np.int32)
                cv2.polylines(img, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)
                u, v = (int(round(c)) for c in g3.project(obj['xyz'], self.intr))
                cv2.circle(img, (u, v), 5, (255, 255, 255), -1)
                cv2.circle(img, (u, v), 5, (0, 0, 0), 1)
            if obj.get('xyz_base') is not None:
                bx, by, bz = (100.0 * c for c in obj['xyz_base'])
                label = f"{spec.name} X{bx:+.1f} Y{by:+.1f} Ztop{bz:+.1f}"
            elif obj['depth'].valid:
                label = f"{spec.name} (no calibration yet)"
            else:
                label = f"{spec.name} depth invalid"
            put(img, label, (b.bbox[0], max(b.bbox[1] - 6, 12)), 0.45)
        if marker and marker.get('detected'):
            pts = marker['corners'].astype(int)
            cv2.polylines(img, [pts], True, (255, 0, 255), 2)
            if marker.get('pose_ok'):
                R = marker.get('R_used', marker['R'])
                t = marker.get('t_used', marker['t'])
                rvec, _ = cv2.Rodrigues(R)
                cv2.drawFrameAxes(img, self.intr.as_matrix(), self.dist, rvec, t,
                                  float(self.get_parameter('marker.length_m').value))
            put(img, 'marker', (pts[0][0], pts[0][1] - 8), 0.5, (255, 0, 255))
        frame = self.get_parameter('robot_frame.frame_id').value
        state = {'frozen': ('camera pose FROZEN', (0, 200, 0)),
                 'live': (f'measuring camera pose {len(self.cal_buf)}/'
                          f"{self.get_parameter('marker.freeze_frames').value}", (0, 200, 255)),
                 'none': ('NO camera pose: marker not visible', (0, 0, 255))}[self.cal_state]
        put(img, f'{state[0]} | frame {frame} | dot = peg top centre, grey + = old centre',
            (8, img.shape[0] - 10), 0.42, state[1])
        return img

    def _debug_mask(self, bgr):
        out = np.zeros_like(bgr)
        for spec in self.specs:
            m = self._last_masks.get(spec.name)
            if m is not None:
                out[m > 0] = spec.draw_bgr
        sm = getattr(self, '_last_socket_mask', None)
        if sm is not None:
            out[sm > 0] = (255, 255, 255)
        return out

    def _publish_rviz_markers(self, header, objects, sockets, T):
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        robot = T is not None
        frame = self.get_parameter('robot_frame.frame_id').value if robot else \
            (self.intr.frame_id or header.frame_id)
        key = 'xyz_base' if robot else 'xyz'
        mid = 0
        for obj in objects:
            if not obj.get('detected') or obj.get(key) is None:
                continue
            b, g, r = obj['spec'].draw_bgr
            for kind in ('sphere', 'text'):
                m = Marker()
                m.header.frame_id, m.header.stamp = frame, header.stamp
                m.ns, m.id = f'pegs_{kind}', mid
                m.pose.orientation.w = 1.0
                m.pose.position.x, m.pose.position.y, m.pose.position.z = \
                    (float(x) for x in obj[key])
                m.color.r, m.color.g, m.color.b, m.color.a = r / 255, g / 255, b / 255, 1.0
                if kind == 'sphere':
                    m.type = Marker.SPHERE
                    m.scale.x = m.scale.y = m.scale.z = 0.015
                else:
                    m.type = Marker.TEXT_VIEW_FACING
                    m.text = obj['colour']
                    if robot:
                        m.pose.position.z += 0.03
                    else:
                        m.pose.position.y -= 0.04
                    m.scale.z = 0.02
                arr.markers.append(m)
            mid += 1
        for k, sk in enumerate(sockets):
            if sk.get(key) is None:
                continue
            m = Marker()
            m.header.frame_id, m.header.stamp = frame, header.stamp
            m.ns, m.id, m.type = 'sockets', k, Marker.CYLINDER
            m.pose.orientation.w = 1.0
            m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(x) for x in sk[key])
            m.scale.x = m.scale.y = 0.03
            m.scale.z = 0.004
            m.color.r = m.color.g = m.color.b = 1.0
            m.color.a = 0.8
            arr.markers.append(m)
        self.pub_markers.publish(arr)

    def _print_report(self, objects, marker, sockets, rate_hz, stamp_dt_ms):
        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        frame = gp('robot_frame.frame_id')
        L = ['', '================ PEG DETECTOR ================',
             f'rate {rate_hz:.1f} Hz | colour/depth stamp diff {stamp_dt_ms:.1f} ms | '
             f'camera pose: {self.cal_state.upper()}']
        if self.cal_frozen is not None:
            c = self.cal_frozen
            drift = '' if self.cal_drift is None else \
                f' | live marker vs frozen: {self.cal_drift[0]:.1f} mm, {self.cal_drift[1]:.2f} deg'
            L.append(f"  frozen from {c['n']} frames ({c['source']}, jitter {c['spread_mm']:.1f} mm){drift}")
        L.append(f"  robot frame '{frame}': yaw {gp('robot_frame.yaw_deg'):+.2f} deg, "
                 f"offset {np.round(np.array(gp('robot_frame.offset_m')) * 1000, 1).tolist()} mm")
        L.append('')
        L.append(f'PEGS - top-face centre, {frame} frame, cm')
        for obj in objects:
            name = obj['colour'].capitalize()
            if not obj['detected']:
                L.append(f'  {name:<7} not detected')
                continue
            top = obj.get('top')
            if obj.get('xyz_base') is not None:
                bx, by, bz = (100.0 * c for c in obj['xyz_base'])
                shift = ''
                if obj.get('old_base') is not None:
                    shift = f'  (old method was {np.linalg.norm(obj["old_base"][:2] - obj["xyz_base"][:2]) * 1000:.0f} mm off)'
                L.append(f'  {name:<7} X = {bx:+6.1f}   Y = {by:+6.1f}   Z_top = {bz:+6.1f}   '
                         f'-> {obj["spec"].expected_shape:<8} top px {top.n_top}, '
                         f'flat {top.top_spread_mm:.0f} mm{shift}')
            elif top is not None and not top.valid:
                L.append(f'  {name:<7} top face not found: {top.reason}')
            elif obj.get('xyz') is not None:
                x, y, z = obj['xyz']
                L.append(f'  {name:<7} no camera pose yet -> camera frame X={x:+.3f} Y={y:+.3f} Z={z:+.3f} m')
            else:
                L.append(f"  {name:<7} invalid depth ({obj['depth'].reason})")
        if gp('sockets.enabled'):
            L.append('')
            L.append(f'SOCKETS - centre on the board surface, {frame} frame, cm ({len(sockets)} found)')
            if not sockets:
                L.append('  none detected (check debug_mask: outlines should be white rings)')
            seen = {}
            for sk in sockets:
                seen[sk['shape']] = seen.get(sk['shape'], 0) + 1
                fits = f"<- {sk['fits']}" if sk['fits'] else '<- no peg colour'
                if sk.get('xyz_base') is not None:
                    bx, by, bz = (100.0 * c for c in sk['xyz_base'])
                    pl = sk['plane']
                    L.append(f"  {sk['shape']:<9} X = {bx:+6.1f}   Y = {by:+6.1f}   Z = {bz:+6.1f}   "
                             f"{fits:<10} ({sk['confidence']}, board tilt {pl.tilt_deg:.1f} deg)")
                elif sk.get('plane') is not None and not sk['plane'].valid:
                    L.append(f"  {sk['shape']:<9} board plane failed: {sk['plane'].reason}")
                elif sk.get('xyz') is not None:
                    L.append(f"  {sk['shape']:<9} no camera pose yet (camera frame only)")
                else:
                    L.append(f"  {sk['shape']:<9} INVALID DEPTH ({sk['depth'].reason})")
            dupes = [s for s, c in seen.items() if c > 1]
            if dupes:
                L.append(f'  WARNING: more than one socket classified as {dupes}')
        L += ['', '---------- MARKER ----------']
        if not gp('marker.enabled'):
            L.append('disabled')
        elif marker is None or not marker.get('detected'):
            L.append(f"not visible ({gp('marker.dictionary')} id {gp('marker.id')})"
                     + (' - fine, using the frozen pose' if self.cal_frozen else ''))
        elif not marker.get('pose_ok'):
            L.append('pose estimation failed')
        else:
            L.append(f"reprojection {marker['reproj_px']:.2f} px"
                     + (f" - REJECTED: {marker['rejected']}" if marker.get('rejected') else ''))
            ref = marker.get('refine')
            if ref is not None:
                if ref.ok:
                    L.append(f'depth refinement: plane {ref.n_px} px, rms {ref.plane_rms_mm:.1f} mm, '
                             f'tilt correction {ref.normal_change_deg:.2f} deg')
                    L.append(f"marker side measured on the plane {ref.side_m * 1000:.1f} mm vs "
                             f"length_m {gp('marker.length_m') * 1000:.1f} mm "
                             '(> ~3% apart: re-measure length_m)')
                else:
                    L.append(f'depth refinement skipped: {ref.reason} (using PnP pose)')
        L.append('==============================================')
        print('\n'.join(L), flush=True)


def _put_text(img, text, org, scale, colour=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def _quat_to_rot(x, y, z, w) -> np.ndarray:
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _stamp_s(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def main(args=None):
    rclpy.init(args=args)
    node = PegDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
