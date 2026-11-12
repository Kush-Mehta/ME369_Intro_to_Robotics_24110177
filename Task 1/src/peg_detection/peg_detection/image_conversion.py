"""Convert between sensor_msgs/Image and NumPy arrays without cv_bridge.

Doing the conversion ourselves removes one compiled dependency and makes the
units explicit: RealSense publishes depth as 16UC1 integers in millimetres.
"""
import sys

import cv2
import numpy as np
from sensor_msgs.msg import Image

# encoding -> (numpy dtype, channels)
_ENCODINGS = {
    'rgb8': (np.uint8, 3),
    'bgr8': (np.uint8, 3),
    'mono8': (np.uint8, 1),
    '8UC1': (np.uint8, 1),
    '16UC1': (np.uint16, 1),
    'mono16': (np.uint16, 1),
    '32FC1': (np.float32, 1),
}


def image_msg_to_numpy(msg: Image) -> np.ndarray:
    """Return a read-only view of the image data with shape (H, W) or (H, W, C)."""
    if msg.encoding not in _ENCODINGS:
        raise ValueError(f'Unsupported image encoding: {msg.encoding}')
    dtype, channels = _ENCODINGS[msg.encoding]
    dtype = np.dtype(dtype)
    msg_big_endian = bool(msg.is_bigendian)
    host_big_endian = sys.byteorder == 'big'
    if dtype.itemsize > 1 and msg_big_endian != host_big_endian:
        dtype = dtype.newbyteorder()

    buf = np.frombuffer(bytes(msg.data), dtype=dtype)
    row_elems = msg.step // dtype.itemsize          # rows may be padded
    arr = buf.reshape(msg.height, row_elems)[:, : msg.width * channels]
    if channels > 1:
        arr = arr.reshape(msg.height, msg.width, channels)
    return arr


def colour_msg_to_bgr(msg: Image) -> np.ndarray:
    """Colour image -> writable BGR uint8 array (OpenCV's native order)."""
    arr = image_msg_to_numpy(msg)
    if msg.encoding == 'rgb8':
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if msg.encoding == 'bgr8':
        return arr.copy()
    if msg.encoding in ('mono8', '8UC1'):
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    raise ValueError(f'Not a colour encoding: {msg.encoding}')


def depth_msg_to_metres(msg: Image, depth_scale: float) -> np.ndarray:
    """Depth image -> float32 array in metres. Invalid pixels become 0.0.

    16UC1 / mono16: integer units, multiplied by depth_scale (0.001 for D435i).
    32FC1: already metres; NaN/inf are mapped to 0.
    """
    arr = image_msg_to_numpy(msg)
    if msg.encoding in ('16UC1', 'mono16'):
        return arr.astype(np.float32) * float(depth_scale)
    if msg.encoding == '32FC1':
        out = arr.astype(np.float32, copy=True)
        out[~np.isfinite(out)] = 0.0
        return out
    raise ValueError(f'Not a depth encoding: {msg.encoding}')


def bgr_to_image_msg(img: np.ndarray, header) -> Image:
    """BGR uint8 array -> sensor_msgs/Image (bgr8)."""
    msg = Image()
    msg.header = header
    msg.height, msg.width = int(img.shape[0]), int(img.shape[1])
    msg.encoding = 'bgr8'
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(img, dtype=np.uint8).tobytes()
    return msg
