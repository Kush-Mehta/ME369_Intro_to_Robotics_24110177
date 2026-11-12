"""Write a printable ArUco marker PNG.

    ros2 run peg_detection generate_marker --id 0 --out ~/robot_base_marker.png

Print at 100% scale (no "fit to page"), then MEASURE the black square's side
with a ruler and put that value in config: marker.length_m.
"""
import argparse
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(description='Generate an ArUco marker PNG')
    ap.add_argument('--dict', default='DICT_4X4_50')
    ap.add_argument('--id', type=int, default=0)
    ap.add_argument('--pixels', type=int, default=800, help='side of the black square in px')
    ap.add_argument('--border', type=int, default=100, help='white quiet zone in px')
    ap.add_argument('--out', default=os.path.expanduser('~/robot_base_marker.png'))
    args = ap.parse_args()

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dict))
    if hasattr(cv2.aruco, 'generateImageMarker'):
        img = cv2.aruco.generateImageMarker(dictionary, args.id, args.pixels)
    else:
        img = cv2.aruco.drawMarker(dictionary, args.id, args.pixels)
    img = cv2.copyMakeBorder(img, args.border, args.border, args.border, args.border,
                             cv2.BORDER_CONSTANT, value=255)
    cv2.putText(img, f'{args.dict} id={args.id}', (10, img.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    out = os.path.expanduser(args.out)
    if not cv2.imwrite(out, img.astype(np.uint8)):
        raise SystemExit(f'Could not write {out}')
    print(f'Wrote {out}  ({args.dict}, id {args.id}).')
    print('Print at 100% scale, measure the black square side in metres, '
          'and set marker.length_m in config/detection_params.yaml.')


if __name__ == '__main__':
    main()
