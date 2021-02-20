import nnrt
import sys
import cv2
import numpy as np

PROGRAM_EXIT_SUCCESS = 0
PROGRAM_EXIT_FAIL = 1


def main():
    depth1 = cv2.imread("example_data/test/seq017/depth/000300.png", cv2.IMREAD_UNCHANGED)
    
    filtered_depth1 = nnrt.filter_depth(depth1, 2)

    fx = 575.548
    fy = 577.46
    cx = 323.172
    cy = 236.417
    point_cloud = nnrt.backproject_depth_ushort(filtered_depth1, fx, fy, cx, cy, 1000)





    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
