import nnrt
import sys
import cv2
import numpy as np

PROGRAM_EXIT_SUCCESS = 0
PROGRAM_EXIT_FAIL = 1


def main():
    depth1 = cv2.imread("../example_data/test/seq017/depth/000300.png", cv2.IMREAD_UNCHANGED)
    filtered_depth1 = nnrt.filter_depth(depth1, 2)




    cv2.waitKey()

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
