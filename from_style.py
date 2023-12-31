import os
import cv2
import numpy as np
import argparse
from common.cube import identity, wrapper
from common.io import writeLutCube, writeLutImage

from color_matcher import ColorMatcher
from color_matcher.io_handler import load_img_file, save_img_file, FILE_EXTS
from color_matcher.normalizer import Normalizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', '-i', help="input", type=str, required=True)
    parser.add_argument('-output', '-o', help="output", type=str, required=True)
    args = parser.parse_args()

    img_ref = load_img_file(args.input)
    # print(img_ref.shape)

    
    img_lut = wrapper(identity(), int(np.sqrt(64)))
    # cv2.imwrite("test.png", cv2.cvtColor(img_lut, cv2.COLOR_RGB2BGR))

    cm = ColorMatcher()
    img_res = cm.transfer(src=img_lut, ref=img_ref, method='mkl')
    img_res = Normalizer(img_res).uint8_norm()
    save_img_file(img_res, args.output)