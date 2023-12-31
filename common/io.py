import cv2
import numpy as np

from common.cube import wrapper

def writeLutCube(filename, lut, size):
    # apply any highlight, shadow, thesholding here... I'm skipping it for now with CUBE luts
    with open(filename, 'w') as the_file: # save the CUBE LUT to this colab folder I guess
        the_file.write("LUT_3D_SIZE "+str(size)+"\n")
        lut_hald = (lut/255.0).reshape((-1,3))
        for x in range(size*size*size):
            the_file.write("{:1.6f}".format(lut_hald[x,0])+" "+"{:1.6f}".format(lut_hald[x,1])+" "+"{:1.6f}".format(lut_hald[x,2])+'\n')


def writeLutImage(filename, lut, size, standard="square"):
    cv2.imwrite(filename, cv2.cvtColor(wrapper(standard, lut, int(np.sqrt(size))), cv2.COLOR_RGB2BGR))