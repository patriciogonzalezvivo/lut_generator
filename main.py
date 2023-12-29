import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

import colour

from colour_checker_detection import (
    ROOT_RESOURCES_EXAMPLES,
    SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC,
    colour_checkers_coordinates_segmentation,
    detect_colour_checkers_segmentation)
from colour_checker_detection.detection.segmentation import (
    adjust_image)

colour.plotting.colour_style()
colour.utilities.describe_environment();

image = colour.cctf_decoding(colour.io.read_image("assets/R0010007_crop.jpg"))
# colour.plotting.plot_image(colour.cctf_encoding(image));

SWATCHES = []
for colour_checker_swatches_data in detect_colour_checkers_segmentation(image, additional_data=True):
    swatch_colours, colour_checker_image, swatch_masks = (colour_checker_swatches_data.values)
    SWATCHES.append(swatch_colours)

    # Convert swatch_colours to numpy array
    swatch_colours = np.array(swatch_colours)
    print(swatch_colours.shape)
    
    # Using the additional data to plot the colour checker and masks.
    masks_i = np.zeros(colour_checker_image.shape)
    for i, mask in enumerate(swatch_masks):
        masks_i[mask[0]:mask[1], mask[2]:mask[3], ...] = 1

    colour.plotting.plot_image(
        colour.cctf_encoding(
            np.clip(colour_checker_image + masks_i * 0.25, 0, 1)));

    total_rows = 4  # Change to the number of rows in your color chart. ie: 4
    total_columns = 6 # Change to the number of columns in your color chart. ie: 6
    test = np.array(swatch_colours * 255).reshape((total_rows,total_columns,3)).astype("uint8") 
    test = cv2.cvtColor(test, cv2.COLOR_RGB2BGR)
    test = cv2.resize(test,(200,150),interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("test_{}.png".format(len(SWATCHES)), test)
