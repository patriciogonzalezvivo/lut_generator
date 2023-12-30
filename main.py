import cv2
import numpy as np
import argparse

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import scipy as sp
import scipy.interpolate

import colour
from colour_checker_detection import detect_colour_checkers_segmentation

colour.plotting.colour_style()
# colour.utilities.describe_environment()

# colour.plotting.plot_image(colour.cctf_encoding(image));
D65 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']

CHECKER_ROW = 4 # Change to the number of rows in your color chart. ie: 4
CHECKER_COL = 6 # Change to the number of columns in your color chart. ie: 6
SPYDER_sRGB = [ [98, 187, 166],[126, 125, 174],[82, 106, 60],[87, 120, 155],[197, 145, 125],[112, 76, 60],
                [222, 118, 32],[58, 89, 160],[195, 79, 95],[83, 58, 106],[157, 188, 54],[238, 158, 25],
                [0, 127, 159],[192, 75, 145],[245, 205, 0],[186, 26, 51],[57, 146, 64],[25, 55, 135],
                [249, 242, 238],[202, 198, 195],[161, 157, 154],[122, 118, 116],[80, 80, 78],[43, 41, 43] ]


def newRainbow():
    grid = np.linspace(0,255, 2)
    rainbow = np.zeros((len(grid)*len(grid)*len(grid),3))
    counter = 0
    for r in grid:
        for g in grid:
            for b in grid:
                rainbow[counter,0]=r
                rainbow[counter,1]=g
                rainbow[counter,2]=b
                counter+=1
    return rainbow


# lut size of 64; 33 is also a common option for some other apps
def identity(ls=64):
    r=0
    g=0
    b=0
    img = np.zeros((ls*ls*ls,3), np.float32)
    for x in range(0,ls*ls*ls): # create a neutral LUT first
        if r >= ls:
            r = 0
            g += 1
        if g >= ls:
            g =0
            b +=1
        img[x,0] = 1/(ls-1) * r
        img[x,1] = 1/(ls-1) * g
        img[x,2] = 1/(ls-1) * b
        r+=1
    img = img.reshape((-1,ls*ls,3))
    return img * 255.0


def wrapper(standard, array, _size, _rows=8, _flip=False):
    if standard == "hald":
        array = array.reshape(_size**3, _size**3, 3)
    else:
        rows = _size
        if 0 < _rows < _size and _size % _rows == 0:
            rows = _rows
        array = array.reshape((rows,int(_size**2/rows+.5),_size**2,_size**2, 3))
        array = np.concatenate([np.concatenate(array[row], axis=1) for row in range(rows)])

    return (array,np.flipud(array))[_flip]

def applyPoly(image, poly):
    assert(image.dtype == "float32")
    for rgb in range(3):
        p = np.poly1d(poly[rgb])
        image[:,:,rgb] = p(image[:,:,rgb])
    return image

## FIT THE DATA TO OUR MULTIVARIATE FUNCTIONS
def polyfit3d(rgb, degrees, x0):  
    degrees = [(i, j, k) for i in range(degrees) for j in range(degrees) for k in range(degrees)]  # list of monomials x**i * y**j to use
    matrix = np.stack([np.prod(rgb.T**d, axis=1) for d in degrees], axis=-1)   # stack monomials like columns
    coeff = np.linalg.lstsq(matrix, x0, rcond=-1)[0]    # lstsq returns some additional info we ignore
    #print("Coefficients", coeff)    # in the same order as the monomials listed in "degrees"
    fit = np.dot(matrix, coeff)
    #print(np.sqrt(np.mean((x0-fit)**2)))  ## error
    return coeff
        

## PREDICT / SOLVE the function for our input data (getting our target data)
def poly3d(rgb, coeff, pp):  
    degrees = [(i, j, k) for i in range(pp) for j in range(pp) for k in range(pp)]  # list of monomials x**i * y**j to use
    matrix = np.stack([np.prod(rgb.T**d, axis=1) for d in degrees], axis=-1)   # stack monomials like columns
    fit = np.dot(matrix, coeff)
    return fit


def apply(image, _coes, _degrees):
    sss = np.shape(image[:,:,0])
    rgb = image.reshape(-1,3).T

    Zr = poly3d(rgb, _coes[0], _degrees).reshape(sss)
    Zg = poly3d(rgb, _coes[1], _degrees).reshape(sss)
    Zb = poly3d(rgb, _coes[2], _degrees).reshape(sss)

    image[:,:,0] = Zr
    image[:,:,1] = Zg
    image[:,:,2] = Zb
    
    return image


def compute(filename):
    coes = {}
    image = colour.cctf_decoding(colour.io.read_image(filename))
    SWATCHES = []
    for colour_checker_swatches_data in detect_colour_checkers_segmentation(image, additional_data=True):
        swatch_colours, colour_checker_image, swatch_masks = (colour_checker_swatches_data.values)
        SWATCHES.append(swatch_colours)

        # Convert swatch_colours to numpy array
        swatches_sRGB = colour.XYZ_to_sRGB( colour.RGB_to_XYZ(swatch_colours, 'sRGB', D65) )
        swatches_sRGB = np.array(swatches_sRGB) * 255
        
        # Using the additional data to plot the colour checker and masks.
        masks_i = np.zeros(colour_checker_image.shape)
        for i, mask in enumerate(swatch_masks):
            masks_i[mask[0]:mask[1], mask[2]:mask[3], ...] = 1

        # colour.plotting.plot_image( colour.cctf_encoding( np.clip(colour_checker_image + masks_i * 0.25, 0, 1)) );

        extracted_palette = np.array(swatches_sRGB).reshape((CHECKER_ROW,CHECKER_COL,3)).astype("uint8") 
        extracted_palette = cv2.cvtColor(extracted_palette, cv2.COLOR_RGB2BGR)
        extracted_palette = cv2.resize(extracted_palette,(200,150),interpolation=cv2.INTER_NEAREST)
        cv2.imwrite("extracted_palette_{}.png".format(len(SWATCHES)), extracted_palette)

        degreesA = 1  # <<====  Number of Polynomial degrees.  1 Recommended. 2 Max.
        
        colors = ["Red","Green","Blue"]
        poly = {}
        for rgb in range(3):
            www = np.ravel(swatches_sRGB)[rgb::3]
            rrr = np.ravel(SPYDER_sRGB)[rgb::3]

            poly[rgb] = np.polyfit(www, rrr, degreesA)
            p = np.poly1d(poly[rgb])
            
            delta = rrr - p(www)
            power = np.power(delta,2)
            mean = np.mean(power)
            final = np.sqrt(mean)
            print(colors[rgb],"error:",final)
            if (final>30):
                print("\t ^ High number implies result is not that optimized")
                continue

        degreesB = 2  ## 2 or 3 only. 3 can lead to overfitting. 2 recommend for most users.

        ## Create more data points using previous step's function; this will help prevent overfitting with our next step.
        rainbow = newRainbow()
        swatches_sRGB0 = np.vstack((swatches_sRGB, rainbow))

        for rgb in range(3):
            p = np.poly1d(poly[rgb])
            print(colors[rgb],"Pre-Process range:",int(np.min(rainbow[:,rgb])),"-",int(np.max(rainbow[:,rgb])))
            rainbow[:,rgb] = p(rainbow[:,rgb])
            print(colors[rgb],"Post-Process range:",int(np.min(rainbow[:,rgb])),"-",int(np.max(rainbow[:,rgb])))

        spyder_sRGB0 = np.vstack((SPYDER_sRGB, rainbow))

        r1 = np.ravel(swatches_sRGB0)[0::3].astype(np.float32, copy=False)  # swatches_sRGB
        g1 = np.ravel(swatches_sRGB0)[1::3].astype(np.float32, copy=False)
        b1 = np.ravel(swatches_sRGB0)[2::3].astype(np.float32, copy=False)

        r0 = np.ravel(spyder_sRGB0)[0::3].astype(np.float32, copy=False)  # spyder_sRGB
        g0 = np.ravel(spyder_sRGB0)[1::3].astype(np.float32, copy=False)
        b0 = np.ravel(spyder_sRGB0)[2::3].astype(np.float32, copy=False)

        rgb = np.array([r1,g1,b1])

        ## Generate and Save the functions; one function for each color type
        coes[0] = polyfit3d(rgb, degreesB, r0) 
        coes[1] = polyfit3d(rgb, degreesB, g0)
        coes[2] = polyfit3d(rgb, degreesB, b0)

        return coes, degreesB
    
    return coes, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', '-i', help="input", type=str, required=True)
    parser.add_argument('-output', '-o', help="output", type=str, default="lut")
    parser.add_argument('-target', '-t', help="brightness", type=str, default=None)
    parser.add_argument('-brightness', '-b', help="brightness", type=float, default=0)
    parser.add_argument('-size', '-s', help="LUT Cube size", type=int, default=64)
    args = parser.parse_args()

    coes, degrees = compute(args.input)


    if args.target is not None:
        LUT2 = np.zeros((492)) # LIGHT
        for y in range(22500,25500):
            x = int(round((y/100-225)/1 + 1/(1.2**(225-y/100))+225))
            LUT2[x-226] = int(round(y/100.0))
        LUT1 = 255-np.flip(LUT2)  # DARK

        # Load target
        lutimg = cv2.imread(args.target, 1) 
        lutimg = cv2.cvtColor(lutimg, cv2.COLOR_BGR2RGB)

        ## ANALYTICS
        avglum = np.mean(np.sqrt( 0.299*lutimg[:,:,0]**2 + 0.587*lutimg[:,:,1]**2 + 0.114*lutimg[:,:,2]**2 ))  # via: https://stackoverflow.com/questions/596216/formula-to-determine-brightness-of-rgb-color
        print("PRE PROCESSING - Dynamic range:",np.min(lutimg),"-",np.max(lutimg), ", Average Lumin Value:",avglum)

        lutimg = lutimg.astype(np.float32, copy=False)

        ## If we want to increase the brightness
        ## We will use what was set earlier if not defined below
        if (args.brightness != 0):
            print("\tfyi: Brightness is being adjusted from default; adding",str(mb),"more brightness")

        ## APPLY THE COLOR CORRECTION
        lutimg = apply(lutimg, coes, degrees) + args.brightness  ## Experimental alternative to applyLUT

        ## APPLY HIGHLIGHT GAMMA CURVE
        where = np.where((lutimg>225) &(lutimg<492))
        out = LUT2[np.uint16(np.round(lutimg[where]-225))]
        lutimg[where] = out

        ## APPLY SHADOW GAMMA CURVE
        where = np.where((lutimg<=30) & (lutimg>-461))  
        out = LUT1[np.uint16(np.round(lutimg[where]+461))]
        lutimg[where] = out

        ## CLEAN UP OUT-OF-BOUNDS DATA
        lutimg[np.where(lutimg>255)]=255 # cut out bad highlights
        lutimg[np.where(lutimg<0)]=0 # cut out bad shadows
        lutimg = lutimg.astype(np.uint8, copy=False)

        ## ANALYTICS
        avglum = np.mean(np.sqrt( 0.299*lutimg[:,:,0]**2 + 0.587*lutimg[:,:,1]**2 + 0.114*lutimg[:,:,2]**2 ))  # via: https://stackoverflow.com/questions/596216/formula-to-determine-brightness-of-rgb-color
        print("POST-PROCESSING - Dynamic range:",np.min(lutimg),"-",np.max(lutimg), ", Average Lumin Value:",avglum)
        cv2.imwrite(args.target + "_" + args.output + ".png", cv2.cvtColor(lutimg,cv2.COLOR_RGB2BGR))

    lut = identity(args.size)
    lut = apply(lut, coes, degrees) + args.brightness # apply the basic color correction

    # apply any highlight, shadow, thesholding here... I'm skipping it for now with CUBE luts
    with open(args.output + '.cube', 'w') as the_file: # save the CUBE LUT to this colab folder I guess
        the_file.write("LUT_3D_SIZE "+str(args.size)+"\n")
        lut_hald = (lut/255.0).reshape((-1,3))
        for x in range(args.size*args.size*args.size):
            the_file.write("{:1.6f}".format(lut_hald[x,0])+" "+"{:1.6f}".format(lut_hald[x,1])+" "+"{:1.6f}".format(lut_hald[x,2])+'\n')

    if args.target == None:
        size = int(np.sqrt(args.size))
        cv2.imwrite(args.output + ".png", cv2.cvtColor(wrapper("square", lut, size), cv2.COLOR_RGB2BGR))

    
