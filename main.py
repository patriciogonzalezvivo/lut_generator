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
colour.utilities.describe_environment();

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
def newLUT(ls=64):
    r=0
    g=0
    b=0
    img = np.zeros((ls*ls*ls,3), np.float32)
    for x in range(0,ls*ls*ls): # create a neutral LUT first
        if r>=ls:
            r=0
            g+=1
        if g>=ls:
            g=0
            b+=1
        img[x,0]=1/(ls-1)*r
        img[x,1]=1/(ls-1)*g
        img[x,2]=1/(ls-1)*b
        r+=1
    img = img.reshape((-1,ls*ls,3))
    return img * 255.0


## FIT THE DATA TO OUR MULTIVARIATE FUNCTIONS
def polyfit3d(rgb,pp,x0):  
    degrees = [(i, j, k) for i in range(pp) for j in range(pp) for k in range(pp)]  # list of monomials x**i * y**j to use
    matrix = np.stack([np.prod(rgb.T**d, axis=1) for d in degrees], axis=-1)   # stack monomials like columns
    coeff = np.linalg.lstsq(matrix, x0)[0]    # lstsq returns some additional info we ignore
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


def applyLUT(image):
    assert(image.dtype == "float32")
    for rgb in range(3):
        p = np.poly1d(poly[rgb])
        print("PRE range:",np.min(image[:,:,rgb]),"-",np.max(image[:,:,rgb]))
        image[:,:,rgb] = p(image[:,:,rgb])
        print("POST range:",np.min(image[:,:,rgb]),"-",np.max(image[:,:,rgb]))
        print()
    return image


def applyLUT2(image, pp, _coes):
    sss = np.shape(image[:,:,0])
    rgb = image.reshape(-1,3).T
    print(np.shape(rgb))
    Zr = poly3d(rgb,_coes[0], pp).reshape(sss)
    Zg = poly3d(rgb,_coes[1], pp).reshape(sss)
    Zb = poly3d(rgb,_coes[2], pp).reshape(sss)
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

        colour.plotting.plot_image( colour.cctf_encoding( np.clip(colour_checker_image + masks_i * 0.25, 0, 1)) );

        extracted_palette = np.array(swatches_sRGB).reshape((CHECKER_ROW,CHECKER_COL,3)).astype("uint8") 
        extracted_palette = cv2.cvtColor(extracted_palette, cv2.COLOR_RGB2BGR)
        extracted_palette = cv2.resize(extracted_palette,(200,150),interpolation=cv2.INTER_NEAREST)
        cv2.imwrite("extracted_palette_{}.png".format(len(SWATCHES)), extracted_palette)

        degreesA = 1  # <<====  Number of Polynomial degrees.  1 Recommended. 2 Max.

        poly={}
        colors = ["Red","Green","Blue"]
        xp = np.linspace(0, 255, 255)
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
            # plt.subplot(3,1,rgb+1)
            # _ = plt.plot(www, rrr, '.', xp, xp, '-', xp, p(xp), '--')
            # plt.ylim(0,255)
            # plt.show()
            if (final>30):
                print("\t ^ High number implies result is not that optimized")
                continue
        
        degreesB=2  ## 2 or 3 only. 3 can lead to overfitting. 2 recommend for most users.

        ## Create more data points using previous step's function; this will help prevent overfitting with our next step.
        rainbow = newRainbow()
        swatches_sRGB0 = np.vstack((swatches_sRGB, rainbow))

        for rgb in range(3):
            p = np.poly1d(poly[rgb])
            print(colors[rgb],"Pre-Process range:",int(np.min(rainbow[:,rgb])),"-",int(np.max(rainbow[:,rgb])))
            rainbow[:,rgb] = p(rainbow[:,rgb])
            print(colors[rgb],"Post-Process range:",int(np.min(rainbow[:,rgb])),"-",int(np.max(rainbow[:,rgb])))
            print()

        spyder_sRGB0 = np.vstack((SPYDER_sRGB,rainbow))

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

        # ## PLOT DATA POINTS FOR ANALYSIS
        # rgb0 = np.array([   poly3d(rgb,coes[0],degreesB),
        #                     poly3d(rgb,coes[1],degreesB), 
        #                     poly3d(rgb,coes[2],degreesB)])

        # fig = plt.figure(figsize=(12,6))
        # ax = axes3d.Axes3D(fig)

        # for i in range(len(rgb[0])):
        #     ax.plot(  [rgb[0,i],rgb0[0,i]],  [rgb[1,i],rgb0[1,i]],  [rgb[2,i],rgb0[2,i]],  'ro-')

        # ax.scatter3D(rgb0[0],rgb0[1],rgb0[2], c='g')
        # ax.scatter3D(rgb[0],rgb[1],rgb[2], c='b') 

        # # The graph reflects the transformation of each RGB colour value

        return coes, degreesA
    
    return coes, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', '-i', help="input", type=str, required=True)
    args = parser.parse_args()

    coes, degrees = compute(args.input)

    mb = 0

    ## We are going to download a "neutral" PNG LUT and modify it. You can use your own LUTs instead if you want to modify the code a bit
    img_file = "neutral-lut.png"

    # LOAD INTO PYTHON
    lutimg = cv2.imread(img_file, 1) 
    lutimg = cv2.cvtColor(lutimg, cv2.COLOR_BGR2RGB)

    ## ANALYTICS
    avglum = np.mean(np.sqrt( 0.299*lutimg[:,:,0]**2 + 0.587*lutimg[:,:,1]**2 + 0.114*lutimg[:,:,2]**2 ))  # via: https://stackoverflow.com/questions/596216/formula-to-determine-brightness-of-rgb-color
    print("PRE PROCESSING - Dynamic range:",np.min(lutimg),"-",np.max(lutimg), ", Average Lumin Value:",avglum)

    lutimg = lutimg.astype(np.float32, copy=False)

    ## If we want to increase the brightness
    ## We will use what was set earlier if not defined below
    # mb = 0 # default
    if (mb!=0):
        print()
        print("\tfyi: Brightness is being adjusted from default; adding",str(mb),"more brightness")
    
    print()
    print(" R-> G-> B-> ")
    print()

    ## APPLY THE COLOR CORRECTION
    #lutimg = applyLUT(lutimg) + mb
    lutimg = applyLUT2(lutimg,degrees, coes) + mb  ## Experimental alternative to applyLUT

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
    print()
    print("Done processing!")
    print("--> RIGHT CLICK IMAGE AND SELECT 'SAVE IMAGE AS' TO SAVE; as .png filetype ideally <--")
    ## GENERATE THE FINAL LUT
    # cv2_imshow(cv2.cvtColor(lutimg,cv2.COLOR_RGB2BGR)) 
    cv2.imwrite("lut.png", lutimg,cv2.COLOR_RGB2BGR)



    # ls = 64
    # lut = newLUT(ls)
    # lut = applyLUT2(lut, degrees, coes) + mb # apply the basic color correction

    # # apply any highlight, shadow, thesholding here... I'm skipping it for now with CUBE luts
    # lut = lut/255.0
    # lut = lut.reshape((-1,3))

    # with open('result.cube', 'w') as the_file: # save the CUBE LUT to this colab folder I guess
    #     the_file.write("LUT_3D_SIZE "+str(ls)+"\n")
    #     for x in range(ls*ls*ls):
    #         the_file.write("{:1.6f}".format(lut[x,0])+" "+"{:1.6f}".format(lut[x,1])+" "+"{:1.6f}".format(lut[x,2])+'\n')

    
