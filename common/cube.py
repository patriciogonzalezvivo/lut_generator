import numpy as np

# lut size of 64; 33 is also a common option for some other apps
def identity(ls=64):
    r = 0
    g = 0
    b = 0
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


def wrapper(array, size, standard="square", _rows=8, _flip=False):
    if standard == "hald":
        array = array.reshape(size**3, size**3, 3)
    else:
        rows = size
        if 0 < _rows < size and size % _rows == 0:
            rows = _rows
        array = array.reshape((rows,int(size**2/rows+.5),size**2,size**2, 3))
        array = np.concatenate([np.concatenate(array[row], axis=1) for row in range(rows)])

    return (array,np.flipud(array))[_flip]