# -*- coding: utf-8 -*-

""" 
Basic color bar alterations

"""

__title__ = "Replace volors"
__author__ = "Arden Burrell"
__version__ = "1.0 (18.02.2018)"
__email__ = "arden.burrell@gmail.com"

# modules
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mpc
import webcolors as wc

import pdb

def _ColorConversion(color):
    """
    A better converter for getting the color into RGB
    args:
        color in hex or named color format
    returns R G B """
    try:
        if  type(color).__name__ == "tuple": # color is RGB
            cR = color[0]
            cG = color[1]
            cB = color[2]
        elif mpc.is_color_like(color): 
            try:
                cR, cG, cB = wc.hex_to_rgb(color)
            except ValueError:
                try:
                    cR, cG, cB = wc.name_to_rgb(color)
                except ValueError:
                    cR, cG, cB, _ = mpc.to_rgba_array(color)[0] * 255
        return cR, cG, cB
    except UnboundLocalError:
        raise ValueError(
            "Color cannot be converted into RGB: " + color + "is not a valid color")

# def _segmented(cmap, ShiftInd, cdict):



    # pass

def ReplaceColor(cmap, color = 'w', start = 0., locpoint = 0.5, stop = 1.0, 
    name = 'centered', alpha = 1.0, segmented=True):
    """ Replace the colour of a value standing anywhere in the new cmap
    (relatively to the two extremes start & stop or min & max) with whichever
    value / colour of the input cmap (by default the midpoint).
    The locpoint value cannot be the min or max (start or stop).
    Args: 
        cmap An existing color map,
        color: the color to replace, can be hex, RGB or named color. defualt = white
    """

    # ========== convert the colors into RGB ==========
    cA = alpha
    cR, cG, cB = _ColorConversion(color)

    # declare a colour + transparency dictionary
    cdict={'red':[], 'green':[], 'blue':[], 'alpha':[]}

    # regular index to compute the colors
    # pdb.set_trace()
    this_zero = 1.e-9

    RegInd = np.linspace(start, stop, cmap.N)

    # shifted index to match what the data should be centered on
    ShiftInd = np.hstack([np.linspace(0., locpoint, cmap.N/2, endpoint = False),np.linspace(locpoint, 1., cmap.N/2, endpoint = True)])
    # ShiftInd = np.hstack([np.linspace(0., 0.5, cmap.N/2, endpoint = False),0.5,np.linspace(0.5+this_zero, 1., cmap.N/2, endpoint = True)])
    # norm = mpc.BoundaryNorm(ShiftInd, cmap)
    # return norm
    # pdb.set_trace()
    # associate the regular cmap's colours 
    for RI, SI in zip(RegInd, ShiftInd):
        if SI == locpoint: #replace the locpoint with 
            cdict['red'].append((SI, cR, cR))
            cdict['green'].append((SI, cG, cG))
            cdict['blue'].append((SI, cB, cB))
            cdict['alpha'].append((SI, cA, cA))
        else:
            # get standard indexation of red, green, blue, alpha
            r, g, b, a = cmap(RI)
            cdict['red'].append((SI, r, r))
            cdict['green'].append((SI, g, g))
            cdict['blue'].append((SI, b, b))
            cdict['alpha'].append((SI, a, a))
    
    # pdb.set_trace()
    return LinearSegmentedColormap(name, cdict)
    # return mpc.ListedColormap(name, cdict)