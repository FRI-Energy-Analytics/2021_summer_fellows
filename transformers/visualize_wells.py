""" Module docstring
TODO describe module
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scripts.csv_extractor import get_gamma,get_depth

# Unused
##import pandas as pd

def plot_well(file_name):
    """ Plot wells
    TODO describe
    """
    cmap = mpl.cm.bone_r  # select our color map
    try:
        x = get_gamma(None)[::50]
        x[np.isnan(x)] = 0
        y = get_depth(None)[::50]
        z = x
    except Exception as E:
        print(file_name)
        print(E)
        raise Exception()

    normalize = mpl.color.Normalize(vmin=z.min(), vmax=z.max())

    plt.plot(x, y, color="gray")
    ax = plt.gca()

    ax.invert_yaxis()

    for j in range(x.size - 1):
        plt.fill_betweenx(
            [y[j], y[j + 1]], [x[j], x[j + 1]], x2=z.max(), color=cmap(normalize(z[j]))
        )
    plt.show()
