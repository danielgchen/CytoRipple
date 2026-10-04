# Imports
# > Public Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List
# > Private Packages
from consts import *


def plot_vectors(vectors: pd.DataFrame, **kwargs) -> plt.Figure:
    """
    Plot a given set of vectors with additional kwargs determining plotting outcome

    Parameters:
    - vectors = dataframe with vector centroids (x0, y0) and unit vectors (unit_u, unit_v)
    - figsize = figure size to plot on, defaults to FIGSIZE_PLOT or [8, 8]
    - width = width of the arrow, defaults to QUIVER_WIDTH 0.002
    - scale = scale of the arrow, defaults to QUIVER_SCALE 50
    - xlim = x-axis limits of the plot, defaults to XLIM_PLOT [0, 2048]
    - ylim = y-axis limits of the plot, defaults to YLIM_PLOT [0, 2048]
    Returns:
    - matplotlib figure with the plotted data
    """
    # prepare plotting values
    # > figure size
    if 'figsize' not in kwargs: kwargs['figsize'] = FIGSIZE_PLOT
    # > quiver parameters
    if 'width' not in kwargs: kwargs['width'] = QUIVER_WIDTH
    if 'scale' not in kwargs: kwargs['scale'] = QUIVER_SCALE
    # > axis limits
    if 'xlim' not in kwargs: kwargs['xlim'] = XLIM_PLOT
    if 'ylim' not in kwargs: kwargs['ylim'] = YLIM_PLOT
    # > centroid size and color
    if 'color' not in kwargs: kwargs['color'] = CENTROID_COLOR
    if 's' not in kwargs: kwargs['s'] = CENTROID_SIZE
        
    # setup matplotlib figure
    fig, ax = plt.subplots(figsize=kwargs['figsize'])
    ax.grid(False)
    # plot quivers
    ax.scatter(vectors['x0'], vectors['y0'], color=kwargs['color'], s=kwargs['s'])
    ax.quiver(vectors['x0'], vectors['y0'], vectors['unit_u'], vectors['unit_v'],
              width=kwargs['width'], scale=kwargs['scale'])
    # set axis limits for plotting
    ax.set_xlim(*XLIM_PLOT)
    ax.set_ylim(*YLIM_PLOT)
    return fig


def plot_curl_and_divergence(XI: np.ndarray, YI: np.ndarray, curl: np.ndarray, divergence: np.ndarray, **kwargs) -> plt.Figure:
    """
    Plots both the curl and divergence of a grid-snapped vector field
    
    Parameters:
    - XI = grid-snapped x-coordinates
    - YI = grid-snapped y-coordinates
    - curl = degree of curl in a given area
    - divergence = degree of divergence in a given area
    Returns:
    - matplotlib figure with one subplot on the curl and one subplot on the divergence
    """
    # prepare plotting values
    # > figure size
    if 'figsize' not in kwargs: kwargs['figsize'] = FIGSIZE_CD
    # > colormap
    if 'cmap' not in kwargs: kwargs['cmap'] = CMAP_CD
    # > axis limits
    if 'xlim' not in kwargs: kwargs['xlim'] = XLIM_PLOT
    if 'ylim' not in kwargs: kwargs['ylim'] = YLIM_PLOT
        
    # setup the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=kwargs['figsize'])
    # plot curl of the vector field
    im1 = ax1.imshow(curl, extent=XLIM_PLOT+YLIM_PLOT, origin='lower', cmap=kwargs['cmap'], aspect='equal')
    ax1.set_title('Curl of Vector Field')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1, label='Curl')
    # plot divergence of the vector field
    im2 = ax2.imshow(divergence, extent=XLIM_PLOT+YLIM_PLOT, origin='lower', cmap=kwargs['cmap'], aspect='equal')
    ax2.set_title('Divergence of Vector Field')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y')
    plt.colorbar(im2, ax=ax2, label='Divergence')
    # tighten layout
    fig.tight_layout()
    return fig