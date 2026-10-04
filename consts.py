import torch

# Scale Factor
# 0.390625 for MIBI
# 0.2125 for Xenium
PIXEL_TO_MICRON = 0.390625
MICRON_TO_PIXEL = 1 / PIXEL_TO_MICRON

# Axis Limits for Plotting
XLIM_PLOT = YLIM_PLOT = [0, 2048]

# Figure Size for Plotting
FIGSIZE_PLOT = [8, 8]

# Minimum # of Points to be a Cell
MIN_POINTS = 3

# Maximum Distance to Consider When Smoothing
SMOOTH_DIST_LIMIT_MICRONS = 30
SMOOTH_DIST_LIMIT_PIXEL = SMOOTH_DIST_LIMIT_MICRONS * MICRON_TO_PIXEL

# Minimum Vector Magnitude to Consider
MIN_VECTOR_MAGNITUDE = 1
# Number of Iterations to Implement Spatial Smoothing
SMOOTH_ITERS = 1

# Number of Bins for Curl/Divergence Grid Metrics on the X and Y Axes
NGRIDX = 100
NGRIDY = 100

# Arrow Parameters
QUIVER_WIDTH = 0.002
QUIVER_SCALE = 50

# Centroid Parameters
CENTROID_COLOR = 'lightgray'
CENTROID_SIZE = 2

# Curl and Divergence Plotting Parameters
FIGSIZE_CD = [16, 8]
CMAP_CD = 'viridis'
