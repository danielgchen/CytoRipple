# Imports
# > Public Packages
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
from typing import Dict, List
# > Private Packages
from consts import *

def process_segment_acc(points: torch.Tensor, intensities_acc: torch.Tensor) -> List[float]:
    """
    Compute the directionality of a given set of intensities for a given segment

    Parameters:
    - points: numpy array of the segmented points defining a cell
    - intensities_acc: intensities of all points
    Returns:
    - centroid (x0, y0) of the cell by its segmentation
    - movement vector (xv, yv) not scaled yet
    """
    # compute centroid of the segmented object
    centroid = torch.mean(points, dim=0)
    # move centroid to cpu and decompose into x, y arms
    x0, y0 = centroid[0].item(), centroid[1].item()
    # retrieve intensities for the segmented points casting to int/long
    cs = intensities_acc[points[:, 1].long(), points[:, 0].long()]
    # compute average intensities on each side of the centroid
    xl = cs[points[:, 0] < x0].mean()
    xh = cs[points[:, 0] >= x0].mean()
    yl = cs[points[:, 1] < y0].mean()
    yh = cs[points[:, 1] >= y0].mean()
    # compute single-dimension vector components
    xv = (xh - xl).item()
    yv = (yh - yl).item()
    return [x0, y0, xv, yv]


def generate_vector_field(segmentation_dict: Dict[int, List[int]], intensities_acc: torch.FloatTensor, device: torch.device) -> pd.DataFrame:
    """
    Generate vectors for all QC-passing segmented objects

    Parameters:
    - segmentation_dict: dictionary mapping cell's to segmented points
    - intensities_acc: intensities of all points
    Returns:
    - dataframe containing objects' centroids and unscaled movement vectors
    """
    # instantiate tracking objects for segmentation id and vectors
    df = []
    # process each segmented object
    for seg_id in segmentation_dict:
        # retrieve points for said objects
        seg_points = torch.tensor(segmentation_dict[seg_id], device=device, dtype=torch.float32)
        # skip objects with too few points (likely noise)
        if len(seg_points) < MIN_POINTS: continue
        # retrieve centroid and movement vector
        result = process_segment_acc(seg_points, intensities_acc)
        # save results and add segment id
        df.append(result + [seg_id])
    # convert into pandas dataframe
    df = pd.DataFrame(df, columns=['x0', 'y0', 'xv', 'yv', 'segment'])
    # calculate magnitude to allow for proper scaling later on
    df['magnitude'] = np.linalg.norm(df[['xv', 'yv']], axis=1)
    return df


def _spatial_smooth(values: pd.Series, dist_segment: pd.DataFrame, smooth_dist_limit_pixel: int) -> pd.Series:
    """
    Smoothes a given set of values by their spatial distances to other segments

    Parameters:
    - values = raw values at a segment resolution
    - dist_segment = segment to segment distances
    - smooth_dist_limit_pixel = maximum distance to consider for spatially weighted smoothing
    Returns:
    - spatial distance smoothed values
    """
    # instantiate a tracker of new values
    new_values = []
    # for each barcode in the current list of values
    for bc in values.index:
        # retrieve the distance to other segments
        dists = dist_segment.loc[values.index, bc]
        # identify the maximum limit to consider in microns
        # only keep distances within said maximum limit
        dists = dists[dists < smooth_dist_limit_pixel]
        # compute weights as a function of distance away from segment
        weights = (smooth_dist_limit_pixel - dists) / smooth_dist_limit_pixel
        # renormalize so weights sum to one
        weights /= weights.sum()
        # derive the distance weighted smoothed values
        new_values.append((values.loc[weights.index] * weights).mean())
    # move into new pandas series for easier access
    new_values = pd.Series(new_values, index=values.index, dtype=float)
    return new_values
    

def spatial_smooth(values_original: pd.Series, t: int, dist_segment: pd.DataFrame, smooth_dist_limit_pixel: int) -> pd.Series:
    """
    Implements spatial smoothing `t` number of times

    Parameters:
    - values_original = original raw values at a segment resolution
    - t = number of times to implement spatial smoothing
    - dist_segment = segment to segment distances
    - smooth_dist_limit_pixel = maximum distance to consider for spatially weighted smoothing
    Returns:
    - spatial distance smoothed values
    """
    # copy over the original values to the new object
    values = values_original.copy()
    # implement spatial smoothing t times
    for _ in range(t):
        values = _spatial_smooth(values, dist_segment, smooth_dist_limit_pixel)
    return values


def smooth_vector_field(df: pd.DataFrame, min_magnitude: int, smooth_iters: int, smooth_dist_limit_pixel: int) -> pd.DataFrame:
    """
    Smooth the non-scaled vectors and convert into unit vectors

    Parameters:
    - df = dataframe of the unscaled vectors and their centroids
    - min_magnitude = minimum magnitude of the unscaled vectors to consider as valid
    - smooth_iters = the number of iterations to perform spatially-weighted smoothing
    - smooth_dist_limit_pixels = the radius to consider when smoothing by distance
    Returns:
    - dataframe with the segment id, centroid, and unit movement vector
    """
    # calculate distance matrix between each object
    dist_segment = cdist(df[['x0','y0']], df[['x0','y0']])
    dist_segment = pd.DataFrame(dist_segment, index=df.index, columns=df.index)
    # only consider vectors above a minimum magnitude
    mask = df['magnitude'] >= min_magnitude
    data = df.loc[mask].copy()
    # smooth vectors by spatial distance
    data['smooth_u'] = spatial_smooth(data['xv'], smooth_iters, dist_segment, smooth_dist_limit_pixel)
    data['smooth_v'] = spatial_smooth(data['yv'], smooth_iters, dist_segment, smooth_dist_limit_pixel)
    # normalize into unit vectors
    data['smooth_magnitude'] = np.linalg.norm(data[['smooth_u', 'smooth_v']], axis=1)
    data['unit_u'] = data['smooth_u'] / data['smooth_magnitude']
    data['unit_v'] = data['smooth_v'] / data['smooth_magnitude']
    return data[['segment','x0', 'y0', 'unit_u', 'unit_v']]


def calculate_curl_and_divergence(df: pd.DataFrame) -> List[np.ndarray]:
    """
    Calculates the curl of the grid-snapped vectors and the XY divergence of said vectors

    Parameters:
    - df = dataframe with x,y coordinates of vectros and their u,v unit velocities
    Returns:
    - grid-snapped X and Y coordinates and curl and divergence metrics
    """
    # unpack object
    x, y, u, v = df['x0'], df['y0'], df['unit_u'], df['unit_v']
    # create a regular grid to calculate 
    xi = np.linspace(x.min(), x.max(), NGRIDX)
    yi = np.linspace(y.min(), y.max(), NGRIDY)
    XI, YI = np.meshgrid(xi, yi)
    # interpolate u and v components onto the regular grid
    UI = griddata((x, y), u, (XI, YI), method='linear')
    VI = griddata((x, y), v, (XI, YI), method='linear')
    # calculate gradients of u and v as a function of x and y
    dudy, dudx = np.gradient(UI, yi, xi)
    dvdy, dvdx = np.gradient(VI, yi, xi)
    # calculate curl which is change in dy by x minus change in dx by change in y
    curl = dvdx - dudy
    # calculate divergence, how much do the vectors separate from each other
    divergence = dudx + dvdy
    return [XI, YI, curl, divergence]

