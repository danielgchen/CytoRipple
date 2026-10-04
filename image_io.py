# Imports
# > Public Packages
import numpy as np
import skimage.io as skio
import tifffile
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Tuple


def load_data(segmentation_file: str, marker_file: str) -> Tuple[np.ndarray, skio.ImageCollection]:
    """
    Loads segmentation and marker files

    Parameters:
    - segmentation_file = name of segmentation TIFF
    - marker_file = name of TIFF with markers of interest
    Returns
    - image collection objects for segmentation and intensities
    """
    # read in segmentation image stack
    segmentation_imstack = skio.imread(segmentation_file, plugin='tifffile')
    # read in marker image stack
    marker_imstack = skio.imread_collection(marker_file, plugin='tifffile')
    
    return segmentation_imstack, marker_imstack


def create_segmentation_dict(segmentation_imstack: np.ndarray) -> Dict[int, List[int]]:
    """
    Creates dictionary to map each cell to its segmented points

    Parameters:
    - segmentation_imstack = numpy array of the segementation image
    Returns:
    - dictionary of each cell's segmented points
    """
    # instantiate dictionary with default values as empty lists
    segmentation_dict = defaultdict(list)
    # loop through each segmentation image y-coordinate
    for y in range(len(segmentation_imstack)):
        # loop through each x-coordinate for that given x-coordinate
        for x in range(len(segmentation_imstack[y])):
            # retrieve segmentation id
            seg_id = segmentation_imstack[y][x]
            segmentation_dict[seg_id].append([x, y])
    # IDs of zero are not segments thus need to be removed
    del segmentation_dict[0]
    return segmentation_dict


def create_marker_list_xeniumcodex(marker_file: str) -> List[str]:
    """
    Retrieves list of marker names from a given TIFF file in a xenium or codex like format

    Parameters:
    - marker_file = name of TIFF with markers of interest
    Returns:
    - list of markers within the given TIFF file
    """
    # read in marker TIFF
    with tifffile.TiffFile(marker_file) as tif:
        # parse OME-XML metadata
        ome_metadata = tif.ome_metadata
    # parse the XML
    root = ET.fromstring(ome_metadata)
    # define the namespace
    namespace = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
    # retrieve all channel names
    # TODO: !!! change for MIBI, etc.
    channel_names = []
    for image in root.findall('ome:Image', namespace):
        for channel in image.find('ome:Pixels', namespace).findall('ome:Channel', namespace):
            name = channel.get('Name')
            channel_names.append(name)
    return channel_names


def create_marker_list_mibiv1(marker_file: str) -> List[str]:
    """
    Retrieves list of marker names from a given TIFF file in a MIBI like format

    Parameters
    - marker_file = name of TIFF with markers of interest
    Returns:
    - list of markers within the given TIFF file
    """
    # instantiate list of channel names
    channel_names = []
    with tifffile.TiffFile(marker_file) as tif:
        for index, page in enumerate(tif.pages):
            channel_names.append(page.tags['PageName'].value.split(' (', 1)[0])
    return channel_names


def create_marker_list_mibiv2(marker_file: str) -> List[str]:
    """
    Retrieves list of marker names from a given TIFF file in a MIBI like format,
    this is a secondary function that may work instead of the first for certain images

    Parameters
    - marker_file = name of TIFF with markers of interest
    Returns:
    - list of markers within the given TIFF file
    """
    # instantiate list of channel names
    channel_names = []
    with tifffile.TiffFile(marker_file) as tif:
        for index, page in enumerate(tif.pages):
            channel_names.append(page.tags[270].value.split('channel.target": "')[-1].split('"')[0])
    return channel_names


# create a mapping between intensities file type and methods to utilize
FILETYPE2FUNCTION = {'Xenium': [create_marker_list_xeniumcodex],
                     'CODEX': [create_marker_list_xeniumcodex],
                     'MIBI': [create_marker_list_mibiv1, create_marker_list_mibiv2],
                     'AUTO': [create_marker_list_xeniumcodex, create_marker_list_mibiv1, create_marker_list_mibiv2]}
