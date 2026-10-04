# Imports
# > Private Packages
from image_io import *
from calculations import *
from consts import *
from plotting import *
# > Public Packages
import argparse
import logging
import os
import torch

# Instantiate Argument Parser
parser = argparse.ArgumentParser(description='Cytoripple: Computing protein directionality from image proteomic data.')
# > Add required arguments
parser.add_argument('-seg', '--segmentation_file', type=str, help='Path to the segmentation TIFF file.', required=True)
parser.add_argument('-int', '--intensities_file', type=str, help='Path to the intensities TIFF file.', required=True)
parser.add_argument('-moi', '--marker_of_interest', type=str, help='The name of the marker to analyze.', required=True)
parser.add_argument('-out', '--output_directory', type=str, help='Path to directory to output results in.', required=True)
# > Add optional arguments for overall operations
parser.add_argument('--intensities_file_type', type=str, help='File type of the intensities TIFF, required=False (\'MIBI\' tries MIBI methods, \'Xenium\' and \'CODEX\' attempts the Xenium/CODEX method, \'AUTO\' tries all methods).', default='AUTO')
parser.add_argument('--logging_file', type=str, help='Path to the file to output logging information.', default='cytoripple.log', required=False)
parser.add_argument('--device', type=str, help='Device to utilize (e.g. \'cpu\' or \'cuda\'), by default will attempt to utilize GPU if available.', default='AUTO', required=False)
# > Add optional arguments for smoothing
parser.add_argument('--min_magnitude', type=float, help='Minimum unscaled vector magnitude required to be considered as a vector for smoothing and scaling.', default=MIN_VECTOR_MAGNITUDE, required=False)
parser.add_argument('--smooth_iters', type=int, help='Number of spatial smoothing iterations to perform on vectors.', default=SMOOTH_ITERS, required=False)
parser.add_argument('--smooth_dist_limit_pixel', type=float, help='Only cells within this radius are considered for vector smoothing. Please compute this based on your image\'s micron to pixels and maximum radius in microns (e.g. 0.2125µm/px for Xenium/CODEX and 0.390625µm/px for MIBI).', default=SMOOTH_DIST_LIMIT_PIXEL, required=False)
# > Parse arguments
args = parser.parse_args()

# Instantiate Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# > Save the log to a file of interest
file_handler = logging.FileHandler(args.logging_file)
file_handler.setLevel(logging.INFO)
# > Create a logging format for the messages
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
# Add the handler to the logger
logger.addHandler(file_handler)

# Define Main Pipeline
def main():
    # Preamble to the logger
    logger.info('>>> Starting Cytoripple Analysis')
    logger.info('> Proceeding with the following inputted files...')
    logger.info(f'Segmentation File: {args.segmentation_file}')
    logger.info(f'Intensities File: {args.intensities_file}')
    logger.info(f'Marker of Interest: {args.marker_of_interest} <')

    # Read in the segmentation and marker intensities file
    logger.info('> Attempting to load data from files...')
    segmentation_imstack, marker_imstack = load_data(args.segmentation_file, args.intensities_file)
    segmentation_dict = create_segmentation_dict(segmentation_imstack)
    logger.info('Data loaded successfully from files <')

    # Retrive markers of interest and identify relevant image
    logger.info('> Attempting to retrieve marker list from intensities TIFF...')
    # > Attempt to retrieve marker lists from TIFF
    functions = FILETYPE2FUNCTION[args.intensities_file_type]
    function_index = 0
    found_marker_list = False
    while (not found_marker_list) and (function_index < len(functions)):
        # > Retrieve function and attempt to utilize it
        function = functions[function_index]
        try:
            marker_list = function(args.intensities_file)
            found_marker_list = True
        except:
            pass
        # > Continue on if failed then proceed to next function or exit
        function_index += 1
    # > Exit if marker list cannot be found
    if not found_marker_list:
        logger.error('No marker list could be retrieved, please edit the repository to add a custom function <')
        return
    # > If successful continue onwards and report marker list
    marker_list_str = ', '.join(marker_list)
    logger.info(f'Marker list was successfully retrieved see: \'{marker_list_str}\' <')
    
    # Attempt to find marker of intrest
    logger.info('> Attempting to find marker of interest from marker list...')
    try:
        image_stack_index = marker_list.index(args.marker_of_interest)
        logger.info(f'Found \'{args.marker_of_interest}\' at index {image_stack_index} in the marker list <')
    except ValueError:
        logger.error(f'Marker \'{args.marker_of_interest}\' not found in the provided intensities file <')
        return

    # Retrieve intensities for the marker of interest
    logger.info('> Moving relevant intensities to accelerator...')
    intensities = marker_imstack[0][image_stack_index]
    # > Set device as needed, based on user input
    if args.device == 'AUTO':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        try:
            device = torch.device(args.device)
        except ValueError:
            logger.error(f'Provided device is not valid <')
            return
    intensities_acc = torch.tensor(intensities, device=device, dtype=torch.float32)
    logger.info('Successfully moved intensities to accelerator device <')

    # Generate vectors
    logger.info('> Generating vector field...')
    vectors = generate_vector_field(segmentation_dict, intensities_acc, device)
    logger.info('Successfully computed raw vector field <')

    # Smooth and scale vectors
    logger.info('> Smoothing vector field...')
    smoothed_vectors = smooth_vector_field(
        vectors,
        min_magnitude=args.min_magnitude,
        smooth_iters=args.smooth_iters,
        smooth_dist_limit_pixel=args.smooth_dist_limit_pixel
    )
    logger.info('Succesfully smoothed and scaled vector field <')

    # Calculate curl and divergence metrics
    logger.info('> Calculating curl and divergence...')
    XI, YI, curl, divergence = calculate_curl_and_divergence(smoothed_vectors)
    metrics = pd.DataFrame([XI.flatten(), YI.flatten(), curl.flatten(), divergence.flatten()],
                           index=['x_index','y_index','curl','divergence']).T
    logger.info('Sucessfully calculated curl and divergence metrics <')

    # Write metrics
    logger.info('> Attempting to write results to output directory...')
    # > Create directory if needed
    os.makedirs(args.output_directory, exist_ok=True)
    # > Write files
    vectors.to_csv(f'{args.output_directory}/raw_unscaled_vectors.csv', index=False)
    smoothed_vectors.to_csv(f'{args.output_directory}/smoothed_scaled_vectors.csv', index=False)
    metrics.to_csv(f'{args.output_directory}/curl_divergence_metrics.csv', index=False)
    logger.info('Successfully wrote results to outut directory <')

    # Finish and exit
    logger.info('Cytoripple Analysis Complete <<<')


if __name__ == '__main__':
    main()