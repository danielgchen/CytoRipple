# CytoRipple
## Summary
<p>
    <img src="static/logo.png" alt="CytoRipple logo of blue cell causing ripples" width="90" align="left"/>
    CytoRipple computes cellular vectors based on subcellular
    <br>
    abundances of molecules of interest (e.g. tubulin protein).
    <br>
    See our mauscript at <a href="google.com">XXX</a> for more information.
</p>

## Quick Usage
### Jupyter Notebook
Please see the example notebook in `ExampleNotebookforMIBI.ipynb` for a walk through of an interactive implentation of CytoRipple. Running CytoRipple in a notebook allows for greater customization than via the command line interface. We also provide an example notebook in `ExampleNotebookforXenium.ipynb` that provides a walk through for the calculation of "streams" (cells with similar polarization in tissue).

### Command Line Interface
For users with ready segmentation and intensity files, the command line interface (CLI) offers a quick avenue for vector field computation. Simply replace the following command with your own files of interest `python cytoripple_cli.py -seg <SEGMENTATION_FILE> -int <INTENSITIES_FILE> -moi <MARKER_OF_INTEREST> -out <OUTPUT_DIRECTORY>`.

## Contact Us
Daniel G. Chen (DGChen@mednet.ucla.edu)