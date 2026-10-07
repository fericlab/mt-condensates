# mt-condensates
This is the source data accompanying the paper:

"Phase coexistence of mitochondrial nucleoids and RNA granules in the transfer of nascent RNA"

Image processing, quantification, plotting, data visualization scripts; image files and raw data are organized by figure in 'Main and Supplementary Figures'. 'Supplementary Videos' contains video files.'mt-condensate_scripts' contains the main scripts used for analysis in Fig 1, Fig 2, Fig 3 and Fig 5.

Contact: mjf6624@psu.edu

## System requirements

### Hardware requirements
  - Standard computer with enough RAM to run the software
  - No non-standard hardware is required

### The software is compatible and has been tested with:
  - Windows (11, 10)
  - macOS (Ventura 13.3.1)

### Python requirements (scripts work with both versions)
  - Python (3.12.7), updated Python (3.14.6)
  - Anaconda Distribution
  - Trackpy (0.6.4), updated Trackpy (0.7)
  - czifile (2019.7.2.1), updated czifile (2026.6.12)
  - OpenCV-Python (4.13.0.90)

## Installation guides

1. Python(< 5 minutes): https://wiki.python.org/moin/BeginnersGuide/Download  
2. Anaconda Distribution (~ 15 minutes): https://www.anaconda.com/docs/getting-started/anaconda/install  
3. Trackpy (< 5 minutes): https://soft-matter.github.io/trackpy/v0.7/installation.html  
4. Czifile (< 5 minutes):  
   - Open Anaconda Navigator  
   - Open Anaconda Command Shell  
   - type "pip install czifile" into the terminal and hit enter  
5. OpenCV-Python (< 5 minutes): https://pypi.org/project/opencv-python/#installation-and-usage

### Distance and Nearest Neighbor Pipeline (< 5 minutes):
  - Download the .py files in the folders titled "Fig 1_Fig2_Fig5 and Fig 3".
  - Download the .py files ('CrossPlotwithScale', 'SelfPlotwithScale') in the folder titled "Parikh_Fig1_Suppl-Fig2_Fig3"
  - Download the .py files ('20260816_EU-plotAndFitRunningScript-NP-pdfeditable') in the folder titled "Parikh_Fig2_Suppl-Fig4"
  - Download the .py files ('CrossPlot_IMT1B_0vs15vs30min', 'CrossPlot_IMT1B_0vs30min') in the folder titled "Parikh_Fig5_Suppl-Fig7"
  - Place all .py files in one folder (for the respective figure)
  - Create a new folder inside the overall folder titled "testimage"
  - Download demo image in .czi format, or place an experimentally derived .czi image in the "testimage" folder

## Demo: Running the Distance and Nearest Neighbor pipeline 
### Instructions (Fig 1, 2, 5)
  - Open the file titled "distNNRunningScript20Apr25_d_5um_20251119" in an IDE
  - In the variable titled "directory" modify the statement of the variable to reflect the file path of "testimage" folder (the statement should look something like {r'C:\...\testimage'} for windows or {r'/.../testimage'} for macOS)
  - In the variable titled "zHeight" modify the statement to reflect the distance between z-slices in microns (for our demo image, the distance between subsequent z-slices is 0.130um or 130nm)
  - Execute the file

### Expected output

The following .csv files will be written to the folder containing the .py files
  - "3ChannelImages.csv": complete table of distances between nearest neighbor pairs of 3ChannelImages
  - "3ChannelImages_upto5um.csv": table of distances between nearest neighbor pairs of 3ChannelImages for small distance pairs (<5um)
  - “distNN_upto5um_long.csv”: table of distances between nearest neighbor pairs of 3ChannelImages for small distance pairs (<5um)along with source and target channel intensities and punctum major axis size for each punctum
  - “3ChannelImages_upto5um_HighInt_1v3_1v4.csv” and “3ChannelImages_upto5um_LowInt_1v3_1v4.csv”: table of 1vs3 and 1vs4 distances filtered for punctum with >1000 or <1000 intensity values
  -“ 3ChannelSubset.csv”: table of distances between nearest neighbor pairs of 3ChannelImages after size and intensity filtering.

The following plots will be displayed
  - For all channels, distances to the nearest puncta within the same channel
  Ex. mtRNA_channel.pdf(.eps): Distances to Nearest Puncta within the mtRNA Channel
  - “Size_vs_Intensity_{channel_label}.pdf” (.eps) : plots of size vs intensity of each punctum within the same channel
  - “1_vs_4_ch1int_lt1000_dle5um.pdf” and “1_vs_4_ch1int_ge1000_dle5um.pdf” (.eps) : plots with 1vs4 distances filtered for <1000 or >1000 intensity values in channel 1

## Demo: Running the plotting script(s):

### General Instructions:
  - For a given script file, replace any references to an input .csv file with the desired input file. Plots in figures were made using data from “3ChannelImages_upto5um.csv”.

### General Output:
  - Plots that correspond to the indicated figure

### Example Instructions for Fig. 2: 
  - For the script file "plot-stats-correlation_NP_20251218.py" replace the argument of the "read.csv" function defining the variable "df" with the .csv file "3ChannelImagesupto5um.csv" for the corresponding datasets

### Example expected output
  - Correlation bar plot comparing each channel

### Expected Runtime: (~10 minutes)

## Instructions for use

See above demo instructions

**-Follow similar instructions for Fig 3 by downloading file 'AverageIntensityRunningScript-NP-x_centered_distances-nm' for complete averaging profiles and 'SubpopulationAveragingByFile' for sub-population averaging profiles (mixed, wetting and demixed categories) and executing it.Expected outputs include 2D heatmaps for averaged intensities for 1vs(1,2,3,4) and so on for all 4 channels. These intensities will also be saved as 2D arrays in .csv files. Subsequent plotting scripts, instructions and raw data can be found under folder "Parikh_Fig1_Extended_Fig3".**


