<h2><img align="center" src="img/RSS_logo.png"> Restorable segmentation synthesis (RSS) tool</h2>

## Description
The Restorable Segmentation Synthesis (RSS) Tool is software in **Python** that generates synthetic segmentation contours with user-specified parameters to control segmentation errors on pre-defined truth contours. The RSS tool offers a graphical user interface (GUI) and command-line functions that can be inserted into a user’s own code. The GUI allows for visualization of the synthesis segmentation, interactive tuning of the parameters, and display of several segmentation evaluation results. Both GUI and command-line functions allow batch processing.

The RSS tool provides **image restorable segmentation synthesis function**. This function was designed such that the average across the synthetic contours asymptotically converges to the original truth contour. This allows evaluation of truthing methods by simulation of multiple observers’ segmentations that can be fused with a truthing method to define a reference standard (truth). More importantly, the RSS tool enables the creation of benchmark datasets to compare different truthing methods and can also be used for data augmentation in training AI for medical imaging.

## Intended Purpose

The RSS tool supports multiple activities by end users and AI developers including:
 - Investigating properties of segmentation performance metrics and informing segmentation metric selection.
 - Investigating truthing methods and informing truthing method selection by allowing users to assess the impact of different augmentation methods for combining multiple segmentation (truth) masked provided by a set of truthers.
 - Augmenting segmentation masks for improving training of AI segmentation models.

The intended users of this RSS tool include AI segmentation algorithm developers and reviewers.  The clinical use cases include AI-based segmentation applied to Digital Pathology and Radiology image datasets.

## Installation
This section will help you to install the packages needed for `RSS-tool`.


### Pre-requirements
`RSS-tool` was developed and tested in the [Environment](environment.txt) of Python 3.8.
And it depends on the core packages listed in the [Requirements](requirements.txt):
```
numpy
opencv-python
scipy
scikit-image
scikit-learn
pillow
simpleitk~=2.1.1
matplotlib~=3.4.3
labelfusion~=1.0.14
tqdm
```

### Preparation

* Download the whole repository from its GitHub and put all files as their original structure in a folder (named "RSS-tool").
```
https://github.com/didsr/RSS-tool
```

* Use GUI
  - Activate the installed Python environment.
  - Under the folder `RSS-tool`, run `RSS_GUI_main.py` file in Python: `python RSS_GUI_main.py`.
  
    <img src="img/start_gui.png" width="300"/>
  
* Use Command-line Functions
  - All main functions are in the files `IPfunctions.py`, `LabelFusion.py`, and `seg_measures.py`.
  - Use `from <FileName> import <FunctionName>` in Python to include the functions into your codes.
  - Details of available functions can be found in the "The Use of Codes (Core Functions)" chapter in the **User's Manual: [Link](https://github.com/DIDSR/RSS-tool/blob/main/Restorable%20Segmentation%20Synthesis%20(RSS)%20Tool%20User%20Guide.pdf)**.

## User's Manual: [Link](https://didsr.github.io/RSS-tool/Restorable%20Segmentation%20Synthesis%20(RSS)%20Tool%20User%20Guide_PUB.html)*

## Cite this repository

If you find that `RSS-tool` is useful or if you use it in your project, please cite this code and the paper:


```
https://github.com/DIDSR/RSS-tool
```

```
The paper to be published.
```

### Contact
For any questions/suggestions/collaborations regarding this tool, please contact Shuyue Guan (shuyue.guan@fda.hhs.gov) or Weijie Chen (weijie.chen@fda.hhs.gov).

## Acknowledgment 
* This project was supported in part by an appointment to the ORISE Research Participation Program at the Center for Devices and Radiological Health, U.S. Food and Drug Administration, administered by the Oak Ridge Institute for Science and Education through an interagency agreement between the U.S. Department of Energy and FDA/CDRH.
