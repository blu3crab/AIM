# **AIM - Artifact Interest Meter**

Please view this article for high-level usage patterns:

https://matucker.medium.com/artifact-interest-meter-whats-the-story-40836b0af0a8

The **ipynb** directory contains jupyter notebooks exercising the lower-level utility methods in the **py** directory.  There are many experimental notebooks as well.  The utility **py** files are intended to be imported from the selected Google drive.

The key files for each step in the pipeline are:

**aim_xformer.ipynb** exercises **aim_xformer.py** to generate a variety of image transformations.

**aim_scanner.ipynb** exercises **aim_scanner.py** to predict words and bounding boxes in the source image.

**aim_rater.ipynb** exercises **aim_rater.py** to determine percentages of exact matches and partial matches between the various image transformation predictions and the ground truth.

**aim_overlayer.ipynb** exercise **aim_overlayer.py** to overlay predictions and their associated bounding box on the source image.

**aim_pipeline1.ipynb** is an example of processing a source image through each of the steps in the pipeline.
