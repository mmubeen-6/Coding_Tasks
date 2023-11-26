# Coding Tasks

## Task 1

There’s a point in a 3D space with coords [1, 2, 3]. We rotate the coordinate space around the Y axis for 45\* and translate the origin by vector [3, 2, 1].

Find the point coordinates in the new coordinate system.

## Task 2

Use PyTorch ecosystem libraries to solve the following

- Find an open dataset for the semantic segmentation task (3+ classes)
- Train DeepLabV3+ model with resnet-32 encoder using this dataset.
- Use W&B to log the training process
- Create a publicly available report in W&B containing examples of ground truth + predicted masks overlaid on the images
- Publish a report with the link to the ipynb file with the solution (github)

### Setup

```bash
conda env create -f environment.yml
conda activate yolov8
```
