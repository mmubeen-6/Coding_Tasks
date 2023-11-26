## Task 2

Use PyTorch ecosystem libraries to solve the following

- Find an open dataset for the semantic segmentation task (3+ classes)
- Train DeepLabV3+ model with resnet-32 encoder using this dataset.
- Use W&B to log the training process
- Create a publicly available report in W&B containing examples of ground truth + predicted masks overlaid on the images
- Publish a report with the link to the ipynb file with the solution (github)

When checking, attention will be paid to the metrics used, all the solution details (logic), overall elegance of the solution description (everything has to be described/commented, DRY, formulas are made in LaTex or high resolution images etc.).

## To Run

- Make sure you have login to wandb

```bash
wandb login
```

- Either run the notebook or run the python file

```bash
python train.py
```

- Trained weights are available at [link](). Download and use them for inference using the notebook or python file. For inference using python file, make sure you have the set correct parameters in the file

```bash
python inference.py
```

- If you training using the notebook, make sure to update paths in the corresponding configs cell. For the script `train.py`, use the `config.py` file to update the paths/params.
