# Run results

The structure of the folder is:
- `images/`: superpixels segmentation of the test dataset (Berkeley Segmentation Dataset)
- `loss-train/`: loss evaluated for each batch and each epoch of one run on the training dataset
- `loss-validation/`: loss evaluated for each epoch on the validation dataset (final weights are those whose epoch minimize the validation loss)
- `weights/`: network weights for each batch and each epoch of one run on the training dataset
