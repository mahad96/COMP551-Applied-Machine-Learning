# Project Title

Mini-Project 3 for COMP551 - Applied Machine Learning, Winter 2019, McGill. 

### Authors
Benjamin MacLellan

John McGowan

Mahad Khan

## Prerequisites

All packages used in this project are common Python distributions, which can be installed via `pip`.

```
pytorch
numpy
pandas
matplotlib.pyplot
cv2
```

Note, that although `cv2` was used to test some preprocessing techniques, it did not improve the model - thus it is not used in the final model(s). The script can be found in `preprocess.py`.

Trained models saved for future analysis are saved in `trained_models/`, submissions to Kaggle are stored in `submissions/`, and `input/` contains (on our local machines) the dataset files.

## Training and testing the models

In `main.py`, change the `train_images_path`, `train_labels_path`, and `test_images_path` to match where the training and test image datasets, and training labels are (or place into `inputs/`, such that it matches the current path directory). 

To run the CNN, use `main.py` -- which contains the main script and training procedure. Models are stored in `models.py`, the a custom Pytorch dataset class is defined in `dataset_loaders.py`. 

The models included in our writeup are Net4 (based off vgg net), Net5 (Net4 + skip connections), and Net6 (Using inception layers).
