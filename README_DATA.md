# CCLya-Payne: Neural Network Emulation of Lyman-alpha Profiles - Data Information

This repository includes all the essential code for running **CCLya-Payne**, but due to GitHub's file size limitations, the training and validation data used for the neural network are not included here.

## Data Files

To download the data (specifically, the training and validation spectra), please visit the following URL:

[**INSERT URL**]

The files available at this URL include:
- `cclyapayne_training_spectra.npz`: Training spectra for the convolved network.
- `cclyapayne_validation_spectra.npz`: Validation spectra for the convolved network.
- `cclyapayne_training_spectra_raw.npz`: Training spectra for the raw network (unconvolved).
- `cclyapayne_validation_spectra_raw.npz`: Validation spectra for the raw network (unconvolved).

These spectra were generated using the **TLAC** radiative transfer code and are essential for training and evaluating the neural network in **CCLya-Payne**.

## How to Use

1. Download the appropriate `.npz` files from the link above.
2. Place the downloaded files into the `./data/` directory within this repository.
3. Follow the instructions in the main README to set up and run the neural network.
