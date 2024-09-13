# CCLya-Payne: Neural Network Emulation of Lyman-alpha Profiles
[![DOI](https://zenodo.org/badge/624616131.svg)](https://zenodo.org/doi/10.5281/zenodo.13755691)

**CCLya-Payne** (**C**ode for **C**ontinuous **Ly**man-**a**lpha **P**rofile **A**nal**Y**sis via **N**eural **E**mulation) is a neural network designed to emulate Lyman-alpha (Lyα) profiles from radiative transfer simulations. The network is trained on synthetic spectra convolved with the instrumental point spread function (PSF) of the Magellan MIKE echelle spectrograph (10 km/s resolution). With this tool, you can quickly generate Lyα profiles based on a set of input parameters that describe intrinsic dynamics and physical conditions of the source.

### Installation
```bash
cd ~
git clone git@github.com/highzclouds/CCLya-Payne.git
cd CCLya-Payne/cclya_payne
sudo python3 setup.py install 
```

### Features
- Generate Lyα profiles quickly for a given set of input parameters across continuous parameter spaces.
- Fit Lyα models to observed spectra.

**Input Parameters**
- **Intrinsic dynamics** (σ<sub>int</sub>): sets the intrinsic velocity dispersion of the Lya photons from the emitting source [as sigma_int/(km * s<sup>-1</sup>)]. The photons are sampled from a Gaussian with this width.
- **Neutral hydrogen column density** (logN<sub>HI</sub>): sets the overall column density of the shell model [as log(N_HI / cm<sup>-2</sup>)].
- **Expansion velocity** (v<sub>exp</sub>): sets the inflow/outflow velocity of the shell [as v<sub>exp</sub>/(km * s<sup>-1</sup>)]. This is uniform throughout the shell.
- **Effective temperature** (logT): sets the temperature for the combined thermal and bulk motion of the hydrogen [as log(T / K)].

See Solhaug et al. 2024 for the range of the model grid.

The **column density** and **temperature** parameters are provided in logarithmic form. The output Lyα profiles reflect the physical conditions of gas in outflowing systems.

### Using Different Versions of the Neural Network

There are two versions of the neural network available:

1. **Convolved Spectra Network**: This network is trained on synthetic Lyα spectra convolved with a high-resolution Gaussian line spread function (LSF) of the Magellan MIKE echelle spectrograph (FWHM = 10 km/s). The corresponding files are:
   - Neural Network: `cclya_neuralnetwork.npz`
   - Training Spectra: `cclya_training_spectra.npz`
   - Validation Spectra: `cclya_validation_spectra.npz`
   - Training Loss: `cclya_training_loss.npz`

   Using this version will yield Lyα profiles as observed through the high-resolution MIKE spectrograph. To use the network with other instruments, you can either modify the code to convolve the network outputs with the Gaussian kernel corresponding to your instrument's LSF or use the raw spectra version and apply the necessary convolution.

2. **Raw Spectra Network**: This network is trained on raw (not convolved) synthetic Lyα spectra. The corresponding files are:
   - Neural Network: `cclya_neuralnetwork_raw.npz`
   - Training Spectra: `cclya_training_spectra_raw.npz`
   - Validation Spectra: `cclya_validation_spectra_raw.npz`
   - Training Loss: `cclya_training_loss_raw.npz`

   To adapt this network for different instruments, you need to convolve the network outputs with a Gaussian kernel matching your instrument's LSF. Alternatively, you can modify the `get_neural_network` module to load this raw-trained network and then apply the convolution as needed.


### Training Data
The neural network is trained on simulated Lyα spectra produced by the **TLAC** radiative transfer code (Gronke et al. 2015). These spectra are convolved with the PSF of the Magellan MIKE instrument to match the resolution of high-resolution spectroscopic data (10 km/s). If you wish to apply the neural network to other instruments, you may want to use the training.py package to train the neural network on the training data convolved to match the PSF of the instrument you're using, or change this code to convolve each neural network-produced model to match the PSF of your instrument.

**Note:** Due to GitHub's file size limitations, the training and validation spectra used for the neural network are hosted externally. To download these data files, please refer to the [README_DATA.md](./README_DATA.md) file for the download link and instructions.

### Repository Structure
This repository contains several modules and files that support the neural network and fitting processes:

- **`utils.py`**: Helper functions for data processing and analysis.
- **`fitting.py`**: Functions for fitting Lyα profiles to observed spectra.
- **`training.py`**: Scripts and methods for training the neural network on synthetic spectra.
- **`radam.py`**: An implementation of the Rectified Adam optimizer used in the training process.

### Getting Started
To get started with fitting Lyα profiles to observed spectra:
1. CCLya-Payne works with Lyα profiles in velocity-space. You will need to convert your observed spectra to velocity-space before fitting.
2. Follow the steps outlined in the **`tutorial.ipynb`** Jupyter notebook for a quick guide on how to run the code and perform fits. A sample spectrum and fitting is included.
3. Modify the **`guess.txt`** and **`priors.txt`** files to set your initial parameter guesses and parameter priors.
4. Fitting will produce a few files in the ./mcmc directory. Inspect these file sto make sure your fitting is working as expected.

### Citations and references
If you use any of these tools in published work, please cite the following papers.
- Ting et al. (2019, [ApJ 879, 69](https://ui.adsabs.harvard.edu/abs/2019ApJ...879...69T/abstract)), which originally fit stellar atmosphere spectra using a neural network approach.  In particular, the **fitting** and **radam** module, along with certain functions in the **utils** module (read_in_neural_network, leaky_relu, get_spectrum_from_neural_net, load_training_data, get_loss), are adapted from **[The Payne](https://github.com/tingyuansen/The_Payne)** developed by these authors.  The modifications in this repository have been tailored to fit Lyα spectra;
- Gronke and Dijkstra (2014, [MNRAS 444, 1095](https://ui.adsabs.harvard.edu/abs/2014MNRAS.444.1095G/abstract)).  The synthetic spectra used for training the neural network were generated using the **TLAC** radiative transfer code by these authors; and
- Solhaug et al. (2024), submitted to the Open Journal of Astrophysics, which produced all the tools that combine the TLAC radiative transfer calculations with the machine learning framework to facilitate Lya profile analyses.

### Authors
- [Erik Solhaug](https://astrophysics.uchicago.edu/people/profile/erik-solhaug/) (The University of Chicago) --- eriksolhaug at uchicago dot edu
- [Hsiao-Wen Chen](https://astrophysics.uchicago.edu/people/profile/hsiao-wen-chen/) (The University of Chicago)
