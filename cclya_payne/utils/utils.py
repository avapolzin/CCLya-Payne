# a few low-level functions that are used throughout
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import emcee
import corner
import pickle
import time
import os
import pkg_resources
DATA_PATH = pkg_resources.resource_filename('cclya_payne', 'data/')

def read_in_neural_network():
    '''
    Read in the weights and biases parameterizing the neural network.

    This function loads a neural network's parameters, including weights and biases, from a .npz file located in the 'data/' directory.
    You can use this function to load pre-trained networks or to read in networks that you have trained yourself by modifying this function as needed.

    Returns:
    - NN_coeffs (tuple): A tuple containing:
        - w_array_0, w_array_1, w_array_2, w_array_3: Weights for the neural network's layers.
        - b_array_0, b_array_1, b_array_2, b_array_3: Biases for the neural network's layers.
        - x_min, x_max: Normalization parameters for input features.
    '''

    nn_fname = pkg_resources.resource_filename('cclya_payne', 'data/'+'cclya_neuralnetwork.npz') # os.path.join(os.path.dirname(os.path.realpath(__file__)),'data/cclya_neuralnetwork.npz')

    tmp = np.load(nn_fname)

    w_array_0 = tmp["w_array_0"]
    w_array_1 = tmp["w_array_1"]
    w_array_2 = tmp["w_array_2"]
    w_array_3 = tmp["w_array_3"]
    b_array_0 = tmp["b_array_0"]
    b_array_1 = tmp["b_array_1"]
    b_array_2 = tmp["b_array_2"]
    b_array_3 = tmp["b_array_3"]
    x_min = tmp["x_min"]
    x_max = tmp["x_max"]
    NN_coeffs = (w_array_0, w_array_1, w_array_2, w_array_3, b_array_0, b_array_1, b_array_2, b_array_3, x_min, x_max)
    tmp.close()

    return NN_coeffs

def leaky_relu(z):
    '''
    Leaky ReLU activation function.

    This function applies the Leaky ReLU activation function, which allows a small, non-zero gradient when the input is negative, preventing the "dying ReLU" problem.

    Parameters:
    - z (numpy.ndarray): Input array to which the activation function is applied.

    Returns:
    - numpy.ndarray: Output array after applying the Leaky ReLU activation function.
    '''
    return z*(z > 0) + 0.01*z*(z < 0)

def sigmoid(z):
    '''
    Sigmoid activation function.

    This function applies the sigmoid activation function, which maps input values to a range between 0 and 1.

    Parameters:
    - z (numpy.ndarray): Input array to which the activation function is applied.

    Returns:
    - numpy.ndarray: Output array after applying the sigmoid activation function.
    '''
    return 1 / (1 + np.exp(-z))

def get_spectrum_from_neural_net(scaled_labels, NN_coeffs):
    '''
    Emulate the spectrum of a shell model Lyman-alpha profile using a neural network.

    Parameters:
    - scaled_labels (numpy.ndarray): Scaled input parameters for the neural network, representing the physical conditions of the shell model.
    - NN_coeffs (tuple): Coefficients for the neural network including weights and biases for each layer. The tuple contains:
        - w_array_0, w_array_1, w_array_2, w_array_3: Weights for the network's layers.
        - b_array_0, b_array_1, b_array_2, b_array_3: Biases for the network's layers.

    Returns:
    - spectrum (numpy.ndarray): The emulated Lyman-alpha spectrum output by the neural network.
    '''

    # The NN has three hidden layers and an output layer.
    w_array_0, w_array_1, w_array_2, w_array_3, b_array_0, b_array_1, b_array_2, b_array_3, _, _ = NN_coeffs
    inside = np.einsum('ij,j->i', w_array_0, scaled_labels) + b_array_0
    outside = np.einsum('ij,j->i', w_array_1, leaky_relu(inside)) + b_array_1
    mid = np.einsum('ij,j->i', w_array_2, leaky_relu(outside)) + b_array_2
    spectrum = np.einsum('ij,j->i', w_array_3, sigmoid(mid)) + b_array_3

    return spectrum

def load_wavelength_array():
    '''
    Read in the default wavelength grid for the neural network.

    This function loads the wavelength array used by the neural network from a .npz file located in the 'data/' directory.

    Returns:
    - wavelength (numpy.ndarray): Array of wavelengths corresponding to the spectral data used by the neural network.
    '''
    
    wav_fname =  pkg_resources.resource_filename('cclya_payne', 'data/'+'cclya_neuralnetwork.npz') # os.path.join(os.path.dirname(os.path.realpath(__file__)),'data/cclya_wavelength.npz')  #os.path.join(os.path.dirname(os.path.realpath(__file__)),'data/cclya_wavelength.npz')
        
    tmp = np.load(wav_fname)
    wavelength = tmp['wavelength']
    tmp.close()
    
    return wavelength

def load_training_data():
    '''
    Read in the training and validation spectra produced with TLAC.

    This function loads the training and validation spectra used to train the default neural network.
    - Training spectra: 75% of the total spectra.
    - Validation spectra: 25% of the total spectra.

    The spectra and their corresponding labels are read from .npz files located in the 'data/' directory.

    Returns:
    - training_labels (numpy.ndarray): Labels for the training spectra.
    - training_spectra (numpy.ndarray): Training spectra.
    - validation_labels (numpy.ndarray): Labels for the validation spectra.
    - validation_spectra (numpy.ndarray): Validation spectra.
    '''
    
    training_fname =  pkg_resources.resource_filename('cclya_payne', 'data/'+'cclya_neuralnetwork.npz') # os.path.join(os.path.dirname(os.path.realpath(__file__)),'data/cclya_training_spectra.npz')  #os.path.join(os.path.dirname(os.path.realpath(__file__)),'data/cclya_training_spectra.npz')
    validation_fname =  pkg_resources.resource_filename('cclya_payne', 'data/'+'cclya_neuralnetwork.npz') # os.path.join(os.path.dirname(os.path.realpath(__file__)),'data/cclya_validation_spectra.npz')  #os.path.join(os.path.dirname(os.path.realpath(__file__)),'data/cclya_validation_spectra.npz')
    
    # Load training spectra
    tmp = np.load(training_fname)
    training_labels = (tmp["labels"].T)
    training_spectra = tmp["spectra"]
    tmp.close()
    # Load validation spectra
    tmp = np.load(validation_fname)
    validation_labels = (tmp["labels"].T)
    validation_spectra = tmp["spectra"]
    tmp.close()

    return training_labels, training_spectra, validation_labels, validation_spectra

def get_loss():
    '''
    Read in the training and validation loss from file.

    This function loads the training and validation loss values from a .npz file stored in the 'data/' directory.

    Returns:
    - training_loss (numpy.ndarray): Array of training loss values per 100 training step.
    - validation_loss (numpy.ndarray): Array of validation loss values per 100 training step.
    '''

    loss_fname = pkg_resources.resource_filename('cclya_payne', 'data/'+'cclya_neuralnetwork.npz') # os.path.join(os.path.dirname(os.path.realpath(__file__)),'data/cclya_training_loss.npz') # os.path.join(os.path.dirname(os.path.realpath(__file__)),'data/cclya_training_loss.npz')
    
    tmp = np.load(loss_fname) # the output array also stores the training and validation loss
    
    print(f'Read training and validation loss of {loss_fname}.')
    
    training_loss = tmp["training_loss"]
    validation_loss = tmp["validation_loss"]
    tmp.close()
    
    return training_loss, validation_loss

def get_model(sigma_int, logN_HI, v_exp, logT):
    '''
    Generate a Lyman-alpha spectrum for a given set of input parameters using the neural network.

    Parameters:
    - sigma_int (float): Intrinsic width of the Lyman-alpha emission line.
    - logN_HI (float): Logarithm of the neutral hydrogen column density.
    - v_exp (float): Expansion velocity of the shell.
    - logT (float): Logarithm of the temperature.

    Returns:
    - real_spec (numpy.ndarray): Emulated Lyman-alpha spectrum generated by the neural network.
    '''

    NN_coeffs = read_in_neural_network()
    _, _, _, _, _, _, _, _, x_min, x_max = NN_coeffs
    
    real_labels = scaled_labels = [sigma_int, logN_HI, v_exp, logT]

    scaled_labels = (real_labels-x_min)/(x_max-x_min) - 0.5

    real_spec = get_spectrum_from_neural_net(scaled_labels = scaled_labels, NN_coeffs = NN_coeffs)

    return real_spec


def get_model_interp(v_lya, wav_arr, sigma_int, logN_HI, v_exp, logT):
    '''
    Get the interpolated Lyman-alpha profile from the neural network for the given wavelength array.

    This function calculates the Lyman-alpha profile using the neural network model, incorporates a velocity shift 
    (v_lya) relative to the rest-frame Lyman-alpha velocity, and interpolates the model over the specified 
    wavelength array (wav_arr).

    Parameters:
    - v_lya (float): Velocity shift to be applied to the Lyman-alpha profile.
    - wav_arr (numpy.ndarray): Array of wavelengths at which to interpolate the model.
    - sigma_int (float): Intrinsic width of the Lyman-alpha profile.
    - logN_HI (float): Logarithm of the column density of neutral hydrogen.
    - v_exp (float): Expansion velocity of the shell model.
    - logT (float): Logarithm of the temperature of the shell model.

    Returns:
    - interp_model_arr (numpy.ndarray): Interpolated Lyman-alpha profile at the specified wavelengths.
    '''
    wavelength = load_wavelength_array()
    model = get_model(sigma_int, logN_HI, v_exp, logT)

    wavelength = wavelength + v_lya
    
    if (v_lya > 0):
        indices = find_indices_larger_than_1400(wavelength)
        wavelength = wavelength[:indices[0]]
        model = model[:indices[0]]
        zeros_array = np.zeros(len(indices))
        model = np.concatenate((zeros_array, model))
        left_wavelengths = np.linspace(-1400, wavelength[0], len(indices))
        wavelength = np.concatenate((left_wavelengths, wavelength))
        if (wavelength[0] != 1400): # Add end point of 1400 to correct the valid range
            wavelength = np.concatenate((wavelength, [1400]))
            model = np.concatenate((model, [0]))
    elif (v_lya < 0):
        indices = find_indices_smaller_than_neg1400(wavelength)
        wavelength = wavelength[indices[-1]+1:]
        model = model[indices[-1]+1:]
        zeros_array = np.zeros(len(indices))
        model = np.concatenate((model, zeros_array))
        right_wavelengths = np.linspace(wavelength[-1], 1400, len(indices))
        wavelength = np.concatenate((wavelength, right_wavelengths))
        if (wavelength[0] != -1400): # Add end point of -1400 to correct the valid range
            wavelength = np.concatenate(([-1400], wavelength))
            model = np.concatenate(([0], model))    
    # Interpolate the model over the wavelength array
    model_interp = interp1d(wavelength, model)
    # Find model values at the wavelength values given by wav_arr
    interp_model_arr = model_interp(wav_arr)
    
    return interp_model_arr

def find_indices_larger_than_1400(array):
    '''
    Find indices of elements in the array that are larger than 1400.

    Parameters:
    - array (numpy.ndarray): Array to search for values larger than 1400.

    Returns:
    - indices (list of int): List of indices where the values in the array are larger than 1400.
    '''

    indices = []
    for i in range(len(array)):
        if array[i] > 1400:
            indices.append(i)
    return indices

def find_indices_smaller_than_neg1400(array):
    '''
    Find indices of elements in the array that are smaller than 1400.

    Parameters:
    - array (numpy.ndarray): Array to search for values smaller than 1400.

    Returns:
    - indices (list of int): List of indices where the values in the array are smaller than 1400.
    '''

    indices = []
    for i in range(len(array)):
        if array[i] < -1400:
            indices.append(i)
    return indices

def plot_lya(lya_x, lya_y, obs_x=None, obs_y=None, obs_err=None, train_x=None, train_y=None, outfile=None, xmin=-800, xmax=800):
    '''
    Plot the Lyman-alpha model profile and optionally overlay observed spectra and training spectra.

    Parameters:
    - lya_x (numpy.ndarray): Wavelength values for the Lyman-alpha model.
    - lya_y (numpy.ndarray): Flux values for the Lyman-alpha model.
    - obs_x (numpy.ndarray, optional): Wavelength values for observed spectra.
    - obs_y (numpy.ndarray, optional): Flux values for observed spectra.
    - obs_err (numpy.ndarray, optional): Error values for observed spectra.
    - train_x (numpy.ndarray, optional): Wavelength values for training spectra.
    - train_y (numpy.ndarray, optional): Flux values for training spectra.
    - outfile (str, optional): File path to save the plot as a PDF. If not provided, the plot is shown but not saved.
    - xmin (float, optional): Minimum x-axis value for the plot. Default is -800.
    - xmax (float, optional): Maximum x-axis value for the plot. Default is 800.

    Returns:
    - fig (matplotlib.figure.Figure): The created figure object.
    - ax (matplotlib.axes.Axes): The axes object for the plot.
    '''

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(lya_x, lya_y, label='Lya model', color='#648fff', linewidth=2.5, alpha=0.85)
    if (obs_x is not None) and (obs_y is not None) and (obs_err is not None):
        ax.step(obs_x, obs_y, label='obs. spec', color='k', linewidth=3)
        ax.step(obs_x, obs_err, label='obs. error', color='#dc267f', linewidth=1.5, linestyle='--')
    if (train_x is not None) and (train_y is not None):
        ax.plot(train_x, train_y, label='training spec.', color='orange', linewidth=3)
    ax.set_xlim(xmin, xmax)
    ax.legend(prop={'size': 20}, loc='upper right')

    # Formatting
    # Add minor ticks every 100 x-values
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    # Set y-ticks for every one decimal
    ax.yaxis.set_minor_locator(AutoMinorLocator(10))
    # Tick sizes and direction
    ax.tick_params(axis="x", which='major', length=10, width=3, direction='in', left=True, right=True)
    ax.tick_params(axis="x", which='minor', length=7, width=2, direction='in', left=True, right=True)
    # Tick sizes and direction
    ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis="y", length=10, width=3, direction='in')
    ax.tick_params(axis="y", which='minor', length=7, width=2, direction='in')
    # Add border
    ax.spines["top"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["right"].set_linewidth(3)

    if outfile:
        fig.savefig(outfile, format='pdf', bbox_inches='tight')
        print(f"Figure saved successfully as {outfile}")
    else:
        print("No file path provided. Skipping file saving.")
    plt.show()
    
    return fig, ax

def lya_mc(guess, priors, spec, num_walkers, num_steps, outfile=None, priorplot_path=None):
    '''
    Run a Markov Chain Monte Carlo (MCMC) simulation to fit a Lyman-alpha model 
    to an observed spectrum using a neural network model.

    Parameters:
    - guess (list or array-like): Initial guess for the parameters.
    - priors (tuple): Bounds for the parameters in the form 
      (sigma_int_max, sigma_int_min, logN_HI_max, logN_HI_min, 
       v_exp_max, v_exp_min, logT_max, logT_min, v_lya_max, v_lya_min).
    - spec (tuple): The observed spectrum and its error, in the form (x, y, yerr).
    - num_walkers (int): Number of MCMC walkers.
    - num_steps (int): Number of MCMC steps to perform.
    - outfile (str, optional): Path to save the MCMC backend. If None, no backend is saved.
    - priorplot_path (str, optional): Path to save prior distribution plots. If None, plots are not saved.

    Returns:
    - sampler (emcee.EnsembleSampler): The MCMC sampler object.
    '''

    # All parameters
    value = np.array(guess)
    print("Guess: " , value)
    num_rows = num_walkers
    pos = np.ones((num_rows, len(value))) * value
    
    # Get priors
    sigma_int_max, sigma_int_min, logN_HI_max, logN_HI_min, v_exp_max, v_exp_min, logT_max, logT_min, v_lya_max, v_lya_min = priors

    # Drawing guesses from uniform distribution
    # sigma_int
    pos[:, 0] = np.random.uniform(sigma_int_min, sigma_int_max, num_walkers)
    # logN_HI
    pos[:, 1] = np.random.uniform(logN_HI_min, logN_HI_max, num_walkers)
    # v_exp
    pos[:, 2] = np.random.uniform(v_exp_min, v_exp_max, num_walkers)
    # logT
    pos[:, 3] = np.random.uniform(logT_min, logT_max, num_walkers)
    #v_lya
    pos[:, 4] = np.random.uniform(v_lya_min, v_lya_max, num_walkers)
    nwalkers, ndim = pos.shape
    
    # Set up the backend
    if outfile is not None:
        backend = emcee.backends.HDFBackend(outfile)
        backend.reset(nwalkers, ndim)

    # Plot prior distributions
    for i in np.arange(0, ndim):
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(4,4)

        ax.hist(pos[:, i], bins=20, color='#ffb000')
        fig.savefig(f"{priorplot_path}_{i}.pdf")
        plt.close(fig)
    
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(spec, priors), backend=backend
    )

    print("Booting MCMC...")
    sampler.run_mcmc(pos, num_steps, progress=True)
    
    return sampler

def log_likelihood(params, spec):
    '''
    Compute the log-likelihood of the model given the parameters and the observed spectrum.

    Parameters:
    - params (list or array-like): Model parameters [sigma_int, logN_HI, v_exp, logT, v_lya].
    - spec (tuple): The observed spectrum and its error, in the form (x, y, yerr).

    Returns:
    - log_likelihood (float): The log-likelihood of the model given the observed data.
    '''
    
    sigma_int, logN_HI, v_exp, logT, v_lya = params
    x, y, yerr = spec
    lya_y = get_model_interp(v_lya, x, sigma_int, logN_HI, v_exp, logT)
    lya_y = lya_y/sum(lya_y) # Normalize profile
    A_lya = calc_A(y, lya_y, yerr)
    y_pred = A_lya * lya_y
    
    return -0.5 * np.sum(((y - y_pred) / yerr)**2)

def log_prior(params, priors):
    '''
    Compute the log-prior of the parameters assuming uniform priors within given bounds.

    Parameters:
    - params (list or array-like): Model parameters [sigma_int, logN_HI, v_exp, logT, v_lya].
    - priors (tuple): Bounds for the parameters in the form 
      (sigma_int_max, sigma_int_min, logN_HI_max, logN_HI_min, 
       v_exp_max, v_exp_min, logT_max, logT_min, v_lya_max, v_lya_min).

    Returns:
    - log_prior (float): The log-prior probability of the parameters.
    '''

    sigma_int, logN_HI, v_exp, logT, v_lya = params
    
    sigma_int_max, sigma_int_min, logN_HI_max, logN_HI_min, v_exp_max, v_exp_min, logT_max, logT_min, v_lya_max, v_lya_min = priors

    if (sigma_int > sigma_int_max):
        return -np.inf
    if (sigma_int < sigma_int_min):
        return -np.inf

    if (logN_HI > logN_HI_max):
        return -np.inf
    if (logN_HI < logN_HI_min):
        return -np.inf

    if (v_exp > v_exp_max):
        return -np.inf
    if (v_exp < v_exp_min):
        return -np.inf

    if (logT > logT_max):
        return -np.inf
    if (logT < logT_min):
        return -np.inf

    if (v_lya > v_lya_max):
        return -np.inf
    if (v_lya < v_lya_min):
        return -np.inf
    
    return 0.0

def log_probability(params, spec, priors):
    '''
    Compute the log-probability of the parameters, which is the sum of the log-prior and log-likelihood.

    Parameters:
    - params (list or array-like): Model parameters [sigma_int, logN_HI, v_exp, logT, v_lya].
    - spec (tuple): The observed spectrum and its error, in the form (x, y, yerr).
    - priors (tuple): Bounds for the parameters in the form 
      (sigma_int_max, sigma_int_min, logN_HI_max, logN_HI_min, 
       v_exp_max, v_exp_min, logT_max, logT_min, v_lya_max, v_lya_min).

    Returns:
    - log_probability (float): The log-probability of the parameters.
    '''
    lp = log_prior(params, priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, spec)

def corner_plot(sampler, labels, outfile=None, discard=200, thin=None, flat=True):
    '''
    Produce a corner plot from the MCMC sampling results.

    Parameters:
    - sampler (emcee.EnsembleSampler): The MCMC sampler object containing the samples.
    - labels (list of str): List of parameter names corresponding to the dimensions of the MCMC chain.
    - outfile (str, optional): Path to save the corner plot figure as a file. If None, the figure is not saved.
    - discard (int, optional): Number of initial samples to discard as burn-in. Default is 200.
    - thin (int, optional): Interval between saved samples in the chain. If None, no thinning is applied.
    - flat (bool, optional): Whether to return the flat chain. Default is True.

    Returns:
    - fig (matplotlib.figure.Figure): The corner plot figure object.
    '''
    
    if (thin is not None):
        flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=flat)
    else:
        flat_samples = sampler.get_chain(discard=discard, flat=flat)

    fig = corner.corner(
        flat_samples, show_titles=True, labels=labels, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84]
    )

    if (outfile is not None):
        fig.savefig(outfile)
        plt.close(fig)
        
    return fig

def progress_plot(sampler, labels, outfile=None, discard=200, thin=None):
    '''
    Produce a plot showing the progress of the MCMC walkers over the walker steps.

    Parameters:
    - sampler (emcee.EnsembleSampler): The MCMC sampler object containing the samples.
    - labels (list of str): List of parameter names corresponding to the dimensions of the MCMC chain.
    - outfile (str, optional): Path to save the progress plot figure. If None, the figure is not saved.
    - discard (int, optional): Number of initial samples to discard as burn-in. Default is 200.
    - thin (int, optional): Interval between saved samples in the chain. If None, no thinning is applied.

    Returns:
    - fig (matplotlib.figure.Figure): The progress plot figure object.
    '''
    ndim = len(labels)
    
    if (thin is not None):
        samples = sampler.get_chain(discard=discard, thin=thin)
    else:
        samples = sampler.get_chain(discard=discard)
    print(samples.shape)
    
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    
    if (outfile is not None):
        fig.savefig(outfile)
        plt.close(fig)
    
    return fig

def save_results(flat_samples, outfile, spec):
    '''
    Save the results of the MCMC sampling to a CSV file.

    Parameters:
    - flat_samples (np.ndarray): The array of MCMC samples with shape (n_samples, n_params).
    - outfile (str): Path to the CSV file where results will be saved.
    - spec (tuple): The observed spectrum and its error, in the form (x, y, yerr).

    Returns:
    - The function writes the results to a CSV file and does not return a value.
    '''

    x, y, yerr = spec
    
    # Find best fit and upper and lower one sigma error bars
    sigma_int = np.percentile(flat_samples[:, 0], [50])
    sigma_int_lower = np.percentile(flat_samples[:, 0], [50]) - np.percentile(flat_samples[:, 0], [16])
    sigma_int_upper = np.percentile(flat_samples[:, 0], [84]) - np.percentile(flat_samples[:, 0], [50])
    logN_HI = np.percentile(flat_samples[:, 1], [50])
    logN_HI_lower = np.percentile(flat_samples[:, 1], [50]) - np.percentile(flat_samples[:, 1], [16])
    logN_HI_upper = np.percentile(flat_samples[:, 1], [84]) - np.percentile(flat_samples[:, 1], [50])
    v_exp = np.percentile(flat_samples[:, 2], [50])
    v_exp_lower = np.percentile(flat_samples[:, 2], [50]) - np.percentile(flat_samples[:, 2], [16])
    v_exp_upper = np.percentile(flat_samples[:, 2], [84]) - np.percentile(flat_samples[:, 2], [50])
    logT = np.percentile(flat_samples[:, 3], [50])
    logT_lower = np.percentile(flat_samples[:, 3], [50]) - np.percentile(flat_samples[:, 3], [16])
    logT_upper = np.percentile(flat_samples[:, 3], [84]) - np.percentile(flat_samples[:, 3], [50])
    v_lya = np.percentile(flat_samples[:, 4], [50])
    v_lya_lower = np.percentile(flat_samples[:, 4], [50]) - np.percentile(flat_samples[:, 4], [16])
    v_lya_upper = np.percentile(flat_samples[:, 4], [84]) - np.percentile(flat_samples[:, 4], [50])

    # Calculate normalization factor for the model
    lya_y = get_model_interp(v_lya[0], x, sigma_int[0], logN_HI[0], v_exp[0], logT[0]) # Need "[0]" here to not mess with the pass of labels to the neural network
    lya_y = lya_y/sum(lya_y)
    A_lya = calc_A(y, lya_y, yerr)
    
    # Save to dataframe csv
    df = pd.DataFrame({
        'sigma_int': sigma_int,
        'sigma_int_lower': sigma_int_lower,
        'sigma_int_upper': sigma_int_upper,
        'logN_HI': logN_HI,
        'logN_HI_lower': logN_HI_lower,
        'logN_HI_upper': logN_HI_upper,
        'v_exp': v_exp,
        'v_exp_lower': v_exp_lower,
        'v_exp_upper': v_exp_upper,
        'logT': logT,
        'logT_lower': logT_lower,
        'logT_upper': logT_upper,
        'v_lya': v_lya,
        'v_lya_lower': v_lya_lower,
        'v_lya_upper': v_lya_upper,
        'A_lya': A_lya,
    })
    df.to_csv(outfile, index=False)

def get_results(results_filename, uncertainties=False, rd=2):
    '''
    Load results from the CSV file containing the results from the MCMC sampling and optionally include uncertainties.

    Parameters:
    - results_filename (str): Path to the CSV file containing results from the MCMC sampling.
    - uncertainties (bool, optional): Whether to include uncertainty values. Default is False.
    - rd (int, optional): Number of decimal places to round the results, except A_lya. Default is 2.

    Returns:
    - tuple: A tuple containing the parameter results. If `uncertainties` is True, the tuple includes uncertainty values.
    '''

    # Load the CSV into a DataFrame
    results_df = pd.read_csv(results_filename)

    # Get parameters from file and get coefficients
    sigma_int = round(results_df['sigma_int'][0],rd)
    logN_HI = round(results_df['logN_HI'][0],rd)
    v_exp = round(results_df['v_exp'][0],rd)
    logT = round(results_df['logT'][0],rd)
    v_lya = round(results_df['v_lya'][0],rd)
    A_lya = results_df['A_lya'][0]

    # Get uncertainties from file
    if uncertainties == True:
        sigma_int_upper = round(results_df['sigma_int_upper'][0],rd)
        sigma_int_lower = round(results_df['sigma_int_lower'][0],rd)
        logN_HI_upper = round(results_df['logN_HI_upper'][0],rd)
        logN_HI_lower = round(results_df['logN_HI_lower'][0],rd)
        v_exp_upper = round(results_df['v_exp_upper'][0],rd)
        v_exp_lower = round(results_df['v_exp_lower'][0],rd)
        logT_upper = round(results_df['logT_upper'][0],rd)
        logT_lower = round(results_df['logT_lower'][0],rd)
        v_lya_upper = round(results_df['v_lya_upper'][0],rd)
        v_lya_lower = round(results_df['v_lya_lower'][0],rd)
        return sigma_int, logN_HI, v_exp, logT, v_lya, A_lya, sigma_int_upper, sigma_int_lower, logN_HI_upper, logN_HI_lower, v_exp_upper, v_exp_lower, logT_upper, logT_lower, v_lya_upper, v_lya_lower
    else:
        return sigma_int, logN_HI, v_exp, logT, v_lya, A_lya

def save_plot(spec, results_filename, plot_path, uncertainties, save_dir):
    '''
    Save a plot of the observed data and model fit and save the fitted model to file.

    Parameters:
    - spec (tuple): The observed spectrum and its error, in the form (x, y, yerr).
    - results_filename (str): Path to the CSV file containing results from the MCMC sampling.
    - plot_path (str): Path to save the plot.
    - uncertainties (bool): Whether to include uncertainty values in the plot.
    - save_dir (str): Directory to save the model predictions.

    Returns:
    - The function saves the plot and model predictions to files and does not return a value.
    '''

    x, y, yerr = spec
    results = get_results(results_filename, uncertainties=uncertainties)
    lya_y = get_model_interp(results[4], x, results[0], results[1], results[2], results[3])
    lya_y = lya_y/sum(lya_y) # Normalize profile
    lya_y = results[5] * lya_y
    plot_lya(obs_x=x, obs_y=y, obs_err=yerr, lya_x=x, lya_y=lya_y, outfile=plot_path, xmin=-800, xmax=800)
    save_model_pred(x, lya_y, save_dir)
    return

def save_model_pred(x, y, save_dir):
    '''
    Save the fitted model predictions to a CSV file.

    Parameters:
    - x (np.ndarray): Array of velocity values.
    - y (np.ndarray): Array of flux values.
    - save_dir (str): Directory to save the CSV file.

    Returns:
    - The function saves the model predictions to a CSV file and does not return a value.
    '''
    data = np.column_stack((x, y))
    # Define the filename
    filename = 'output_model.csv'
    filepath = os.path.join(save_dir, filename)
    # Save to file in the specified format
    header = "vel,flux"  # Header for the file
    np.savetxt(filepath, data, delimiter=',', header=header, comments='', fmt='%.6e,%.6e')

def load_model_pred(filepath):
    '''
    Load the model predictions from the CSV file saved with save_model_pred.

    Parameters:
    - filepath (str): Path to the CSV file containing the model predictions.

    Returns:
    - tuple of np.ndarray: Arrays of velocity and flux values from the CSV file.
    '''

    # Read the CSV file into a numpy array
    data = np.loadtxt(filepath, delimiter=',', skiprows=1) # ignore the header

    # Separate the velocity and flux columns
    vel, flux = data[:, 0], data[:, 1]

    return vel, flux

def calc_chi2(obs_y, model_y, sigma):
    '''
    Calculate the chi-squared statistic for a set of observed and model data.

    Parameters:
    - obs_y (array-like): Observed data values.
    - model_y (array-like): Model data values.
    - sigma (array-like): Error spectrum of the observed data.

    Returns:
    - chi2 (float): The chi-squared statistic.
    '''
        
    return np.sum( (obs_y - model_y)**2 / (sigma**2) )

def calc_A(obs_y, model_y, sigma):
    '''
    Calculate the normalization factor A_lya by minimizing the chi-squared statistic.

    Parameters:
    - obs_y (array-like): Observed data values.
    - model_y (array-like): Model data values.
    - sigma (array-like): Error spectrum of the observed data.

    Returns:
    - A_lya (float): The normalization factor A_lya.
    '''
    r1 = np.sum((model_y/sigma)**2)
    r2 = np.sum((obs_y*model_y)/(sigma**2))
    A_lya = r2 / r1
    
    return A_lya

def read_input(filename):
    '''
    Reads the input from a file with the file format used in this code. Specifically, this is used to read the guess.txt and prior.txt files.

    Parameters:
    - filename (str): The path to the input file.

    Returns:
    - input (list of float): A list of floating-point numbers parsed from the file.
    '''
    input = []
    with open(filename, 'r') as f:
        for line in f:
            # Skip comments or empty lines
            if line.startswith("#") or line.strip() == "":
                continue
            # Parse the parameter value
            _, value = line.split(":")
            input.append(float(value.strip()))
    return input

def find_training_spectrum(input_array, training_labels, training_spectra, validation_labels, validation_spectra):
    '''
    Find and retrieve the spectrum corresponding to the input labels from the training or validation sets.

    This function searches for a given set of input labels in both the training and validation datasets. 
    If a match is found, it returns the corresponding spectrum and the dataset type (either "training" or "validation"). 
    If no match is found, it prints a message and returns `None` for both the spectrum and dataset type.

    Parameters:
    - input_array (numpy.ndarray): Array of labels to search for in the training or validation datasets.
    - training_labels (numpy.ndarray): Array of labels used in the training dataset.
    - training_spectra (numpy.ndarray): Array of spectra corresponding to the training labels.
    - validation_labels (numpy.ndarray): Array of labels used in the validation dataset.
    - validation_spectra (numpy.ndarray): Array of spectra corresponding to the validation labels.

    Returns:
    - spectrum (numpy.ndarray or None): The spectrum corresponding to the input labels if found; otherwise, `None`.
    - dataset_type (str or None): The type of dataset ("training" or "validation") where the spectrum was found; 
                                  otherwise, `None` if no match was found.
    '''

    # Search in training labels
    training_index = np.where((training_labels == input_array).all(axis=1))[0]
    if len(training_index) > 0:
        # Return the spectrum from training_spectra
        spectrum = training_spectra[training_index[0]]
        return spectrum, "training"
    
    # Search in validation labels
    validation_index = np.where((validation_labels == input_array).all(axis=1))[0]
    if len(validation_index) > 0:
        # Return the spectrum from validation_spectra
        spectrum = validation_spectra[validation_index[0]]
        return spectrum, "validation"
    
    # If no match is found in either
    print("No matching label found in training or validation sets.")
    return None, None
