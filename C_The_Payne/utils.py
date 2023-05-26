# a few low-level functions that are used throughout
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
import os

def read_in_neural_network(load_local=False, nn_fname=""):
    '''
    read in the weights and biases parameterizing a particular neural network.
    You can read in existing networks from the neural_nets/ directory, or you
    can train your own networks and edit this function to read them in.
    
    ##
    nn_file - a string, contains filename or filepath of a trained neural network file. F.ex: 'NN_normalized_spectra_v4.npz'
    '''
    
    if load_local==False:
        nn_fname = os.path.join(os.path.dirname(os.path.realpath(__file__)),'other_data/NN_normalized_spectra.npz')
    
    tmp = np.load(nn_fname)
    
    print(f'Neural network file {nn_fname} has been read.')
    
    w_array_0 = tmp["w_array_0"]
    w_array_1 = tmp["w_array_1"]
    w_array_2 = tmp["w_array_2"]
    b_array_0 = tmp["b_array_0"]
    b_array_1 = tmp["b_array_1"]
    b_array_2 = tmp["b_array_2"]
    x_min = tmp["x_min"]
    x_max = tmp["x_max"]
    NN_coeffs = (w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max)
    tmp.close()
    
    return NN_coeffs


def load_wavelength_array(load_local=False, wav_fname=""):
    '''
    read in the default wavelength grid onto which we interpolate all spectra
    
    load_local - a boolean, choose to load a file on your local computer
    wav_fname - a string, contains the filename of the wavelength npz file you want to read
    '''
    
    if load_local==False:
        wavelength = os.path.join(os.path.dirname(os.path.realpath(__file__)),'other_data/TLAC_wavelength.npz')
        
    tmp = np.load(wav_fname)
    wavelength = tmp['wavelength'] # VELOCITY SPACE
    tmp.close()
    
    return wavelength


def load_apogee_mask(load_local=False, mask_fname=""): ## NEED SPECIFIC MASK FOR EACH INSTRUMENT
    '''
    read in the pixel mask with which we will omit bad pixels during spectral fitting

    load_local - a boolean, choose to load a file on your local computer
    mask_fname - a string, contains the filename of the mask npz file you want to read
    '''
    
    if load_local==False:
        mask_fname = os.path.join(os.path.dirname(os.path.realpath(__file__)),'other_data/TLAC_mask.npz')
    
    tmp = np.load(mask_fname)
    mask = tmp['apogee_mask']
    tmp.close()
    
    return mask


def load_cannon_contpixels(): ## MODIFY NUMBER OF PIXELS IN CONTINUUM FILE # No continuum fitting needed in my case
    '''
    read in the default list of pixels for continuum fitting.
    '''
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'other_data/TLAC_cont_pixels.npz') # MODIFY PIXELS
    tmp = np.load(path)
    pixels_cannon = tmp['pixels_cannon']
    tmp.close()
    return pixels_cannon


def load_training_data(load_local=False, training_fname="", validation_fname=""): # MODIFY NUMBER OF SPECTRA (FROM 800 TO 'NEW')
    '''
    read in the default Kurucz training spectra for APOGEE

    Here we only consider a small number (<1000) of training spectra.
    In practice, more training spectra will be better. The default
    neural network was trained using 12000 training spectra.
    '''
    
    if load_local==False:
        training_fname = os.path.join(os.path.dirname(os.path.realpath(__file__)),'other_data/TLAC_training_spectra.npz')
        validation_fname = os.path.join(os.path.dirname(os.path.realpath(__file__)),'other_data/TLAC_validation_spectra.npz')
    
    # TRAINING
    tmp = np.load(training_fname)
    training_labels = (tmp["labels"].T)
    training_spectra = tmp["spectra"]
    tmp.close()
    # VALIDATION
    tmp = np.load(validation_fname)
    validation_labels = (tmp["labels"].T)
    validation_spectra = tmp["spectra"]
    tmp.close()

    return training_labels, training_spectra, validation_labels, validation_spectra


def get_apogee_continuum(spec, spec_err = None, cont_pixels = None): ### NO USE FOR MY PURPOSES AS I HAVE MY OWN FITTING CODE
    ## CHECK HOW THIS IS DIFFERENT WITH VELOCITY-SPACE SPECTRA VS. WHAT THE PAYNE USES
    ## REARRANGE THE BLUE, GREEN, RED CATEGORIES TO ONE CATEGORY FOR LY CONTINUUM
    
    '''
    continuum normalize spectrum.
    pixels with large uncertainty are weighted less in the fit.
    '''
    if cont_pixels is None:
        cont_pixels = load_cannon_contpixels()
    cont = np.empty_like(spec)

    wavelength = load_wavelength_array() # VELOCITY SPACE

    deg = 4

    # if we haven't given any uncertainties, just assume they're the same everywhere.
    if spec_err is None:
        spec_err = np.zeros(spec.shape[0]) + 0.0001

    # Rescale wavelengths
    bluewav = 2*np.arange(2920)/2919 - 1
    greenwav = 2*np.arange(2400)/2399 - 1
    redwav = 2*np.arange(1894)/1893 - 1

    blue_pixels= cont_pixels[:2920]
    green_pixels= cont_pixels[2920:5320]
    red_pixels= cont_pixels[5320:]

    # blue
    cont[:2920]= _fit_cannonpixels(bluewav, spec[:2920], spec_err[:2920],
                        deg, blue_pixels) # VELOCITY!!
    # green
    cont[2920:5320]= _fit_cannonpixels(greenwav, spec[2920:5320], spec_err[2920:5320],
                        deg, green_pixels) # VELOCITY!!
    # red
    cont[5320:]= _fit_cannonpixels(redwav, spec[5320:], spec_err[5320:], deg, red_pixels) # VELOCITY!!
    return cont


def _fit_cannonpixels(wav, spec, specerr, deg, cont_pixels):
    ## "_fit_cannonpixels()" DOES NOT NEED WAVELENGTH INSTEAD OF VELOCITY - JUST TAKES Y-VALUES AND USES THE INDICES OF THE PIXELS
    
    '''
    Fit the continuum to a set of continuum pixels
    helper function for get_apogee_continuum()
    '''
    chpoly = np.polynomial.Chebyshev.fit(wav[cont_pixels], spec[cont_pixels],
                deg, w=1./specerr[cont_pixels])
    return chpoly(wav)

## I ADDED THE FUNCTIONS BELOW
def get_loss(load_local=False, loss_fname=""):
    
    if load_local==False:
        loss_fname = os.path.join(os.path.dirname(os.path.realpath(__file__)),'other_data/training_loss.npz')
    
    tmp = np.load(loss_fname) # the output array also stores the training and validation loss
    
    print(f'Read training and validation loss of {loss_fname}.')
    
    training_loss = tmp["training_loss"]
    validation_loss = tmp["validation_loss"]
    tmp.close()
    
    return training_loss, validation_loss

def get_validation_spectra(load_local=False, validation_fname=""):
    
    if load_local==False:
        validation_fname = os.path.join(os.path.dirname(os.path.realpath(__file__)),'other_data/TLAC_validation_spectra.npz')
    
    tmp = np.load(validation_fname) # the output array also stores the training and validation loss
    
    print(f'Read validation spectra of {validation_fname}.')
    
    spec_arr = tmp["spectra"]
    label_arr = tmp["labels"]
    tmp.close()
    
    return spec_arr, label_arr
