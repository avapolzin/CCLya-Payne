# main code for fitting lya profiles to observed spectras

import pickle
import os

from .cclya_payne import utils
#import cclya_payne.utils

def fit_lya(obs_x, obs_y, obs_err, num_walkers=50, num_steps=50, discard=10, thin=5, guess_file='guess.txt', prior_file='prior.txt', save_dir=os.getcwd()):
    '''
    Perform MCMC fitting for observed Lyman-alpha spectra and save results.

    Parameters:
    - obs_x (numpy.ndarray): Wavelength values for observed spectra.
    - obs_y (numpy.ndarray): Flux values for observed spectra.
    - obs_err (numpy.ndarray): Error values for observed spectra.
    - num_walkers (int, optional): Number of MCMC walkers. Default is 50.
    - num_steps (int, optional): Number of MCMC steps to perform. Default is 50.
    - discard (int, optional): Number of initial steps to discard from the MCMC chain. Default is 10.
    - thin (int, optional): Thinning factor for the MCMC chain. Default is 5.
    - guess_file (str, optional): Path to the file containing initial guesses for the parameters. Default is 'guess.txt'.
    - prior_file (str, optional): Path to the file containing prior distributions for the parameters. Default is 'prior.txt'.
    - save_dir (str, optional): Directory where the results and plots will be saved. Default is the current working directory.

    Returns:
    - None: This function performs the fitting, saves results, and generates plots.
    '''

    # Load observed spectrum
    spec = obs_x, obs_y, obs_err

    # Declare parameter names
    labels = ['sigma_int', 'logN_HI', 'v_exp', 'logT', 'v_lya']

    # How many free parameters?
    num_params = len(labels)
    
    # Read guess and priors for parameters
    guess = utils.read_input(guess_file)
    priors = utils.read_input(prior_file)

    # Create file paths for a variety of saved files
    # MCMC backend path
    backend_path = os.path.join(save_dir, "backend.dat")
    # Sampler path
    sampler_path = os.path.join(save_dir, f"sampler.txt")
    # Samples path
    samples_path = os.path.join(save_dir, f"samples.txt")
    # Samples path
    flatsamples_path = os.path.join(save_dir, f"flatsamples.txt")
    # Results path
    results_path = os.path.join(save_dir, f"results.csv")
    # Chi2 path
    chi2_path = os.path.join(save_dir, f"chi2.txt")
    # Plot path
    plot_path = os.path.join(save_dir, f"plot.pdf")
    # Progress path
    progress_path = os.path.join(save_dir, f"progress.pdf")
    # Corner path
    corner_path = os.path.join(save_dir, f"cornerplot.pdf")
    # MCMC prior plot path
    priorplot_path = os.path.join(save_dir, f"priorplot") # Skipping suffix here because there is one plot per parameter

    # Start MCMC
    sampler = utils.lya_mc(guess, priors, spec, num_walkers=num_walkers, num_steps=num_steps, outfile=backend_path, priorplot_path=priorplot_path)
    # Collect samples
    samples = sampler.get_chain()
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    # Save the sampler object
    with open(sampler_path, 'wb') as f:
        pickle.dump(sampler, f)
    # Save the samples
    with open(samples_path, 'wb') as f:
        pickle.dump(samples, f)
    # Save the flat samples
    with open(flatsamples_path, 'wb') as f:
        pickle.dump(flat_samples, f)

    # Save results
    utils.save_results(flat_samples, results_path, spec)
    
    # Save plot
    utils.save_plot(spec, results_path, plot_path, uncertainties=True, save_dir=save_dir)

    # Load the saved predicted spectrum
    pred_filepath = os.path.join(save_dir, 'output_model.csv')
    _, model_y = utils.load_model_pred(pred_filepath)

    # Save chi2
    chi2 = utils.calc_chi2(obs_y, model_y, obs_err)
    dof = len(obs_y) - num_params
    chi2_dof = chi2/dof
    with open(chi2_path, "w") as file:
        file.write("# MCMC chi2\n")
        file.write(f"chi2 = {chi2}\n")
        file.write(f"chi2_dof = {chi2_dof}\n")

    # Save corner plot
    utils.corner_plot(sampler, labels, outfile=corner_path, discard=discard, thin=thin, flat=True)

    # Save progress plot
    utils.progress_plot(sampler, labels, outfile=progress_path, discard=discard, thin=thin)
