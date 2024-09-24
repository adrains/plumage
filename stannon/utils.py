"""Utilities for the Stan implementation of the Cannon. Developed from code
originally written by Dr Andy Casey.
"""
import os
import logging
import pickle
import yaml
import numpy as np
import pystan as stan
import pystan.plots as plots
from tqdm import tqdm
from tempfile import mkstemp
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from PyAstronomy.pyasl import instrBroadGaussFast

__all__ = ["read", "sampling_kwds", "plots"]


def read(path, cached_path=None, recompile=False, overwrite=True,
    verbose=True):
    r"""
    Load a Stan model from a file. If a cached file exists, use it by default.

    :param path:
        The path of the Stan model.

    :param cached_path: [optional]
        The path of the cached Stan model. By default this will be the same  as
        :path:, with a `.cached` extension appended.

    :param recompile: [optional]
        Recompile the model instead of using a cached version. If the cached
        version is different from the version in path, the model will be
        recompiled automatically.
    """

    cached_path = cached_path or "{}.cached".format(path)

    with open(path, "r") as fp:
        model_code = fp.read()

    while os.path.exists(cached_path) and not recompile:
        with open(cached_path, "rb") as fp:
            model = pickle.load(fp)

        if model.model_code != model_code:
            if verbose:
                logging.warn("Cached model at {} differs from the code in {}; "\
                             "recompiling model".format(cached_path, path))
            recompile = True
            continue

        else:
            if verbose:
                logging.info("Using pre-compiled model from {}".format(cached_path)) 
            break

    else:
        model = stan.StanModel(model_code=model_code)

        # Save the compiled model.
        if not os.path.exists(cached_path) or overwrite:
            with open(cached_path, "wb") as fp:
                pickle.dump(model, fp)


    return model


def sampling_kwds(**kwargs):
    r"""
    Prepare a dictionary that can be passed to Stan at the sampling stage.
    Basically this just prepares the initial positions so that they match the
    number of chains.
    """

    kwds = dict(chains=4)
    kwds.update(kwargs)

    if "init" in kwds:
        kwds["init"] = [kwds["init"]] * kwds["chains"]

    return kwds


class suppress_output(object):
    """ Suppress all stdout and stderr. """

    def __init__(self, suppress_output=True):
        """
        self.null_fds = [
            os.open(os.devnull, os.O_RDWR),
            os.open(os.devnull, os.O_RDWR)
        ]
        """
        self.suppress_output = suppress_output

        if self.suppress_output:
            self.null_fds = [
                mkstemp(),
                mkstemp()
            ]
            # Save the actual stdout (1) and stderr (2) file descriptors.
            self.save_fds = [os.dup(1), os.dup(2)]


    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        if self.suppress_output:
            os.dup2(self.null_fds[0][0], 1)
            os.dup2(self.null_fds[1][0], 2)
        return self

    @property
    def stdout(self):
        with open(self.null_fds[0][1], "r") as fp:
            stdout = fp.read()
        return stdout        

    @property
    def stderr(self):
        with open(self.null_fds[1][1], "r") as fp:
            stderr = fp.read()
        return stderr

    def __exit__(self, *_):

        # Re-assign the real stdout/stderr back to (1) and (2)
        if self.suppress_output:
            os.dup2(self.save_fds[0], 1)
            os.dup2(self.save_fds[1], 2)

            # Close the null files and descriptors.
            for fd in self.save_fds:
                os.close(fd)

            self.outputs = []
            for fd, p in self.null_fds:
                with open(p, "r") as fp:
                    self.outputs.append(fp.read())

                os.close(fd)
                os.unlink(p)


def get_lvec(labels):
    """
    Constructs a label vector for an arbitrary number of labels
    Assumes that our model is quadratic in the labels

    Modified from: 
    https://github.com/annayqho/TheCannon/blob/master/TheCannon/infer_labels.py

    Parameters
    ----------
    labels: numpy ndarray
        pivoted label values for one star
    Returns
    -------
    lvec: numpy ndarray
        label vector
    """
    nlabels = len(labels)
    # specialized to second-order model
    linear_terms = labels 
    quadratic_terms = np.outer(linear_terms, 
                               linear_terms)[np.triu_indices(nlabels)]
    lvec = np.hstack(([1], linear_terms, quadratic_terms))
    return lvec


def multiply_coeff_label_vectors(coeffs, *labels):
    """ Takes the dot product of coefficients vec & labels vector 
    Modified from: 
    https://github.com/annayqho/TheCannon/blob/master/TheCannon/infer_labels.py

    Parameters
    ----------
    coeffs: numpy ndarray
        the coefficients on each element of the label vector
    *labels: numpy ndarray
        label vector
    Returns
    -------
    dot product of coeffs vec and labels vec
    """
    lvec = get_lvec(list(labels))
    return np.dot(coeffs, lvec)


def infer_labels(theta, scatter, fluxes, ivars, lbl_mean, lbl_std, n_labels=3):
    """
    Use coefficients and scatters from a trained Cannon model to infer the 
    labels for a set of normalised spectra.

    Modified from: 
    https://github.com/annayqho/TheCannon/blob/master/TheCannon/infer_labels.py
    
    Parameters
    ----------
    theta: float array
        Coefficients from the Stannon model, of shape [n_pixels, n_coeff] where
        n_coeff corresponds to the number of labels and the polynomial order
        of the model. n_coeff = 10 for a quadratic Stannon with 3 labels.

    scatter: float array
        Intrinsic scatter per pixel from the Stannon model, vector of length
        n_pixels.

    fluxes: float array
        Science fluxes to infer labels for, of shape [n_spectra, n_pixels].
        Must be normalised the same as training spectra.

    ivars: float array
        Science flux inverse variances, of shape [n_spectra, n_pixels]. Must be 
        normalised the same as training spectra.

    lbl_mean: float array
        Mean of each label used to whiten. Vector of length [n_labels].

    lbl_std: float array
        Standard deviation of each label used to whiten. Vector of length 
        [n_labels].

    n_labels: int
        Number of labels, defaults to 3.

    Returns
    -------
    labels_all: float array
        Cannon predicted labels (de-whitened), of shape [n_spectra, n_label].

    errs_all: float array
        ...

    chi2_all: float array
        Chi^2 fit for each star, vector of length [n_spectra].
    """
    # Initialise
    errs_all = np.zeros((len(fluxes), n_labels))
    chi2_all = np.zeros(len(fluxes))
    labels_all = np.zeros((len(fluxes), n_labels))
    starting_guess = np.ones(n_labels)

    lbl = "Inferring labels"

    for star_i, (flux, ivar) in enumerate(zip(tqdm(fluxes, desc=lbl), ivars)):
        # Where the ivar == 0, set normalized flux to 1 and the sigma to 100
        bad = ivar == 0
        flux[bad] = 1.0
        sigma = np.ones(ivar.shape) * 100.0
        sigma[~bad] = np.sqrt(1.0 / ivar[~bad])

        errbar = np.sqrt(sigma**2 + scatter**2)

        try:
            labels, cov = curve_fit(multiply_coeff_label_vectors, theta, flux, 
                                    p0=starting_guess, sigma=errbar, 
                                    absolute_sigma=True)
        except:
            labels = np.zeros(starting_guess.shape)*np.nan
            cov = np.zeros((len(starting_guess),len(starting_guess)))*np.nan
                    
        chi2 = ((flux-multiply_coeff_label_vectors(theta, *labels))**2 
                * ivar / (1 + ivar * scatter**2))
        chi2_all[star_i] = sum(chi2)
        labels_all[star_i,:] = labels * lbl_std + lbl_mean
        errs_all[star_i,:] = np.sqrt(cov.diagonal()) * lbl_std

    return labels_all, errs_all, chi2_all


def load_cannon_settings(yaml_path):
    """Import our Cannon settings YAML file as a dictionary and return the
    object equivalent.

    Parameters
    ----------
    yaml_path: string
        Path to the saved YAML file.

    Returns
    -------
    cs: YAMLSettings object
        YAMLSettings object with attributes equivalent to YAML keys.
    """
    # Load in YAML file as dictionary
    with open(yaml_path) as yaml_file:
        yaml_dict = yaml.safe_load(yaml_file)

    # Add in missing keywods
    yaml_dict["label_names"] = \
        yaml_dict["base_labels"] + yaml_dict["abundance_labels"]
    
    yaml_dict["n_labels"] = len(yaml_dict["label_names"])

    yaml_dict["log_refresh_step"] = \
        int(yaml_dict["max_iter"] / yaml_dict["refresh_rate_frac"])
    
    yaml_dict["is_cross_validated"] = yaml_dict["do_cross_validation"]

    # Do a consistency check
    if len(yaml_dict["lit_std_scale_fac"]) != yaml_dict["n_labels"]:
        raise ValueError("Cannon setting lit_std_scale_fac length != n_labels")

    # Finally convert to our wrapper object form and return
    cs = YAMLSettings(yaml_dict)

    return cs


def load_yaml_settings(yaml_path):
    """Import our settings YAML file as a dictionary and return the object 
    equivalent.

    Parameters
    ----------
    yaml_path: string
        Path to the saved YAML file.

    Returns
    -------
    yaml_settings: YAMLSettings object
        Settings object with attributes equivalent to YAML keys.
    """
    # Load in YAML file as dictionary
    with open(yaml_path) as yaml_file:
        yaml_dict = yaml.safe_load(yaml_file)

    # Correctly set None variables
    for key in yaml_dict.keys():
        if type(yaml_dict[key]) == list:
            yaml_dict[key] = \
                [val if val != "None" else None for val in yaml_dict[key]]
        elif yaml_dict[key] == "None":
            yaml_dict[key] = None

    # Finally convert to our wrapper object form and return
    yaml_settings = YAMLSettings(yaml_dict)

    return yaml_settings


class YAMLSettings:
    """Wrapper object for settings stored in YAML file and opened with
    load_yaml_settings. Has attributes equivalent to keys in dict/YAML file.
    """
    def __init__(self, param_dict):
        for key, value in param_dict.items():
            setattr(self, key, value)

        self.param_dict = param_dict

    def __repr__(self):
        return self.param_dict.__repr__()
    
def broaden_cannon_fluxes(
    wls,
    spec_std_br,
    e_spec_std_br,
    target_delta_lambda,):
    """Broadens spectra using PyAstronomy.instrBroadGaussFast.

    TODO 1: option to take a pre-existing wavelength scale at the new
    resolution to interpolate onto.

    TODO 2: for optimal rigour, we should probably separately broaden spectra
    from each WiFeS arm, rather than doing it together here.

    Parameters
    ----------
    wls: 1D float array
        Wavelength vector of shape [n_px]

    spec_std_br, e_spec_std_br: 2D float array
        Flux and uncertainty arrays of shape [n_star, n_px]

    target_delta_lambda: float
        Broadening kernel in wavelength units (same as wls) from which we
        calculate R to pass to instrBroadGaussFast, which represents R at the
        mean wavelength of our wavelength scale.
    """
    n_std = spec_std_br.shape[0]

    # Determine mean wavelength
    mean_lambda = np.mean(wls)

    # Calculate target resolution given mean wavelength and target_delta_lambda
    res = mean_lambda / target_delta_lambda

    # Supersample?
    pass

    # New wl scale
    wls_new = np.arange(wls[0], wls[-1], target_delta_lambda/2)
    n_wl_new = len(wls_new)

    # Broaden
    spec_std_br_broad = np.zeros((n_std, n_wl_new))
    e_spec_std_br_broad = np.zeros((n_std, n_wl_new))

    for spec_i in tqdm(range(n_std), desc="Broadening", leave=False):
        nan_mask = np.logical_or(
            np.isnan(spec_std_br[spec_i]), np.isnan(e_spec_std_br[spec_i]))

        # Fluxes (broaden + interpolate)
        spec_broad = instrBroadGaussFast(
            wvl=wls[~nan_mask],
            flux=spec_std_br[spec_i][~nan_mask],
            resolution=res,
            equid=True,
            edgeHandling="firstlast",)
        
        interp_spec = interp1d(
            x=wls[~nan_mask],
            y=spec_broad,
            kind="cubic",
            bounds_error=False,)
        
        spec_std_br_broad[spec_i] = interp_spec(wls_new)

        # Uncertainties (broaden + interpolate)
        e_spec_broad = instrBroadGaussFast(
            wvl=wls[~nan_mask],
            flux=e_spec_std_br[spec_i][~nan_mask],
            resolution=res,
            equid=True,
            edgeHandling="firstlast",)
        
        interp_e_spec = interp1d(
            x=wls[~nan_mask],
            y=e_spec_broad,
            kind="cubic",
            bounds_error=False,)
        
        e_spec_std_br_broad[spec_i] = interp_e_spec(wls_new)

    return wls_new, spec_std_br_broad, e_spec_std_br_broad