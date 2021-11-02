"""Stannon class to encapsulate Cannon functionality
"""
import os
import numpy as np
from tqdm import tqdm
import pickle
import stannon.stan_utils as sutils
from datetime import datetime
import plumage.spectra as spec
from scipy.optimize import curve_fit
from stannon.vectorizer import polynomial as svp
from stannon.vectorizer import PolynomialVectorizer
from numpy.polynomial.polynomial import Polynomial


class Stannon(object):
    """Class to encapsulate the Stan implementation of the Cannon, the Stannon
    """
    # Constants
    SUPPORTED_MODELS = ("basic", "label_uncertainties")
    ORDER = 2

    def __init__(self, training_data, training_data_ivar, training_labels, 
                 label_names, wavelengths, model_type, training_variances=None, 
                 adopted_wl_mask=None, data_mask=None, bad_px_mask=None):
        """Stannon class to encapsulate Cannon functionality.

        Parameters
        ----------
        training_data: 2D numpy array
            Training data (spectra) of shape [S spectra, P pixels]

        training_data_ivar: 2D numpy array
            Training data inverse variances of shape [S spectra, P pixels]

        training_labels: 2D numpy array
            Labels for the training set, of shape [S spectra, L labels]

        label_names: 1D array_like
            1D array of label/column names corresponding to training_labels. Of
            length L.

        wavelengths:

        model_type: string
            The kind of Cannon model to construct. Allowable values are 'basic'
            and 'label_uncertainties'.

        training_variances: 2D numpy array, optional
            Label variances corresponding to (and same as) training_labels. 
            Defaults to None if not using an appropriate model.

        adopted_wl_mask: 1D boolean numpy array, optional
            Used to mask out pixels during modelling, False = unused.

        data_mask: 1D boolean numpy array, optional
            Used to mask out training data during modelling, False = unused.

        """
        self.training_data = training_data
        self.training_data_ivar = training_data_ivar
        self.training_labels = training_labels
        self.label_names = label_names
        self.wavelengths = wavelengths
        self.model_type = model_type
        self.training_variances = training_variances
        self.adopted_wl_mask = adopted_wl_mask
        self.data_mask = data_mask
        self.bad_px_mask = bad_px_mask

        self.S, self.P = training_data[:,adopted_wl_mask].shape
        self.L = len(label_names)

        # Set the shape of our theta array
        lvec = svp.terminator(label_names, order=self.ORDER)
        self.N_COEFF = 1 + len(
            svp.parse_label_vector_description(lvec, label_names))

        # Initialise theta and scatter
        self.theta = np.nan * np.ones((self.P, self.N_COEFF))
        self.s2 = np.nan * np.ones(self.P)

        # Initialise cross validation results
        self.cross_val_labels = np.nan * np.ones_like(self.training_labels)

        # Load in the model itself
        self.model = self.get_stannon_model()

        # Initialise masking and whitening (note that this is redone each time
        # the Cannon is trained also)
        self.initialise_masking()
        self.whiten_labels()

    @property
    def training_data_ivar(self):
        return self._training_data_ivar

    @training_data_ivar.setter
    def training_data_ivar(self, value):
        # Check dimensions
        if value.shape != self.training_data.shape:
            raise ValueError("Shape of training data and label inverse var "
                             "inconsistent, must have same dimensions")
        else:
            self._training_data_ivar = value

    @property
    def training_labels(self):
        return self._training_labels

    @training_labels.setter
    def training_labels(self, value):
        # Check dimensions
        if value.shape[0] != self.training_data.shape[0]:
            raise ValueError("Shape of training data and label vectors "
                             "inconsistent, must have same first dimension")
        else:
            self._training_labels = value

    @property
    def wavelengths(self):
        return self._wavelengths

    @wavelengths.setter
    def wavelengths(self, value):
        # Check dimensions
        if len(value) != self.training_data.shape[1]:
            raise ValueError("Length of wavelength scale does not match axis 2"
                             " of training data.")
        else:
            self._wavelengths = value

    @property
    def label_names(self):
        return self._label_names

    @label_names.setter
    def label_names(self, value):
        if len(value) != self.training_labels.shape[1]:
            raise ValueError("Number of label names inconsistent with shape "
                             "of data array, must share second dimension")
        else:
            self._label_names = value

    @property
    def model_type(self):
        return self._model_type

    @model_type.setter
    def model_type(self, value):
        if value not in self.SUPPORTED_MODELS:
            raise ValueError("Not a valid model, see Stannon.SUPPORTED_MODELS "
                             "for allowed values")
        else:
            self._model_type = value

    @property
    def training_variances(self):
        return self._training_variances

    @training_variances.setter
    def training_variances(self, value):
        if value is None:
            raise ValueError("No training data variances provided")

        elif value.shape != self.training_labels.shape:
            raise ValueError("Shape of training data and data variances don't "
                             "match")
        else:
            self._training_variances = value

    @property
    def adopted_wl_mask(self):
        return self._adopted_wl_mask

    @adopted_wl_mask.setter
    def adopted_wl_mask(self, value):
        if value is None:
            self._adopted_wl_mask = np.full(self.training_data.shape[1], True)

        elif len(value) != self.training_data.shape[1]:
            raise ValueError("Dimensions of mask and training data don't match"
                             ", mask length must match 2nd data dimension")
        else:
            self._adopted_wl_mask = np.array(value).astype(bool)

    @property
    def data_mask(self):
        return self._data_mask

    @data_mask.setter
    def data_mask(self, value):
        if value is None:
            self._data_mask = np.full(self.training_data.shape[0], True)

        elif len(value) != self.training_data.shape[0]:
            raise ValueError("Dimensions of mask and training data don't match"
                             ", mask length must match 1st data dimension")
        else:
            self._data_mask = np.array(value).astype(bool)

    @property
    def bad_px_mask(self):
        return self._bad_px_mask

    @bad_px_mask.setter
    def bad_px_mask(self, value):
        if value is None:
            self._bad_px_mask = np.full(self.training_data.shape, False)

        elif value.shape != self.training_data.shape:
            raise ValueError("Dimensions of bad_px_mask is incorrect,"
                             "must be same as training_data.")
        else:
            self._bad_px_mask = np.array(value).astype(bool)

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        if value.shape != (self.P, self.N_COEFF):
            raise ValueError("Dimensions of theta is incorrect, must have "
                             "dimensions [n_px, n_coeff]")
        else:
            self._theta = np.array(value)

    @property
    def s2(self):
        return self._s2

    @s2.setter
    def s2(self, value):
        if len(value) != self.P:
            raise ValueError("Length of s2 incorrect, should be equal to n_px")
        else:
            self._s2 = np.array(value)

    @property
    def cross_val_labels(self):
        return self._cross_val_labels

    @cross_val_labels.setter
    def cross_val_labels(self, value):
        if value.shape != self.training_labels.shape:
            raise ValueError("Dimensions of cross_val_labels is incorrect,"
                             "must be equal to training_labels.")
        else:
            self._cross_val_labels = np.array(value)

    #--------------------------------------------------------------------------
    # Class functions
    #--------------------------------------------------------------------------
    def get_stannon_model(self):
        """Load in the Stan code for the designated model type
        """
        # Get present location
        here = os.path.dirname(__file__)

        if self.model_type == "basic":
            model_name = f"cannon-{self.L:.0f}L-O2.stan"

        elif self.model_type == "label_uncertainties":
            model_name = f"cannon_model_uncertainties.stan"

        else:
            # Note that it shouldn't be possible to get to this position, but
            # force the check anyway
            raise ValueError("Not a valid model, see Stannon.SUPPORTED_MODELS") 

        model = sutils.read(os.path.join(here, model_name))

        return model
    

    def whiten_labels(self):
        """Whiten the *masked* labels and variances
        """
        # Compute mean and standard deviation of the training set (+save)
        self.mean_labels = np.nanmean(self.masked_labels, axis=0)
        self.std_labels = np.nanstd(self.masked_labels, axis=0)

        # Whiten the labels and their variances (+save)
        self.whitened_labels = (self.masked_labels 
                                - self.mean_labels) / self.std_labels

        # Whiten the variances. TODO: check correct formalism.
        if self.training_variances is not None:
            self.whitened_label_vars = self.masked_variances / self.std_labels


    def initialise_masking(self):
        """Initialise the masked data and flux arrays
        """
        # Mask labels
        self.masked_labels = self.training_labels[self.data_mask]

        if self.training_variances is not None:
            self.masked_variances = self.training_variances[self.data_mask]

        # Apply bad pixel mask - set bad pixel flux to 1 and ivar to 0
        masked_data = self.training_data.copy()
        masked_data_ivar = self.training_data_ivar.copy()

        masked_data[self.bad_px_mask] = 1
        masked_data_ivar[self.bad_px_mask] = 1e-8

        # Now mask out wavelength regions we've excluded
        self.masked_data = masked_data[self.data_mask][:, self.adopted_wl_mask]
        self.masked_data_ivar = masked_data_ivar[self.data_mask][:, self.adopted_wl_mask]
        self.masked_wl = self.wavelengths[self.adopted_wl_mask]


    def train_cannon_model(self, suppress_output=True):
        """Train the Cannon model, with training per the model specified.

        Note that only the *masked* data is used for this.

        Parameters
        ----------
        suppress_output: bool
            Boolean flag for whether to suppress Stan output during training.
            If yes, output is suppressed and a progress bar is displayed. Set 
            to false for debugging purposes.
        """
        # Run training steps common to all models

        # Initialise masking
        self.initialise_masking()

        # Whiten labels
        self.whiten_labels()

        # Use local pixel count to account for potential pixel masking
        P = self.masked_data.shape[1]

        # Initialise output arrays
        self.theta = np.nan * np.ones((P, self.N_COEFF))
        self.s2 = np.nan * np.ones(P)

        # Now run specific training steps
        if self.model_type == "basic":
            self._train_cannon_model_basic(suppress_output)
        
        elif self.model_type == "label_uncertainties":
            self._train_cannon_model_label_uncertainties(suppress_output)

        else:
            raise NotImplementedError("Error: this model has not yet been " 
                                      "implemented")


    def _train_cannon_model_basic(self, suppress_output):
        """Trains a basic Cannon model with no uncertainties on label values.

        Parameters
        ----------
        suppress_output: bool
            Boolean flag for whether to suppress Stan output during training.
            If yes, output is suppressed and a progress bar is displayed. Set 
            to false for debugging purposes.
        """
        # Create design matrix
        self.vectorizer = PolynomialVectorizer(self.label_names, 2)
        self.design_matrix = self.vectorizer(self.whitened_labels)

        # Initialise theta array
        init_theta = np.atleast_2d(
            np.hstack([1, np.zeros(self.theta.shape[1]-1)]))

        # Intialise dictionaries to pass to stan
        data_dict = dict(P=1, S=self.S, DM=self.design_matrix.T)
        init_dict = dict(theta=init_theta, s2=1)

        # Grab our number of pixels
        n_px = self.masked_data.shape[1]

        # Suppress output - default behaviour when working. Hides Stan logging
        # and instead shows progress bar
        if suppress_output:
            for px_i in tqdm(range(n_px), smoothing=0.2, desc="Training"):
                # Suppress Stan output. This is dangerous!
                with sutils.suppress_output() as sm:
                    self._train_cannon_pixel(px_i, data_dict, init_dict,)

        # Don't suppress the output when debugging to show stan logging
        else:
            for px_i in range(n_px):
                self._train_cannon_pixel(px_i, data_dict, init_dict,)


    def _train_cannon_model_label_uncertainties(self, suppress_output):
        """Trains a Cannon model with uncertainties on label values.

        TODO: currently broken

        Parameters
        ----------
        suppress_output: bool
            Boolean flag for whether to suppress Stan output during training.
            If yes, output is suppressed and a progress bar is displayed. Set 
            to false for debugging purposes.
        """
        # Create design matrix
        self.vectorizer = PolynomialVectorizer(self.label_names, 2)
        self.design_matrix = self.vectorizer(self.whitened_labels)

        # Initialise theta array
        init_theta = np.atleast_2d(
            np.hstack([1, np.zeros(self.theta.shape[1]-1)]))

        # Intialise dictionaries to pass to stan
        data_dict = dict(
            S=self.S,
            P=1,
            L=self.L,
            C=self.N_COEFF,
            label_means=self.whitened_labels,
            label_variances=self.whitened_label_vars,
            design_matrix=self.design_matrix.T)

        init_dict = dict(
            true_labels=self.whitened_labels,
            s2=[1],
            theta=init_theta)

        # Grab our number of pixels
        n_px = self.masked_data.shape[1]

        # Suppress output - default behaviour when working. Hides Stan logging
        # and instead shows progress bar
        if suppress_output:
            for px_i in tqdm(range(n_px), smoothing=0.2, desc="Training"):
                # Suppress Stan output. This is dangerous!
                with sutils.suppress_output() as sm:
                    self._train_cannon_pixel(px_i, data_dict, init_dict,)

        # Don't suppress the output when debugging to show stan logging
        else:
            for px_i in range(n_px):
                self._train_cannon_pixel(px_i, data_dict, init_dict,)


    def _train_cannon_pixel(self, px_i, data_dict, init_dict,):
        """Train a single pixel of a Cannon model.
        """
        # Update the data dictionary with flux and ivar values for the current
        # spectral pixel
        if self.model_type == "label_uncertainties":
            data_dict.update(
                y=np.atleast_2d(self.masked_data[:, px_i]).T,
                y_var=np.atleast_2d(1/self.masked_data_ivar[:, px_i]).T)
        else:
            data_dict.update(
                y=self.masked_data[:, px_i],
                y_var=1/self.masked_data_ivar[:, px_i])

        kwds = dict(data=data_dict, init=init_dict)
        
        p_opt = self.model.optimizing(**kwds)
        if p_opt is not None:
                self.theta[px_i] = p_opt["theta"]
                self.s2[px_i] = p_opt["s2"]
        """
        try:
            p_opt = self.model.optimizing(**kwds)

        except:
            logging.exception("Exception occurred when optimizing"
                                f"pixel index {px_i}")
        else:
            if p_opt is not None:
                self.theta[px_i] = p_opt["theta"]
                self.s2[px_i] = p_opt["s2"]
        """


    def infer_labels(self, test_data, test_data_ivars):
        """
        Use coefficients and scatters from a trained Cannon model to infer the 
        labels for a set of normalised spectra.

        Modified from: 
        github.com/annayqho/TheCannon/blob/master/TheCannon/infer_labels.py
        
        Parameters
        ----------
        fluxes: float array
            Science fluxes to infer labels for, of shape [n_spectra, n_pixels].
            Must be normalised the same as training spectra.

        ivars: float array
            Science flux inverse variances, of shape [n_spectra, n_pixels].
             Must be normalised the same as training spectra.

        Returns
        -------
        labels_all: float array
            Cannon predicted labels (de-whitened) of shape [n_spectra, n_label]

        errs_all: float array
            ...

        chi2_all: float array
            Chi^2 fit for each star, vector of length [n_spectra].
        """
        # Initialise
        errs_all = np.zeros((len(test_data), self.L))
        chi2_all = np.zeros(len(test_data))
        labels_all = np.zeros((len(test_data), self.L))
        starting_guess = np.ones(self.L)

        lbl = "Inferring labels"

        for star_i, (flux, ivar) in enumerate(
            zip(tqdm(test_data, desc=lbl), test_data_ivars)):
            # Where the ivar == 0, set normalized flux to 1 and sigma to 100
            bad = ivar == 0
            flux[bad] = 1.0
            sigma = np.ones(ivar.shape) * 100.0
            sigma[~bad] = np.sqrt(1.0 / ivar[~bad])

            errbar = np.sqrt(sigma**2 + self.s2**2)

            try:
                labels, cov = curve_fit(multiply_coeff_label_vectors, 
                                        self.theta, flux, 
                                        p0=starting_guess, sigma=errbar, 
                                        absolute_sigma=True)
            except:
                labels = np.zeros(starting_guess.shape)*np.nan
                cov = np.zeros((len(starting_guess),
                                len(starting_guess))) * np.nan
                        
            chi2 = ((flux-multiply_coeff_label_vectors(self.theta, *labels))**2 
                    * ivar / (1 + ivar * self.s2**2))
            chi2_all[star_i] = sum(chi2)
            labels_all[star_i,:] = labels * self.std_labels + self.mean_labels
            errs_all[star_i,:] = (np.sqrt(cov.diagonal()) * self.std_labels)

        return labels_all, errs_all, chi2_all


    def make_sigma_clipped_bad_px_mask(self, flux_sigma_to_clip=5):
        """Generate a bad pixel mask via sigma clipping using a trained Cannon
        model.
        """
        # Initialise blank bad pixel mask
        new_bad_px_mask = np.full(self.training_data.shape, False)

        # For every star, check how many sigma the model is out
        for bm_i in range(len(self.training_labels)):
            # Generate a model spectrum
            labels = self.training_labels[bm_i]
            spec_gen = self.generate_spectra(labels)

            # Add the observed + model pixel uncertainty in quadrature. Note
            # std^2 is just the variance.
            std_px = np.sqrt(
                1/self.training_data_ivar[bm_i][self.adopted_wl_mask] + self.s2)
            
            # Compute the difference
            diff = np.abs(self.training_data[bm_i][self.adopted_wl_mask] - spec_gen)

            # Sigma clip
            new_bad_px_mask[bm_i, self.adopted_wl_mask] = diff > std_px*flux_sigma_to_clip

        # Save
        self.bad_px_mask = np.logical_or(self.bad_px_mask, new_bad_px_mask)


    def run_cross_validation(self, show_timing=True):
        """Runs leave-one-out cross validation for the current training set.

        TODO: parallelise
        """
        start_time = datetime.now()

        # Save trained theta and s2 values if we already have them
        trained_theta = self.theta.copy()
        trained_s2 = self.s2.copy()

        # Decrement number of training standards temporarily so we only train
        # on N-1 for the cross validation
        self.S -= 1

        # Initialise output array
        self.cross_val_labels = np.ones_like(self.training_labels) * np.nan

        # Keep initial masked data, as it'll get overwritten each loop
        masked_data = self.masked_data.copy()
        masked_data_ivar = self.masked_data_ivar.copy()

        # Do leave-one-out training and testing for all 
        for std_i in range(self.S+1):
            print("\nLeave one out validation {:0.0f}/{:0.0f}".format(
                std_i+1, self.S+1))

            # Make a mask
            self.data_mask = np.full(self.S+1, True)

            # Mask out the standard to be left out
            self.data_mask[std_i] = False

            # Run fitting
            self.train_cannon_model(suppress_output=True)

            # Predict labels for the missing standard
            labels_pred, _, _ = self.infer_labels(
                masked_data[~self.data_mask],
                masked_data_ivar[~self.data_mask])

            self.cross_val_labels[std_i] = labels_pred

        # Reset theta, s2, and S
        self.theta = trained_theta
        self.s2 = trained_s2
        self.S += 1

        if show_timing:
            time_elapsed = datetime.now() - start_time
            print("\nValidation duration (hh:mm:ss.ms) {}".format(time_elapsed))


    def generate_spectra(self, labels):
        """Generate spectra from a trained Cannon given stellar parameters.
        """
        # Whiten input labels
        labels = (labels - self.mean_labels) / self.std_labels

        # Construct the label vector and predict spectrum
        label_vector = get_lvec(labels)
        spec_gen = np.matmul(self.theta, label_vector)

        return spec_gen


    def save_model(self, path,):
        """Saves the stannon model. Currently just uses pickle for simplicity,
        but for persistence might want to consider fits as a future update.
        """
        # Create a dictionary of all class variables
        class_dict = {
            "training_data":self.training_data,
            "training_data_ivar":self.training_data_ivar,
            "training_labels":self.training_labels,
            "label_names":self.label_names,
            "wavelengths":self.wavelengths,
            "model_type":self.model_type,
            "training_variances":self.training_variances,
            "adopted_wl_mask":self.adopted_wl_mask,
            "data_mask":self.data_mask,
            "theta":self.theta,
            "s2":self.s2,
            "cross_val_labels":self.cross_val_labels,
            "bad_px_mask":self.bad_px_mask,
        }

        # Construct filename
        filename = os.path.join(
            path, 
            "stannon_model_{}_{}label_{}px_{}.pkl".format(
                self.model_type, self.L, self.P, "_".join(self.label_names)))

        # Dump our model to disk, and overwrite any existing file.
        with open(filename, 'wb') as output_file:
            pickle.dump(class_dict, output_file, pickle.HIGHEST_PROTOCOL)


#------------------------------------------------------------------------------
# Module Functions
#------------------------------------------------------------------------------
def load_model(filename):
    """Load a saved stannon model from file. Currently uses pickle, but in the
    future this may change.
    """
    # Input checking
    if not os.path.isfile(filename):
        raise FileNotFoundError("Pickle not found")

    # Import the saved pickle of class dict and remake object
    with open(filename, 'rb') as input_file:
        class_dict = pickle.load(input_file)

        # Remake object
        sm = Stannon(
            training_data=class_dict["training_data"],
            training_data_ivar=class_dict["training_data_ivar"],
            training_labels=class_dict["training_labels"], 
            label_names=class_dict["label_names"],
            wavelengths=class_dict["wavelengths"],
            model_type=class_dict["model_type"],
            training_variances=class_dict["training_variances"],
            adopted_wl_mask=class_dict["adopted_wl_mask"],
            data_mask=class_dict["data_mask"],
            bad_px_mask=class_dict["bad_px_mask"],)

        # Save theta, s2, and results of cross validation
        sm.theta = class_dict["theta"]
        sm.s2 = class_dict["s2"]
        sm.cross_val_labels = class_dict["cross_val_labels"]

    return sm


def prepare_fluxes(spec_br, e_spec_br,):
    """Prepare fluxes for use in Cannon model. Assume that we are receiving 
    previously combined blue and red arms.
    """
    training_set_flux = spec_br
    training_set_ivar = 1/e_spec_br**2
    
    # Get bad px mask
    bad_px_mask = np.logical_or(
        ~np.isfinite(training_set_flux),
        ~np.isfinite(training_set_ivar)
    )

    # If flux is nan, set to 1 and give high variance (inverse variance of 0)
    # training_set_ivar[bad_px_mask] = 1e-8
    # training_set_flux[bad_px_mask] = 1

    return training_set_flux, training_set_ivar, bad_px_mask


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


def prepare_synth_training_set(
    ref_spec, ref_params, sigma_pc=0.01,
    param_lims={"teff":None, "logg":None, "feh":None, "vsini":None},
    drop_vsini=True):
    """
    """
    PARAM_Is = {"teff":0, "logg":1, "feh":2, "vsini":3}

    # Make parameter cuts
    final_mask = np.full(len(ref_params), True)

    for param in param_lims:
        if param_lims[param] is not None:
            param_i = PARAM_Is[param]
            mask = np.logical_and(ref_params[:,param_i] >= param_lims[param][0],
                                ref_params[:,param_i] <= param_lims[param][1])
            final_mask = np.logical_and(final_mask, mask)
    
    ref_spec = ref_spec[final_mask]
    ref_params = ref_params[final_mask]

    if drop_vsini:
        ref_params = ref_params[:,:3]

    # Create artifical 'uncertainties' using a flat percentage
    sigma = ref_spec[:,1,:] * np.ones_like(ref_spec[:,0,:])*sigma_pc
    #sigma = np.swapaxes(np.atleast_3d(sigma), 1, 2)
    
    # Return params
    ref_wl = ref_spec[0,0,:]
    ref_fluxes = ref_spec[:,1]
    ref_ivar = 1/sigma**2
    #ref_spec = np.concatenate((ref_spec, sigma), axis=1)

    return ref_wl, ref_fluxes, ref_ivar, ref_params


def prepare_training_set(observations, spectra_b, spectra_r, std_params_all):
    """Need to prepare a list of labels corresponding to our science 
    observations. Easiest thing to do now is to just construct a new label
    dataframe with the same order as the observations. Then we won'd have any
    issues doing crossmatches, and we can worry about duplicates later.
    """
    # First thing to do is to select all the observations that are standards
    is_std_mask = np.isin(observations["uid"], std_params_all["source_id"])
    std_observations = observations.copy().iloc[is_std_mask]
    std_spectra_b = spectra_b[is_std_mask]
    std_spectra_r = spectra_r[is_std_mask]

    # Do something with the duplicates
    pass

    # Initialise new label dataframe
    std_params = std_params_all.copy()
    std_params.drop(std_params_all.index, inplace=True)

    # Now need to go through this one row at a time and construct the 
    # corresponding label vector
    for ob_i, source_id in enumerate(std_observations["uid"]):
        # Determine labels
        idx = std_params_all[std_params_all["source_id"]==source_id].index[0]
        std_params = std_params.append(std_params_all.loc[idx])

    return std_observations, std_spectra_b, std_spectra_r, std_params

def prepare_labels(
    obs_join,
    n_labels=3,
    e_teff_quad=60,
    max_teff=4200,
    abundance_labels=[],
    abundance_trends=None,):
    """Prepare our set of training labels using our hierarchy of parameter 
    source preferences.

    Teff: Prefer interferometric measurements, otherwise take the uniform Teff
    scale from Rains+21 which has been benchmarked to the interferometric Teff
    scale. Add Rains+21 uncertainties in quadrature with standard M+15 
    uncertainties to ensure that interferometric benchmarks are weighted more
    highly. Enforce a max Teff limit to avoid warmer stars.

    Logg: uniform Logg from Rains+21 (Mann+15 intial guess, updated from fit)

    [Fe/H]: Prefer CPM binary benchmarks, then M+15, then RA+12, then [Fe/H]
    from other NIR relations (e.g. T+15, G+14), then just default for Solar 
    Neighbourhood with large uncertainties.

    Parameters
    ----------
    TODO

    Returns
    -------
    TODO
    label_values, label_sigma, std_mask, label_sources
    """
    # Intialise mask
    std_mask = np.full(len(obs_join), True)

    # Initialise label vector
    label_values = np.full( (len(obs_join), n_labels), np.nan)
    label_sigma = np.full( (len(obs_join), n_labels), np.nan)

    # Initialise record of label source/s
    label_sources = np.full( (len(obs_join), n_labels), "").astype(object)

    # Go through one star at a time and select labels
    for star_i, (source_id, star_info) in enumerate(obs_join.iterrows()):
        # Only accept properly vetted stars with consistent Teffs
        if not star_info["in_paper"]:
            std_mask[star_i] = False
            continue

        # Only accept interferometric, M+15, RA+12, NIR other, & CPM standards
        elif not (~np.isnan(star_info["teff_int"]) 
            or ~np.isnan(star_info["teff_m15"])
            or ~np.isnan(star_info["teff_ra12"])
            or ~np.isnan(star_info["feh_nir"])
            or ~np.isnan(star_info["feh_cpm"])):
            std_mask[star_i] = False
            continue
        
        # Enforce our max temperature for interferometric standards
        elif star_info["teff_int"] > max_teff:
            std_mask[star_i] = False
            continue

        # Teff: interferometric > Rains+21
        if not np.isnan(star_info["teff_int"]):
            label_values[star_i, 0] = star_info["teff_int"]
            label_sigma[star_i, 0] = star_info["e_teff_int"]
            label_sources[star_i, 0] = star_info["int_source"]

        else:
            label_values[star_i, 0] = star_info["teff_synth"]
            label_sigma[star_i, 0] = (
                star_info["e_teff_synth"]**2 + e_teff_quad**2)**0.5
            label_sources[star_i, 0] = "R21"

        # logg: Rains+21
        label_values[star_i, 1] = star_info["logg_synth"]
        label_sigma[star_i, 1] = star_info["e_logg_synth"]
        label_sources[star_i, 1] = "R21"

        # [Fe/H]: CPM > M+15 > RA+12 > NIR other > Rains+21 > default
        if not np.isnan(star_info["feh_cpm"]):
            label_values[star_i, 2] = star_info["feh_cpm"]
            label_sigma[star_i, 2] = star_info["e_feh_cpm"]
            label_sources[star_i, 2] = star_info["source_cpm"]

        elif not np.isnan(star_info["feh_m15"]):
            label_values[star_i, 2] = star_info["feh_m15"]
            label_sigma[star_i, 2] = star_info["e_feh_m15"]
            label_sources[star_i, 2] = "M15"

        elif not np.isnan(star_info["feh_ra12"]):
            label_values[star_i, 2] = star_info["feh_ra12"]
            label_sigma[star_i, 2] = star_info["e_feh_ra12"]
            label_sources[star_i, 2] = "RA12"

        elif not np.isnan(star_info["feh_nir"]):
            label_values[star_i, 2] = star_info["feh_nir"]
            label_sigma[star_i, 2] = star_info["e_feh_nir"]
            label_sources[star_i, 2] = star_info["nir_source"]

        elif not np.isnan(star_info["phot_feh"]):
            label_values[star_i, 2] = star_info["phot_feh"]
            label_sigma[star_i, 2] = star_info["e_phot_feh"]
            label_sources[star_i, 2] = "R21"

        else:
            label_values[star_i, 2] = -0.14 # Mean for Solar Neighbourhood
            label_sigma[star_i, 2] = 2.0    # Default uncertainty

        # Note the adopted [Fe/H]
        feh_adopted = label_values[star_i, 2]

        # Other abundances
        for abundance_i, abundance in enumerate(abundance_labels):
            label_i = 3 + abundance_i

            # Use the abundance if we have it
            if not np.isnan(star_info[abundance]):
                label_values[star_i, label_i] = star_info[abundance]
                label_sigma[star_i, label_i] = star_info["e{}".format(abundance)]
                label_sources[star_i, label_i] = "M18"
            
            # Otherwise default to the solar neighbourhood abundance trend
            else:
                poly = Polynomial(abundance_trends[abundance].values)
                X_H = poly(feh_adopted)
                label_values[star_i, label_i] = X_H
                label_sigma[star_i, label_i] = 2.0

    return label_values, label_sigma, std_mask, label_sources

def prepare_cannon_spectra_normalisation(
    wls,
    spectra,
    e_spectra,
    wl_min_model=4000,
    wl_max_model=7000,
    wl_min_normalisation=4000,
    wl_broadening=50,
    do_gaussian_spectra_normalisation=True,
    poly_order=5,):
    """Prepares spectra for Cannon input using one of two normalisation methods.
    """
    # Construct mask for emission regions - useful regions are *TRUE*
    adopted_wl_mask = spec.make_wavelength_mask(
        wls,
        mask_emission=True,
        mask_sky_emission=False,
        mask_edges=True,)

    # Enforce minimum and maximum wavelengths
    adopted_wl_mask = adopted_wl_mask * (wls > wl_min_model) * (wls < wl_max_model)

    # Normalise using a Gaussian
    if do_gaussian_spectra_normalisation:
        # Convert uncertainties to inverse variances, get an initial bad pixel mask
        # flagging nan pixels for each spectrum.
        fluxes, flux_ivar, bad_px_mask = prepare_fluxes(
            spectra,
            e_spectra,)

        # Normalise training sample
        fluxes_norm, ivars_norm, continua = spec.gaussian_normalise_spectra(
            wl=wls,
            fluxes=fluxes,
            ivars=flux_ivar,
            adopted_wl_mask=adopted_wl_mask,
            bad_px_masks=bad_px_mask,
            wl_broadening=wl_broadening,)

    # Otherwise do polynomial normalisation
    else:
        spectra_norm, e_spec_norm = spec.normalise_spectra(
            wls,
            spectra,
            e_spectra,
            poly_order=poly_order,
            wl_min=wl_min_normalisation,)
        
        # And put in Cannon form
        fluxes_norm, ivars_norm, bad_px_mask = prepare_fluxes(
            spectra_norm,
            e_spec_norm,)

    # Return
    return fluxes_norm, ivars_norm, bad_px_mask, continua, adopted_wl_mask