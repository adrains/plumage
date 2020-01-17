"""Stannon class to encapsulate Cannon functionality
"""
import os
import numpy as np
from tqdm import tqdm
import stannon.stan_utils as sutils
from scipy.optimize import curve_fit
from stannon.vectorizer import PolynomialVectorizer


class Stannon(object):
    """Class to encapsulate the Stan implementation of the Cannon, the Stannon
    """
    # Constants
    SUPPORTED_MODELS = ("basic", "label_uncertainties")
    SUPPORTED_N_LABELS = (3,)

    def __init__(self, training_data, training_data_ivar, training_labels, 
                 label_names, model_type, training_variances=None, 
                 pixel_mask=None):
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

        model_type: string
            The kind of Cannon model to construct. Allowable values are 'basic'
            and 'label_uncertainties'.

        training_variances: 2D numpy array, optional
            Label variances corresponding to (and same as) training_labels. 
            Defaults to None if not using an appropriate model.
        """
        self.training_data = training_data
        self.training_data_ivar = training_data_ivar
        self.training_labels = training_labels
        self.label_names = label_names
        self.model_type = model_type
        self.training_variances = training_variances
        self.pixel_mask = pixel_mask

        self.S, self.P = training_data.shape
        self.L = len(label_names)

        # Load in the model itself
        self.model = self.get_stannon_model()

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

        elif value.shape[1] not in self.SUPPORTED_N_LABELS:
            raise ValueError("Unsupported number of training set labels, see "
                             "Stannon.SUPPORTED_N_LABELS for allowed values")
        else:
            self._training_labels = value

    @property
    def label_names(self):
        return self._label_names

    @label_names.setter
    def label_names(self, value):
        if len(value) != self.training_labels.shape[1]:
            raise ValueError("Number of label names inconsistent with shape "
                             "of data array, must share second dimension")
        
        # Not sure it is possible to trigger this, but in here for safety
        elif len(value) not in self.SUPPORTED_N_LABELS:
            raise ValueError("Unsupported number of training set labels, see "
                             "Stannon.SUPPORTED_N_LABELS for allowed values")
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
        # Meaningless having training set label variances if not taking into
        # account label uncertainties --> set to None
        if self.model_type != "label_uncertainties":
            self._training_variances = None

        elif value is None:
            raise ValueError("No training data variances provided")

        elif value.shape != self.training_labels.shape:
            raise ValueError("Shape of training data and data variances don't "
                             "match")
        else:
            self._training_variances = value

    @property
    def pixel_mask(self):
        return self._pixel_mask

    @pixel_mask.setter
    def pixel_mask(self, value):
        if value is None:
            self._pixel_mask = np.full(self.training_data.shape[1], True)

        elif len(value) != self.training_data.shape[1]:
            raise ValueError("Dimensions of mask and training data don't match"
                             ", mask length must match 2nd data dimension")
        else:
            self._pixel_mask = np.array(value).astype(bool)

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
            model_name = f"cannon-{self.L:.0f}L-O2-many-pixels.stan"

        else:
            # Note that it shouldn't be possible to get to this position, but
            # force the check anyway
            raise ValueError("Not a valid model, see Stannon.SUPPORTED_MODELS") 

        model = sutils.read(os.path.join(here, model_name))

        return model
    

    def whiten_labels(self):
        """Whiten the labels and variances
        """
        # Compute mean and standard deviation of the training set (+save)
        self.mean_labels = np.nanmean(self.training_labels, axis=0)
        self.std_labels = np.nanstd(self.training_labels, axis=0)

        # Whiten the labels and their variances (+save)
        self.whitened_labels = (self.training_labels 
                                - self.mean_labels) / self.std_labels

        if self.training_variances is not None:
            self.whitened_label_vars = (self.training_variances 
                                        - self.mean_labels) / self.std_labels


    def train_cannon_model(self, suppress_output=True):
        """Train the Cannon model, with training per the model specified

        Parameters
        ----------
        suppress_output: bool
            Boolean flag for whether to suppress Stan output during training.
            If yes, output is suppressed and a progress bar is displayed. Set 
            to false for debugging purposes.
        """
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
        # Mask data
        self.masked_data = self.training_data[:, self.pixel_mask]
        self.masked_data_ivar = self.training_data_ivar[:, self.pixel_mask]

        # Use local pixel count accounting for masking
        P = self.masked_data.shape[1]

        # Setup Cannon
        self.vectorizer = PolynomialVectorizer(self.label_names, 2)
        self.design_matrix = self.vectorizer(self.whitened_labels)

        self.theta = np.nan * np.ones((P, 10))
        self.s2 = np.nan * np.ones(P)

        init_theta = np.atleast_2d(
            np.hstack([1, np.zeros(self.theta.shape[1]-1)]))

        data_dict = dict(P=1, S=self.S, DM=self.design_matrix.T)
        init_dict = dict(theta=init_theta, s2=1)

        # Suppress output - default behaviour when working. Hides Stan logging
        # and instead shows progress bar
        if suppress_output:
            for px in tqdm(range(P), desc="Training"):
                data_dict.update(y=self.masked_data[:, px],
                                 y_var=1/self.masked_data_ivar[:, px])

                kwds = dict(data=data_dict, init=init_dict)

                # Suppress Stan output. This is dangerous!
                with sutils.suppress_output() as sm:
                    try:
                        p_opt = self.model.optimizing(**kwds)
                    except:
                        logging.exception("Exception occurred when optimizing"
                                          f"pixel index {px}")
                    else:
                        if p_opt is not None:
                            self.theta[px] = p_opt["theta"]
                            self.s2[px] = p_opt["s2"]

        # Don't suppress the output when debugging to show stan logging
        else:
            for px in range(P):
                data_dict.update(y=self.masked_data[:, px],
                                 y_var=1/self.masked_data_ivar[:, px])

                kwds = dict(data=data_dict, init=init_dict)

                try:
                    p_opt = self.model.optimizing(**kwds)
                except:
                    logging.exception("Exception occurred when optimizing"
                                      f"pixel index {px}")
                else:
                    if p_opt is not None:
                        self.theta[px] = p_opt["theta"]
                        self.s2[px] = p_opt["s2"]


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
            labels_all[star_i,:] = labels * self.mean_labels + self.std_labels
            errs_all[star_i,:] = (np.sqrt(cov.diagonal()) * self.mean_labels
                                  + self.std_labels)

        return labels_all, errs_all, chi2_all


    def _train_cannon_model_label_uncertainties(self):
        """
        """
        pass

    def test_cannon_model(self):
        """
        """
        pass

#------------------------------------------------------------------------------
# Module Functions
#------------------------------------------------------------------------------
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