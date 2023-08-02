"""Stannon class to encapsulate Cannon functionality
"""
import os
import copy
import numpy as np
from tqdm import tqdm
import pickle
import stannon.stan_utils as sutils
from datetime import datetime
import plumage.spectra as spec
from scipy.optimize import curve_fit
from stannon.vectorizer import polynomial as svp
from stannon.vectorizer import PolynomialVectorizer

class Stannon(object):
    """Class to encapsulate the Stan implementation of the Cannon, the Stannon
    """
    # Constants
    SUPPORTED_MODELS = ("basic", "label_uncertainties")
    ORDER = 2

    def __init__(self, training_data, training_data_ivar, training_labels, 
                 training_ids, label_names, wavelengths, model_type, 
                 training_variances=None, adopted_wl_mask=None, data_mask=None, 
                 bad_px_mask=None):
        """Stannon class to encapsulate Cannon functionality.

        Parameters
        ----------
        training_data: 2D numpy array
            Training data (spectra) of shape [S spectra, P pixels]

        training_data_ivar: 2D numpy array
            Training data inverse variances of shape [S spectra, P pixels]

        training_labels: 2D numpy array
            Labels for the training set, of shape [S spectra, L labels]

        training_ids: 

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
        self.training_ids = training_ids
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
        self.theta =  np.full((self.P, self.N_COEFF), np.nan)
        self.s2 = np.full(self.P, np.nan)

        # If we're using a model with label uncertainties, initialise our
        # 'true labels' vector that we fit for
        if model_type == "label_uncertainties":
            self.true_labels = np.full((self.S, self.L), np.nan)

        # Initialise cross validation results
        self.cross_val_labels = np.nan * np.ones_like(self.training_labels)

        # Load in the model itself
        self.model = self.get_stannon_model()

        # Initialise masking and whitening (note that this is redone each time
        # the Cannon is trained also)
        self.initialise_masking()
        self.whiten_labels()


    def __deepcopy__(self):
        """Deep copy our Stannon object.
        """
        new_sm = Stannon(
            training_data=copy.deepcopy(self.training_data),
            training_data_ivar=copy.deepcopy(self.training_data_ivar),
            training_labels=copy.deepcopy(self.training_labels),
            training_ids=copy.deepcopy(self.training_ids),
            label_names=copy.deepcopy(self.label_names),
            wavelengths=copy.deepcopy(self.wavelengths),
            model_type=copy.deepcopy(self.model_type),
            training_variances=copy.deepcopy(self.training_variances),
            adopted_wl_mask=copy.deepcopy(self.adopted_wl_mask),
            data_mask=copy.deepcopy(self.data_mask),
            bad_px_mask=copy.deepcopy(self.bad_px_mask),)

        # Update theta, s2, and results of cross validation
        new_sm.theta = copy.deepcopy(self.theta)
        new_sm.s2 = copy.deepcopy(self.s2)
        new_sm.cross_val_labels = copy.deepcopy(self.cross_val_labels)

        # And true labels vector if applicable
        if self.model_type == "label_uncertainties":
            new_sm.true_labels = copy.deepcopy(self.true_labels)

        return new_sm

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
    def training_ids(self):
        return self._training_ids

    @training_ids.setter
    def training_ids(self, value):
        # Check dimensions
        if len(value) != self.training_data.shape[0]:
            raise ValueError("Number of training IDs must be the same as the "
                             "first data dimension (i.e. # stars).")
        else:
            self._training_ids = value

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
    def true_labels(self):
        return self._true_labels

    @true_labels.setter
    def true_labels(self, value):
        if value.shape != (self.S, self.L):
            raise ValueError(
                "Dimensions of true_labels is incorrect, must have dimensions"
                " [n_star, n_label]")
        else:
            self._true_labels = np.array(value)

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

        # Construct the model name
        if self.model_type == "basic":
            model_name = f"cannon-{self.L:.0f}L-O2.stan"

        elif self.model_type == "label_uncertainties":
            model_name = f"cannon_model_uncertainties_{self.L:.0f}L-O2.stan"

        else:
            # Note that it shouldn't be possible to get to this position, but
            # force the check anyway
            raise ValueError("Not a valid model, see Stannon.SUPPORTED_MODELS")

        # Check the model file exists
        model_path = os.path.join(here, model_name)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Model file {} doesn't exist!".format(model_path))

        # All good, continue
        model = sutils.read(model_path)

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

        # Whiten the uncertainties, then convert back to variances
        # TODO: properly propagate logarithmic
        if self.training_variances is not None:
            self.whitened_label_vars = \
                (self.masked_variances**0.5 / self.std_labels)**2


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
        self.masked_data_ivar = \
            masked_data_ivar[self.data_mask][:, self.adopted_wl_mask]
        self.masked_wl = self.wavelengths[self.adopted_wl_mask]


    def train_cannon_model(
        self,
        suppress_stan_output=True,
        init_uncertainty_model_with_basic_model=False,
        suppress_training_output=False,
        max_iter=2000,):
        """Train the Cannon model, with training per the model specified.

        Note that only the *masked* data is used for this.

        Parameters
        ----------
        suppress_stan_output: bool, default: True
            Boolean flag for whether to suppress Stan output during training.
            If yes, output is suppressed and a progress bar is displayed. Set 
            to false for debugging purposes.

        init_uncertainty_model_with_basic_model: boolean, default: False
            Only applicable to Cannon models with label uncertainties. If true,
            we train a basic Cannon model using the same training sample and
            use the fitting theta and s2 vectors to initialise our label
            uncertainty model to cut down on training time.

        suppress_training_output: boolean, default: False
            Only applicable to basic Cannon model. Suppresses progress bar
            updates when training. Separate to suppressing Stan output.

        max_iter: int, default: 2000
            Maximum number of fitting iterations Stan will run. Also affects
            how frequently Stan prints fitting updates to stdout (every 1% of
            max_iter).
        """
        # Run training steps common to all models

        # Initialise masking
        self.initialise_masking()

        # Whiten labels
        self.whiten_labels()

        # Use local pixel count to account for potential pixel masking
        P = self.masked_data.shape[1]

        # Initialise output arrays
        self.theta = np.full((P, self.N_COEFF), np.nan)
        self.s2 = np.full(P, np.nan)

        if self.model_type == "label_uncertainties":
            self.true_labels = np.full((self.S, self.L), np.nan)

        # Now run specific training steps
        if self.model_type == "basic":
            self._train_cannon_model_basic(
                suppress_stan_output=suppress_stan_output,
                suppress_training_output=suppress_training_output,
                max_iter=max_iter,)
        
        elif self.model_type == "label_uncertainties":
            self._train_cannon_model_label_uncertainties(
                suppress_stan_output=suppress_stan_output,
                init_with_basic_model=init_uncertainty_model_with_basic_model,
                max_iter=max_iter,)

        else:
            raise NotImplementedError(
                "Error: this model has not yet been implemented")


    def _train_cannon_model_basic(
        self,
        suppress_stan_output,
        suppress_training_output=False,
        max_iter=2000,):
        """Trains a basic Cannon model with no uncertainties on label values.

        This model takes as input:
         - Normalised observed fluxes             [N_star, N_lambda]
         - Normalised observed flux variances     [N_star, N_lambda]
         - Design matrix                          [N_star, N_coeff]

        The model then fits for:
         - Model coefficient vector               [N_coeff, N_lambda]
         - Model intrinsic scatter vector         [N_lambda]

        Note that for this model we assemble the design matrix in python using
        the labels of our benchmark sample and pass it directly to our Stan 
        model, rather than passing in the label vector itself.

        Parameters
        ----------
        suppress_stan_output: bool
            Boolean flag for whether to suppress Stan output during training.
            If yes, output is suppressed and a progress bar is displayed. Set 
            to false for debugging purposes.

        suppress_training_output: boolean, default: False
            Suppresses progress bar updates when trainining. Separate to
            suppressing Stan output.

        max_iter: int, default: 2000
            Maximum number of fitting iterations Stan will run. Also affects
            how frequently Stan prints fitting updates to stdout (every 1% of
            max_iter).
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

        # Train pixel-by-pixel with no output whatsoever
        if suppress_stan_output and suppress_training_output:
            for px_i in range(n_px):
                # Suppress Stan output. This is dangerous!
                with sutils.suppress_output() as sm:
                    self._train_cannon_pixel_basic(
                        px_i, data_dict, init_dict, max_iter,)

        # Train pixel-by-pixel and show progress bar
        elif suppress_stan_output and not suppress_training_output:
            for px_i in tqdm(range(n_px), smoothing=0.2, desc="Training"):
                # Suppress Stan output. This is dangerous!
                with sutils.suppress_output() as sm:
                    self._train_cannon_pixel_basic(
                        px_i, data_dict, init_dict, max_iter,)

        # Train pixel-by-pixel and display stan logging for debugging purposes
        else:
            for px_i in range(n_px):
                print("\n", "-"*100,)
                print("Training pixel #{}/{}".format(px_i+1, n_px))
                print("-"*100,)
                self._train_cannon_pixel_basic(
                    px_i, data_dict, init_dict, max_iter,)


    def _train_cannon_pixel_basic(
        self,
        px_i,
        data_dict,
        init_dict,
        max_iter,):
        """Train a single pixel of a Cannon model *without* label
        uncertainties. Since the spectral pixels are all independent, we train
        the model one pixel at a time and so P (N_px) is always 1.

        Parameters
        ----------
        px_i: int
            The current spectral pixel to be trained.
        
        data_dict: dict
            Dictionary containing data to fit to:
             - data_dict["S"], int
             - data_dict["P"], int
             - data_dict["L"], int
             - data_dict["C"], int
             - data_dict["label_means"], array with [N_stars, N_labels]
             - data_dict["label_variances"], array with [N_stars, N_labels]

        init_dict: dict
            Dictionary containing initial guesses for our matrices we're
            fitting for. This contains:
             - init_dict["s2"], int
             - init_dict["theta"], array [N_coeff]

        max_iter: int
            Maximum number of fitting iterations Stan will run. Also affects
            how frequently Stan prints fitting updates to stdout (every 1% of
            max_iter).
        """
        # Update the data dictionary with flux and ivar values for the current
        # spectral pixel. Note that the basic Cannon model expects 1D arrays.
        data_dict.update(
            y=self.masked_data[:, px_i],
            y_var=1/self.masked_data_ivar[:, px_i])

        kwds = dict(data=data_dict, init=init_dict, iter=max_iter,)
        
        p_opt = self.model.optimizing(**kwds)

        # Update our master arrays with the fitted values
        if p_opt is not None:
            self.theta[px_i] = p_opt["theta"]
            self.s2[px_i] = p_opt["s2"]


    def _train_cannon_model_label_uncertainties(
        self,
        suppress_stan_output,
        init_with_basic_model=False,
        max_iter=2000,):
        """Trains a Cannon model with uncertainties on label values. For this
        model our spectral pixels are not entirely independent as we have a
        single global 'true' label matrix, and so we cannot train our Cannon
        model one pixel at a time.

        This model takes as input:
         - Normalised observed fluxes             [N_star, N_lambda]
         - Normalised observed flux variances     [N_star, N_lambda]
         - Whitened label values                  [N_star, N_label]
         - Whitened label variances               [N_star, N_label]

        The model then fits for:
         - Model coefficient vector               [N_coeff, N_lambda]
         - Model intrinsic scatter vector         [N_lambda]
         - 'True' label vector                    [N_star, N_label]

        Note that for this model pass Stan the label values and uncertainties
        and it assembles the design matrix--we do not do so ourselves in Python
        as we do for the basic model.

        Parameters
        ----------
        suppress_stan_output: bool
            Boolean flag for whether to suppress Stan output during training.
            If yes, output is suppressed. Set to false for debugging purposes.

        init_with_basic_model: boolean, default: False
            If true, we train a basic Cannon model using the same training 
            sample and use the fitting theta and s2 vectors to initialise our
            label uncertainty model to cut down on training time.

        max_iter: int, default: 2000
            Maximum number of fitting iterations Stan will run. Also affects
            how frequently Stan prints fitting updates to stdout (every 1% of
            max_iter).
        """
        # If we're initialising our theta and s2 arrays using those computed
        # from a basic Stannon model, we need to initialise train that model.
        if init_with_basic_model:
            print("Training basic Cannon model for theta initialisation...")
            basic_sm = self.__deepcopy__()
            basic_sm.model_type = "basic"
            basic_sm.model = basic_sm.get_stannon_model()
            basic_sm.train_cannon_model(
                suppress_stan_output=True,
                suppress_training_output=False,)

            # Grab theta and s2 from this model
            init_theta = basic_sm.theta
            init_s2 = basic_sm.s2
        
        # Otherwise we just initialise our vectors with zeroes
        else:
            # Initialise theta array with 1s in the first column, and zeroes
            # everywhere else.
            init_theta = np.hstack(
                [np.ones((self.P, 1)), np.zeros((self.P, self.N_COEFF-1))])

            # Intialise scatter array
            init_s2 = np.ones(self.P)

        # Initialise 'true' label array as just our whitened labels
        init_true_labels = self.whitened_labels.copy()

        # Intialise data dictionary (things our model fits *to*)
        data_dict = dict(
            S=self.S,
            P=self.P,
            L=self.L,
            label_means=self.whitened_labels,
            label_variances=self.whitened_label_vars,
            y=self.masked_data,                         # Possible .T needed
            y_var=1/self.masked_data_ivar,)             # Possible .T needed

        # Initialise our dictionary of fitted *for* matrices
        init_dict = dict(
            theta=init_theta,
            s2=init_s2,
            true_labels=init_true_labels,)

        kwds = dict(data=data_dict, init=init_dict, iter=max_iter,)

        # Hides Stan logging--default behaviour when working. Dangerous!
        if suppress_stan_output:
            with sutils.suppress_output() as sm:
                p_opt = self.model.optimizing(**kwds)

        # Don't suppress the output when debugging
        else:
            p_opt = self.model.optimizing(**kwds)

        # Finished training, save results
        self.theta = p_opt["theta"]
        self.s2 = p_opt["s2"]
        self.true_labels = p_opt["true_labels"]


    def infer_labels(self, test_data, test_data_ivars):
        """
        Use coefficients and scatters from a trained Cannon model to infer the 
        labels for a set of normalised spectra.

        Modified from: 
        github.com/annayqho/TheCannon/blob/master/TheCannon/infer_labels.py
        
        Parameters
        ----------
        test_data: float array
            Science fluxes to infer labels for, of shape [n_spectra, n_pixels].
            Must be normalised the same as training spectra.

        test_data_ivars: float array
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
            zip(tqdm(test_data, desc=lbl, leave=False), test_data_ivars)):
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
                1/self.training_data_ivar[bm_i][self.adopted_wl_mask]+self.s2)
            
            # Compute the difference
            diff = np.abs(
                self.training_data[bm_i][self.adopted_wl_mask] - spec_gen)

            # Sigma clip
            new_bad_px_mask[bm_i, self.adopted_wl_mask] = \
                diff > std_px*flux_sigma_to_clip

        # Save
        self.bad_px_mask = np.logical_or(self.bad_px_mask, new_bad_px_mask)


    def run_cross_validation(
        self,
        suppress_stan_output,
        init_uncertainty_model_with_basic_model,
        max_iter,
        show_timing=True,):
        """Runs leave-one-out cross validation for the current training set,
        and saves the N predicted labels as self.cross_val_labels.

        To do this, we initialise a new Stannon model with the data from N-1
        objects, train it, and use it to predict the value of the missing
        benchmark. We then repreat this for all N benchmarks.

        TODO: parallelise.

        Parameters
        ----------
        suppress_stan_output: bool
            Boolean flag for whether to suppress Stan output during training.
            If yes, output is suppressed. Set to false for debugging purposes.

        init_uncertainty_model_with_basic_model: boolean, default: False
            If true, we train a basic Cannon model using the same training 
            sample and use the fitting theta and s2 vectors to initialise our
            label uncertainty model to cut down on training time.

        max_iter: int
            Maximum number of fitting iterations Stan will run. Also affects
            how frequently Stan prints fitting updates to stdout (every 1% of
            max_iter).

        show_timing: boolean, default True
            Whether to show the duration of cross-validation when finished.
        """
        print("\n", "%"*100, "%"*100, sep="\n",)
        print("\n\t\t\t\t\tRunning cross validation\n",)
        print("%"*100, "%"*100, "\n", sep="\n",)
        start_time = datetime.now()

        # Initialise output array
        self.cross_val_labels = np.ones_like(self.training_labels) * np.nan

        # Do leave-one-out training and testing for all. To do this, we create
        # a duplicate object containing data on N-1 benchmarks, train, and test
        for std_i in range(self.S):
            print("\n\n", "-"*100, sep="",)
            print("Leave one out validation {:0.0f}/{:0.0f}".format(
                std_i+1, self.S+1),)
            print("-"*100,)

            # Make a mask with which we will mask out the std_i'th benchmark
            dm = np.full(self.S, True)

            # Mask out the standard to be left out
            dm[std_i] = False

            # Duplicate the object. Since the best approach is to initialise
            # the object with one fewer benchmark, the easiest solution is to
            # just initialise a new object rather than using the deep copy
            # function to make sure all the changes cascade properly. As such,
            # this new object will have S = N-1.
            cv_sm = Stannon(
                training_data=copy.deepcopy(self.training_data[dm]),
                training_data_ivar=copy.deepcopy(self.training_data_ivar[dm]),
                training_labels=copy.deepcopy(self.training_labels[dm]),
                training_ids=copy.deepcopy(self.training_ids[dm]),
                label_names=copy.deepcopy(self.label_names),
                wavelengths=copy.deepcopy(self.wavelengths),
                model_type=copy.deepcopy(self.model_type),
                training_variances=copy.deepcopy(self.training_variances[dm]),
                adopted_wl_mask=copy.deepcopy(self.adopted_wl_mask),
                data_mask=copy.deepcopy(self.data_mask[dm]),
                bad_px_mask=copy.deepcopy(self.bad_px_mask[dm]),)

            # Train this new Cannon model on N-1 benchmarks
            cv_sm.train_cannon_model(
                suppress_stan_output=suppress_stan_output,
                init_uncertainty_model_with_basic_model=\
                    init_uncertainty_model_with_basic_model,
                max_iter=max_iter,)

            # Predict labels for the missing standard using the newly trained
            # Cannon model.
            labels_pred, _, _ = cv_sm.infer_labels(
                self.masked_data[~dm],
                self.masked_data_ivar[~dm])

            # Add the predicted label to our master list of cross_val labels
            self.cross_val_labels[std_i] = labels_pred

        if show_timing:
            t_elapsed = datetime.now() - start_time
            print("\nValidation duration (hh:mm:ss.ms) {}".format(t_elapsed))


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
            "training_ids":self.training_ids,
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

        if self.model_type == "label_uncertainties":
            class_dict["true_labels"] = self.true_labels

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
            training_ids=class_dict["training_ids"],
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

        if class_dict["model_type"] == "label_uncertainties":
            if "true_labels" in class_dict:
                sm.true_labels = class_dict["true_labels"]
            else:
                print("Warning, using old model: no true_labels")

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
    """Prepares spectra for Cannon input using one of two normalisation methods
    """
    # Construct mask for emission regions - useful regions are *TRUE*
    adopted_wl_mask = spec.make_wavelength_mask(
        wls,
        mask_emission=True,
        mask_sky_emission=False,
        mask_edges=True,)

    # Enforce minimum and maximum wavelengths
    adopted_wl_mask = \
        adopted_wl_mask * (wls > wl_min_model) * (wls < wl_max_model)

    # Normalise using a Gaussian
    if do_gaussian_spectra_normalisation:
        # Convert uncertainties to inverse variances, get an initial bad pixel
        # msdk flagging nan pixels for each spectrum.
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