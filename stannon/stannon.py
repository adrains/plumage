"""Stannon class to encapsulate Cannon functionality
"""
import os
import numpy as np
import matplotlib.pyplot as plt
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
                 label_names, wavelengths, model_type, training_variances=None, 
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

        wavelengths:

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
        self.wavelengths = wavelengths
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

    def initialise_pixel_mask(self, px_min, px_max):
        """
        """
        self.pixel_mask = np.zeros(self.P, dtype=bool)
        self.pixel_mask[px_min:px_max] = True

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
        self.masked_wl = self.wavelengths[self.pixel_mask]

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
            labels_all[star_i,:] = labels * self.std_labels + self.mean_labels
            errs_all[star_i,:] = (np.sqrt(cov.diagonal()) * self.std_labels
                                  + self.mean_labels)

        return labels_all, errs_all, chi2_all

    
    def _train_cannon_model_label_uncertainties(self):
        """
        """
        pass

    def test_cannon_model(self):
        """
        """
        pass


    def plot_label_comparison(self, label_values, labels_pred):
        """
        """
        plt.close("all")
        # Plot Teff comparison
        plt.figure()
        plt.scatter(label_values[:,0],labels_pred[:,0], c=label_values[:,2],marker="o")
        plt.plot(np.arange(2500,5500),np.arange(2500,5500),"-",color="black")
        plt.xlabel(r"T$_{\rm eff}$ (Lit)")
        plt.ylabel(r"T$_{\rm eff}$ (Cannon)")
        cb = plt.colorbar()
        cb.set_label(r"[Fe/H]")
        plt.xlim([2800,5100])
        plt.ylim([2800,5100])
        plt.savefig("plots/presentations/ms_teff_vs_teff.png",fpi=300)

        # Plot logg comparison
        plt.figure()
        plt.scatter(label_values[:,1],labels_pred[:,1], c=label_values[:,0],marker="o")
        plt.plot(np.arange(2.5,5.5,0.1),np.arange(2.5,5.5,0.1),"-",color="black")
        plt.xlim([2.5,5.1])
        plt.ylim([2.5,5.1])
        plt.ylabel(r"$\log g$ (Cannon)")
        plt.xlabel(r"$\log g$ (Lit)")
        cb = plt.colorbar()
        cb.set_label(r"[Fe/H]")
        plt.savefig("plots/presentations/ms_logg_vs_logg.png",fpi=300)

        # Plot Fe/H comparison
        plt.figure()
        plt.scatter(label_values[:,2],labels_pred[:,2], c=label_values[:,0],marker="o",
                    cmap="magma") 
        plt.plot(np.arange(-0.6,0.5,0.05),np.arange(-0.6,0.5,0.05),"-",color="black")
        plt.xlabel(r"[Fe/H] (Lit)")
        plt.ylabel(r"[Fe/H] (Cannon)") 
        cb = plt.colorbar() 
        cb.set_label(r"T$_{\rm eff}$")
        plt.savefig("plots/presentations/ms_feh_vs_feh.png",fpi=300)


    def plot_theta_coefficients(self):
        # Plot of theta coefficients
        fig, axes = plt.subplots(4, 1, sharex=True)
        axes = axes.flatten()

        for star in self.masked_data:
            axes[0].plot(self.masked_wl, star, linewidth=0.2)

        axes[1].plot(self.masked_wl, self.theta[:,1], linewidth=0.5)
        axes[2].plot(self.masked_wl, self.theta[:,2], linewidth=0.5)
        axes[3].plot(self.masked_wl, self.theta[:,3], linewidth=0.5)

        axes[0].set_ylim([0,3])
        axes[1].set_ylim([-0.5,0.5])
        axes[2].set_ylim([-0.5,0.5])
        axes[3].set_ylim([-0.1,0.1])

        axes[0].set_ylabel(r"Flux")
        axes[1].set_ylabel(r"$\theta$ T$_{\rm eff}$")
        axes[2].set_ylabel(r"$\theta$ $\log g$")
        axes[3].set_ylabel(r"$\theta$ $[Fe/H]$")
        plt.xlabel("Wavelength (A)")

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

