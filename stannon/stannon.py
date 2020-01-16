"""Stannon class to encapsulate Cannon functionality
"""
import os
import numpy as np
import stannon.stan_utils as sutils

class Stannon(object):
    """Class to encapsulate the Stan implementation of the Cannon, the Stannon
    """
    # Constants
    SUPPORTED_MODELS = ("basic", "label_uncertainties")
    SUPPORTED_N_LABELS = (3,)

    def __init__(self, training_data, training_labels, label_names, model_type,
                 training_variances=None):
        """Stannon class to encapsulate Cannon functionality.

        Parameters
        ----------
        training_data: 2D numpy array
            Training data (spectra) of shape [S spectra, P pixels]

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
        self.training_labels = training_labels
        self.label_names = label_names
        self.model_type = model_type
        self.training_variances = training_variances

        self.S, self.P = training_data.shape
        self.L = len(label_names)

        # Load in the model itself
        self.model = self.get_stannon_model()
        
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

    def train_cannon_model(self):
        """Train the Cannon model, with training per the model specified
        """
        if self.model_type == "basic":
            self._train_cannon_model_basic()
        
        elif self.model_type == "label_uncertainties":
            self._train_cannon_model_label_uncertainties()

        else:
            raise NotImplementedError("Error: this model has not yet been " 
                                      "implemented")


    def _train_cannon_model_basic(self):
        """
        """
        # Do model related things
        pass


    def _train_cannon_model_label_uncertainties(self):
        """
        """
        pass

    def test_cannon_model(self):
        """
        """
        pass