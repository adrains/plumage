
import os
import logging
import pickle

import pystan as stan
import pystan.plots as plots

from tempfile import mkstemp

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
