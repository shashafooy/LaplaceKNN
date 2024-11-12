# import cPickle as pickle
import pickle
import os
import re
import sys
import numpy as np
from estimators.maf import create_MAF_model

dtype = np.float32


def save(data, file):
    """
    Saves data to a file.
    """

    dir = os.path.dirname(file)
    if dir:
        make_folder(dir)

    with open(file + ".pkl", "wb") as f:
        pickle.dump(data, f)


def load(file):
    """
    Loads data from file.
    """

    with open(file + ".pkl", "rb") as f:
        data = pickle.load(f)

    if hasattr(data, "reset_theano_functions"):
        data.reset_theano_functions()

    return data


def save_txt(str, file):
    """
    Saves string to a text file.
    """

    dir = os.path.dirname(file)
    if dir:
        make_folder(dir)

    with open(file, "w") as f:
        f.write(str)


def load_txt(file):
    """
    Loads string from text file.
    """

    with open(file, "r") as f:
        str = f.read()

    return str


def save_txt_from_numpy(data, file):
    dir = os.path.dirname(file)
    if dir:
        make_folder(dir)

    np.savetxt(file, data, fmt="%f", delimiter=" ")


def make_folder(folder):
    """
    Creates given folder (or path) if it doesn't exist.
    """

    if not os.path.exists(folder):
        os.makedirs(folder)


def update_filename(path, old_name, iter=-1, rename=True):
    reg_pattern = r"\(-?\d{1,3}_iter\)"
    iter_name = "({}_iter)".format(iter)
    match = re.search(reg_pattern, old_name)
    # Replace iteration number, append if it doesn't exist
    if match:
        new_name = re.sub(reg_pattern, iter_name, old_name)
    else:
        new_name = old_name + iter_name

    # Attach unique pid to filename
    if str(os.getpid()) not in new_name:
        new_name = new_name + "_" + str(os.getpid())

    os.makedirs(path, exist_ok=True)
    open(os.path.join(path, new_name + ".pkl"), "a").close()
    if rename:
        os.rename(os.path.join(path, old_name + ".pkl"), os.path.join(path, new_name + ".pkl"))
    return new_name


class Logger:
    """
    Implements an object that logs messages to a file, as well as printing them on the sceen.
    """

    def __init__(self, filename):
        """
        :param filename: file to be created for logging
        """
        self.f = open(filename, "w")

    def write(self, msg):
        """
        :param msg: string to be logged and printed on screen
        """
        sys.stdout.write(msg)
        self.f.write(msg)

    def __enter__(self):
        """
        Context management enter function.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context management exit function. Closes the file.
        """
        self.f.close()
        return False


def save_model(model, name, path="temp_data/saved_models"):
    parms = [np.empty_like(p.get_value()) for p in model.parms]
    masks = [np.empty_like(m.get_value()) for m in model.masks]
    for i, p in enumerate(model.parms):
        parms[i] = p.get_value().copy()
    for i, m in enumerate(model.masks):
        masks[i] = m.get_value().copy()
    save([parms, masks, model.n_inputs, model.n_hiddens, model.n_mades], os.path.join(path, name))

    # util.io.save((model),os.path.join(path,name))


def load_model(model=None, name="model_name", path="temp_data/saved_models", sim_model=None):

    # model = util.io.load(os.path.join(path,name))
    # return model
    # model = create_model(target_model.n_inputs, n_hiddens = target_model.n_hiddens, n_mades = target_model.n_mades)
    try:
        params, masks, n_inputs, n_hiddens, n_mades = load(os.path.join(path, name))
    except FileNotFoundError:
        print(f"model {name} not found at {path}")
        return None

    model = (
        create_MAF_model(n_inputs, n_hiddens=n_hiddens, n_mades=n_mades) if model is None else model
    )

    assert len(params) == len(
        model.parms
    ), "number of parameters is not the same, likely due to different number of stages"
    assert (
        params[0].shape[0] == model.parms[0].get_value().shape[0]
    ), f"invalid model input dimension. Expected {model.parms[0].get_value().shape[0]}, got {params[0].shape[0]}"
    assert (
        params[0].shape[1] == model.parms[0].get_value().shape[1]
    ), f"invalid model, number of nodes per hidden layer. Expected {model.parms[0].get_value().shape[1]}, got {params[0].shape[1]}"

    for i, p in enumerate(params):
        model.parms[i].set_value(p.astype(dtype))

    for i, m in enumerate(masks):
        model.masks[i].set_value(m.astype(dtype))

    return model
