from datetime import timedelta
import time
import numpy as np
from scipy import stats
from ml.models import mafs
import ml.step_strategies as ss
import ml.loss_functions as lf
from ml import trainers
import estimators.knn as knn

dtype = np.float32
rng = np.random


def MAF_entropy(samples, model=None):
    N, dim = samples.shape
    if model is None:
        model = create_MAF_model(dim)
        regularizer = lf.WeightDecay(model.parms, 1e-6)
        model = learn_model(samples=samples, NN_model=model, regularizer=regularizer)
    H = model.eval_trnloss(samples)
    return H, model


def MAF_KNN_entropy(samples, model=None):
    if model is None:
        _, model = MAF_entropy(samples, model)

    u = model.calc_random_numbers(samples)  # gaussian estimate

    correction = -np.mean(model.logdet_jacobi_u(samples))
    H = knn.knn_laplace(u)

    return H + correction, model


def uniformized_entropy(samples, model=None):
    if model is None:
        _, model = MAF_entropy(samples)
    u = model.calc_random_numbers(samples)  # gaussian estimate
    uniform = stats.norm.cdf(u)
    correction = -np.mean(np.log(np.prod(stats.norm.pdf(u), axis=1))) - np.mean(
        model.logdet_jacobi_u(samples)
    )
    H = knn.tkl(uniform)

    return H + correction, model


def create_MAF_model(n_inputs, n_hiddens=[200, 200], n_mades=10):
    """Generate a multi stage Masked Autoregressive Flow (MAF) model
    George Papamakarios, Theo Pavlakou, and Iain Murray. “Masked Autoregressive Flow for Density Estimation”

    Args:
        n_inputs (_type_): dimension of the input sample
        n_hiddens (list, optional): number of hidden layers and hidden nodes per MAF stage. Defaults to [100,100].
        n_mades (int, optional): number of MAF stages. Defaults to 14.

    Returns:
        _type_: MAF model
    """
    act_fun = "tanh"
    rng = np.random

    return mafs.MaskedAutoregressiveFlow(
        n_inputs=n_inputs,
        n_hiddens=n_hiddens,
        act_fun=act_fun,
        n_mades=n_mades,
        input_order="random",
        mode="random",
        rng=rng,
    )


def learn_model(
    sim_model=None,
    samples=None,
    n_samples=1000,
    NN_model=None,
    mini_batch=256,
    fine_tune=True,
    step=ss.Adam(),
    regularizer=None,
    val_tol=None,
    patience=30,
):
    """Create a MAF model and train it with the given parameters

    Args:
        sim_model (_type_): model to generate points from target distribution
        pretrained_model (MaskedAutoregressiveFlow, optional): pretrained neural net model. Create new model with random weights if set to none. Default to none
        n_samples (int, optional): number of samples to train on. Scaled by sim_model dimension. Defaults to 100.
        val_tol (float, optional): validation tolerance threshold to decide if model has improved. Defaults to 0.001.
        patience (int, optional): number of epochs without improvement before exiting training. Defaults to 5.
        n_hiddens (list, optional): number of hidden layers and nodes in a list. Defaults to [200,200].
        n_stages (int, optional): number of MAF stages. Defaults to 14.
        mini_batch (int, optional): Batch size for training. Defaults to 1024
        fine_tune (bool, optional): Set to True to run training twice, first with large step size, then a smaller step size. Defaults to True.
        show_progress (bool,optional): Set to true to print training curve. Defaults to False

    Returns:
        entropy.UMestimator: estimator object used for training and entropy calculation
    """
    # Generate model and samples if needed

    samples = sim_model.sim(n_samples) if samples is None else samples
    samples = np.asarray(samples, dtype)
    n_samples, dim = samples.shape

    NN_model = create_MAF_model(dim) if NN_model is None else NN_model

    # shuffle data
    idx = rng.permutation(n_samples)
    samples = samples[idx]

    # split into train and validation sets
    n_train = int(0.95 * n_samples)
    train_samp, val_samp = samples[:n_train], samples[n_train:]

    monitor_every = min(1e5 / float(n_samples), 1.0)

    start_time = time.time()
    trainer = trainers.SGD(
        NN_model,
        [train_samp],
        NN_model.trn_loss if regularizer is None else NN_model.trn_loss + regularizer,
        val_data=[val_samp],
        val_loss=NN_model.trn_loss,
        step=step,
    )

    trainer.train(
        minibatch=mini_batch,
        monitor_every=monitor_every,
        patience=patience,
        val_Tol=val_tol,
        fine_tune=fine_tune,
    )
    print(f"learning time: {timedelta(seconds=int(time.time() - start_time))}")
    print(f"Final Loss: {NN_model.eval_trnloss(samples):.3f}")

    return NN_model
