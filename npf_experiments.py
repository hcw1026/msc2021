from functools import partial
import numpy as np
import os
import sys
import torch

npf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "Neural-Process-Family")
is_retrain = False
reval = True
is_reuse_across_epochs = True
starting_run = 1
is_valid = True


sys.path.remove(os.path.dirname(os.path.realpath(__file__)))
if npf_path not in sys.path:
    sys.path.append(npf_path)


from npf import AttnCNP, CNP, ConvCNP, CNPFLoss
from npf.architectures import MLP, merge_flat_input, CNN, MLP, ResConvBlock, SetConv, discard_ith_arg
from npf.utils.datasplit import (
    CntxtTrgtGetter,
    GetRandomIndcs,
    get_all_indcs,
)
from npf.neuralproc.base import LatentNeuralProcessFamily
from utils.data import cntxt_trgt_collate
from utils.data.helpers import DatasetMerger
from utils.helpers import count_parameters
from utils.ntbks_helpers import get_all_gp_datasets
from utils.train import train_models



gp_datasets, gp_test_datasets, gp_valid_datasets = get_all_gp_datasets(save_file=None, is_reuse_across_epochs=is_reuse_across_epochs)
get_cntxt_trgt_1d = cntxt_trgt_collate(
    CntxtTrgtGetter(
        contexts_getter=GetRandomIndcs(a=0.0, b=50), targets_getter=get_all_indcs,
    )
)
get_cntxt_trgt_1d_test = cntxt_trgt_collate(
    CntxtTrgtGetter(
        contexts_getter=GetRandomIndcs(a=0.0, b=50), targets_getter=get_all_indcs, indp_target=True
    )
)

def CNP_():
    R_DIM = 128
    KWARGS = dict(
        XEncoder=partial(MLP, n_hidden_layers=1, hidden_size=R_DIM),
        Decoder=merge_flat_input(  # MLP takes single input but we give x and R so merge them
            partial(MLP, n_hidden_layers=4, hidden_size=R_DIM), is_sum_merge=True,
        ),
        r_dim=R_DIM,
    )

    # 1D case
    model_1d = partial(
        CNP,
        x_dim=1,
        y_dim=1,
        XYEncoder=merge_flat_input(  # MLP takes single input but we give x and y so merge them
            partial(MLP, n_hidden_layers=2, hidden_size=R_DIM * 2), is_sum_merge=True,
        ),
        **KWARGS,
    )

    n_params_1d = count_parameters(model_1d())

    KWARGS = dict(
        is_retrain=is_retrain,  # whether to load precomputed model or retrain
        criterion=CNPFLoss,  # Standard loss for conditional NPFs
        chckpnt_dirname=os.path.join(npf_path, "results/pretrained/"),
        device=None,  # use GPU if available
        batch_size=32,
        lr=1e-3,
        decay_lr=10,  # decrease learning rate by 10 during training
        seed=123,
    )


    # 1D
    trainers_1d = train_models(
        gp_datasets,
        {"CNP": model_1d},
        test_datasets=gp_test_datasets,
        valid_datasets=gp_valid_datasets if is_valid else None,
        train_split=None,  # No need of validation as the training data is generated on the fly
        iterator_train__collate_fn=get_cntxt_trgt_1d,
        iterator_valid__collate_fn=get_cntxt_trgt_1d if is_valid else get_cntxt_trgt_1d_test,
        patience=10 if is_valid else None,
        max_epochs=100,
        starting_run=starting_run,
        **KWARGS
    )
    return trainers_1d


def AttnCNP_():
    R_DIM = 128
    KWARGS = dict(
        r_dim=R_DIM,
        attention="transformer",  # multi headed attention with normalization and skip connections
        XEncoder=partial(MLP, n_hidden_layers=1, hidden_size=R_DIM),
        Decoder=merge_flat_input(  # MLP takes single input but we give x and R so merge them
            partial(MLP, n_hidden_layers=4, hidden_size=R_DIM), is_sum_merge=True,
        ),
    )

    # 1D case
    model_1d = partial(
        AttnCNP,
        x_dim=1,
        y_dim=1,
        XYEncoder=merge_flat_input(  # MLP takes single input but we give x and y so merge them
            partial(MLP, n_hidden_layers=2, hidden_size=R_DIM), is_sum_merge=True,
        ),
        is_self_attn=False,
        **KWARGS,
    )

    n_params_1d = count_parameters(model_1d())

    KWARGS = dict(
        is_retrain=is_retrain,  # whether to load precomputed model or retrain
        criterion=CNPFLoss,
        chckpnt_dirname=os.path.join(npf_path, "results/pretrained/"),
        device=None,
        lr=1e-3,
        decay_lr=10,
        batch_size=32,
        seed=123,
    )

    # 1D
    trainers_1d = train_models(
        gp_datasets,
        {"AttnCNP": model_1d},
        test_datasets=gp_test_datasets,
        valid_datasets=gp_valid_datasets if is_valid else None,
        train_split=None,  # No need for validation as the training data is generated on the fly
        iterator_train__collate_fn=get_cntxt_trgt_1d,
        iterator_valid__collate_fn=get_cntxt_trgt_1d if is_valid else get_cntxt_trgt_1d_test,
        patience=10 if is_valid else None,
        max_epochs=100,
        starting_run=starting_run,
        **KWARGS
    )

    return trainers_1d

def ConvCNP_():
    R_DIM = 128
    KWARGS = dict(
        r_dim=R_DIM,
        Decoder=discard_ith_arg(  # disregards the target features to be translation equivariant
            partial(MLP, n_hidden_layers=4, hidden_size=R_DIM), i=0
        ),
    )


    CNN_KWARGS = dict(
        ConvBlock=ResConvBlock,
        is_chan_last=True,  # all computations are done with channel last in our code
        n_conv_layers=2,  # layers per block
    )


    # off the grid
    model_1d = partial(
        ConvCNP,
        x_dim=1,
        y_dim=1,
        Interpolator=SetConv,
        CNN=partial(
            CNN,
            Conv=torch.nn.Conv1d,
            Normalization=torch.nn.BatchNorm1d,
            n_blocks=5,
            kernel_size=19,
            **CNN_KWARGS,
        ),
        density_induced=64,  # density of discretization
        **KWARGS,
    )
    n_params_1d = count_parameters(model_1d())

    KWARGS = dict(
        is_retrain=is_retrain,  # whether to load precomputed model or retrain
        criterion=CNPFLoss,
        chckpnt_dirname=os.path.join(npf_path, "results/pretrained/"),
        device=None,
        lr=1e-3,
        decay_lr=10,
        seed=123,
        batch_size=32,
    )

    trainers_1d = train_models(
        gp_datasets,
        {"ConvCNP": model_1d},
        test_datasets=gp_test_datasets,
        valid_datasets=gp_valid_datasets if is_valid else None,
        iterator_train__collate_fn=get_cntxt_trgt_1d,
        iterator_valid__collate_fn=get_cntxt_trgt_1d if is_valid else get_cntxt_trgt_1d_test,
        patience=10 if is_valid else None,
        max_epochs=100,
        starting_run=starting_run,
        **KWARGS
    )

    return trainers_1d

def gen_p_y_pred(model, X_cntxt, Y_cntxt, X_trgt, n_samples):
    """Get the estimated (conditional) posterior predictive from a model."""

    if X_cntxt is None:
        X_cntxt = torch.zeros(1, 0, model.x_dim)
        Y_cntxt = torch.zeros(1, 0, model.y_dim)

    if isinstance(model, LatentNeuralProcessFamily):
        old_n_z_samples_test = model.n_z_samples_test
        model.n_z_samples_test = n_samples
        p_yCc, *_ = model.forward(X_cntxt, Y_cntxt, X_trgt)
        model.n_z_samples_test = old_n_z_samples_test

    else:
        # using CNPF
        p_yCc, *_ = model.forward(X_cntxt, Y_cntxt, X_trgt)

        if n_samples > 1:
            # using CNPF with many samples => sample from noise of posterior pred
            sampled_y = p_yCc.sample_n(n_samples).detach().numpy()
            return sampled_y[:,0,0,:,:], None

    mean_ys = p_yCc.base_dist.loc.detach().numpy()
    std_ys = p_yCc.base_dist.scale.detach().numpy()

    return mean_ys[0], std_ys[0]


def predict(trainer, X_cntxt, Y_cntxt, X_trgt, n_samples):
    model = trainer.module_
    model.eval()
    model = model.cpu()
    return gen_p_y_pred(model, X_cntxt, Y_cntxt, X_trgt, n_samples)

###################################################################################################
# Experiment
###################################################################################################
getter = CntxtTrgtGetter(
        contexts_getter=GetRandomIndcs(a=0.0, b=50), targets_getter=get_all_indcs, indp_target=True
    )

batch_size = 32
n_points = 128
save_dir = os.path.join(os.path.dirname(os.path.dirname(npf_path)), "Training/NP_experiments{}".format(starting_run))

all_trainers = {"CNP":CNP_(), "AttnCNP":AttnCNP_(), "ConvCNP":ConvCNP_()}

for trainers_name, trainers in all_trainers.items():
    for name, trainer in trainers.items():
        tr_input_ls = []
        tr_output_ls = []
        val_input_ls = []
        val_output_ls = []
        tr_mu_ls = []
        tr_sigma_ls = []
        val_mu_ls = []
        val_sigma_ls = []

        data_name, model_name = name.split("/")[0:2]
        dataset = gp_test_datasets[data_name]

        savepath = os.path.join(save_dir, data_name+"_"+model_name) + ".npz"
        if (not os.path.isfile(savepath)) and reval:
            if isinstance(dataset, DatasetMerger):
                X_all = torch.cat([d[:][0] for d in dataset.datasets], dim=0)
                y_all = torch.cat([d[:][1] for d in dataset.datasets], dim=0)
            else:
                X_all, y_all = dataset[:]

            samples = int(X_all.size(0))
            num_takes = samples//batch_size +1
            for i in range(num_takes):
                num = batch_size if i != (num_takes) - 1 else (samples % batch_size)
                X, y = X_all[(i*batch_size):(i*batch_size+num)], y_all[(i*batch_size):(i*batch_size+num)]
                X_cntxt, y_cntxt, X_trgt, y_trgt = getter(X, y)
                y_trgt_pred_mu, y_trgt_pred_sigma = predict(trainer, X_cntxt, y_cntxt, X_trgt, n_samples=1)
                y_cntxt_pred_mu, y_cntxt_pred_sigma = predict(trainer, X_cntxt, y_cntxt, X_cntxt, n_samples=1)

                tr_input_ls.append(X_cntxt.tolist())
                tr_output_ls.append(y_cntxt.tolist())
                val_input_ls.append(X_trgt.tolist())
                val_output_ls.append(y_trgt.tolist())

                tr_mu_ls.append(y_cntxt_pred_mu.tolist())
                tr_sigma_ls.append(y_cntxt_pred_sigma.tolist())
                val_mu_ls.append(y_trgt_pred_mu.tolist())
                val_sigma_ls.append(y_trgt_pred_sigma.tolist())

            output = dict(
                tr_input = np.array(tr_input_ls, dtype=object),
                tr_output = np.array(tr_output_ls, dtype=object),
                val_input = np.array(val_input_ls, dtype=object),
                val_output = np.array(val_output_ls, dtype=object),
                tr_mu = np.array(tr_mu_ls, dtype=object),
                tr_sigma = np.array(tr_sigma_ls, dtype=object),
                val_mu = np.array(val_mu_ls, dtype=object),
                val_sigma = np.array(val_sigma_ls, dtype=object)
            )

            print("saved at", savepath)
            np.savez(savepath, **output)