"""Does stochastic variational inference train the LSTM in variational_bnn/bnn.py?

Course staff added this check. The authors' Pyro model calls the LSTM encoder and
decoder inside ``model()`` without registering them with ``pyro.module``. Pyro's SVI
optimizer only steps parameters that are in Pyro's parameter store, so the LSTM
weights never change: the "Variational BNN" is a Bayesian linear read-out on top of
an untrained recurrent network. Registering the module is the one-line fix.

Run from the repository root, with torch, pyro-ppl and pandas installed:

    python staff/check_pyro_training.py

No data is needed; the check uses random tensors of the right shape.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pyro
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from variational_bnn.bnn import BayesianTrajectoryPredictor, VariationalLSTM

STEPS = 30


def run(register_lstm: bool):
    pyro.clear_param_store()
    torch.manual_seed(0)
    pyro.set_rng_seed(0)

    lstm = VariationalLSTM(input_dim=2, hidden_dim=128, pred_len=12, num_layers=2, dropout_p=0.1)
    predictor = BayesianTrajectoryPredictor(lstm, torch.device("cpu"))

    if register_lstm:
        # The fix: tell Pyro that the LSTM's parameters belong to the model.
        def model(obs_seq, gt=None):
            pyro.module("lstm", lstm)
            return predictor.model(obs_seq, gt)
    else:
        model = predictor.model  # as in the authors' train_variational()

    svi = SVI(model, predictor.guide, Adam({"lr": 1e-3}), Trace_ELBO())
    obs = torch.randn(64, 8, 2)
    gt = torch.randn(64, 12, 2)

    before = {name: p.detach().clone() for name, p in lstm.named_parameters()}
    lstm.train()
    for _ in range(STEPS):
        svi.step(obs, gt)

    moved = max((p - before[name]).abs().max().item() for name, p in lstm.named_parameters())
    w_mu_moved = pyro.param("w_mu").abs().max().item()  # initialised at zero
    n_sites = len(list(pyro.get_param_store().keys()))
    return moved, w_mu_moved, n_sites


if __name__ == "__main__":
    print(f"{STEPS} SVI steps on random data, largest absolute change in any weight\n")
    print(f"{'':34s} {'LSTM weights':>14s} {'output w_mu':>12s} {'param sites':>12s}")
    for label, flag in (("as committed (no pyro.module)", False), ("with pyro.module('lstm', lstm)", True)):
        moved, w_mu, n = run(flag)
        print(f"{label:34s} {moved:14.6f} {w_mu:12.4f} {n:12d}")
    print("\nIf the first row shows 0.000000 for the LSTM weights, SVI never trained the encoder or decoder.")
