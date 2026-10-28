<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/banner-dark.png">
  <img alt="The University of Chicago, MS in Applied Data Science. ADSP 32014 course project." src=".github/assets/banner-light.png">
</picture>

# Bayesian Trajectory Prediction for Autonomous Driving Safety

> **Example project from Winter 2026.** This is a copy of [khadijahslawal/bayesian-trajectory-prediction](https://github.com/khadijahslawal/bayesian-trajectory-prediction), shown with the authors' work and commit history intact. Their own write-up is [REPORT.md](REPORT.md). Course staff wrote this README and made the edits listed under [Changes by course staff](#changes-by-course-staff), including the section on [what to learn from this project](#what-to-learn-from-this-project) and a short script, `staff/check_pyro_training.py`, that demonstrates its first point. No data is in this repository.

A car that shares the road with pedestrians has to guess where each of them will be a few seconds from now, and one guess is not enough: a point estimate looks the same whether the pedestrian is walking a straight line or hesitating at a kerb. Can a trajectory model say how sure it is, and can a car use that to choose between proceeding and yielding? The team trained an LSTM on the ETH/UCY pedestrian benchmark to predict 4.8 seconds of movement from 3.2 seconds of history, then built two Bayesian versions of it: one that keeps dropout switched on at inference and averages 50 stochastic passes, and one whose output layer has a variational posterior over its weights, fit with Pyro. A notebook turns the spread of the 50 predicted paths into a safety score and a proceed, slow or yield decision.

## Summary

The data are the ETH and UCY pedestrian recordings in the text format redistributed by Trajectron++: frame, pedestrian id, x and y in metres, at 2.5 Hz. Every run of 20 consecutive annotated frames of one pedestrian is a sample, 8 observed steps and 12 to predict, with the observed steps expressed relative to the first of them and the targets relative to the last observed position. Five scenes, ETH, Hotel, Univ, Zara1 and Students001, supply 15,328 training windows and 2,720 validation windows.

**Baseline.** A two-layer LSTM encoder with 128 hidden units reads the 8 steps; an LSTM cell decodes 12 steps one at a time, feeding each prediction back as the next input, with a linear layer to x and y. Trained for 100 epochs with Adam on the mean Euclidean displacement, which is the ADE metric itself.

**MC dropout.** The same network with dropout at 0.3 on the hidden state after encoding and at every decoder step. At inference the dropout layers stay in training mode, 50 passes are averaged, and their variance is the uncertainty.

**Variational output layer.** The same network again, with the 258 weights of the final linear layer given a Normal(0, 1) prior and a factorized Normal variational posterior, fit by stochastic variational inference on an ELBO with a fixed observation noise of 0.1 m. Fifty weight samples give 50 paths.

**Safety framework.** On one batch of 32 windows, the mean predictive variance over the 12 steps is thresholded into SAFE, CAUTION and UNSAFE, drawn as fans of 50 paths, checked against actual error in a calibration plot, and illustrated as a crossing scenario in which a car proceeds past a confident prediction and yields to an uncertain one.

The headline numbers: the baseline and MC dropout are tied at an average error of 0.42 m and a final error of 0.89 m on the validation windows; the variational model is worse, 0.58 m and 1.03 m, with a far smaller variance, which the write-up reads as posterior collapse and a reason to prefer MC dropout. The staff section explains why the variational model's LSTM was never trained, why no test evaluation took place, and why the trajectory plots are not pictures of trajectories.

## Team and contributions

Team "Guardians of the Graph", Winter 2026.

- **Khadija Shuaib** ([@khadijahslawal](https://github.com/khadijahslawal)): everything on the main branch, in 22 commits: the data loader, the three model scripts, the safety notebook and its plots, the architecture and intuition figures, and the write-up. She appears as Khadijah in the commit history.
- **Amulya Rayasam**: two further variational implementations on the team's `archived` branch, a Bayes-by-backprop LSTM decoder cell with an annealed KL weight, and a Pyro model with an automatic diagonal-Normal guide and heteroscedastic outputs, both evaluated with negative log-likelihood and expected calibration error. Neither reached a useful error, about 6 m on average, and neither made the main branch.
- **Haiyang Hu** ([@haiyanghu2432](https://github.com/haiyanghu2432)): a first safety-framework notebook on the archived branch, with an MC dropout model trained on squared error, a plotted AV risk zone and a braking threshold on variance over time.
- **Nikhil Sharma** ([@npsharma03](https://github.com/npsharma03)): a refactor of the MC dropout script on the archived branch, with notes on a teacher-forcing variant that performed worse, the short README that preceded the final one, and the team's Canvas submissions.
- **Ian Kidwell** ([@iakidwell](https://github.com/iakidwell)): the initial repository layout, a data exploration notebook and the first baseline LSTM notebook, in an `experiments/` folder that the restructuring of February 27 replaced.

The repository ran from February 20 to March 12, 2026. This copy has the default branch only; the `archived` branch with the other members' earlier work is in the authors' repository. The attributions follow the commit history there.

## Getting started

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install torch numpy pandas matplotlib pyro-ppl jupyter
```

Build `data/raw/raw/{train,val,test}/` as described in [data/README.md](data/README.md). Then, from the repository root:

```sh
python baseline/baseline_lstm.py
python -m mc_dropout.mcmc
python -m variational_bnn.bnn
```

Each script trains for 100 epochs with seed 42, evaluates on the validation windows every 5 epochs, saves the checkpoint with the lowest final displacement error under its `models/` folder and writes a results CSV. The two Bayesian scripts import `src` as a package, so they must be run as modules from the root; the baseline script puts the root on the path itself. The safety notebook, `safety/safety_analysis.ipynb`, needs both Bayesian checkpoints, which are not committed, and expects to be started from the `safety/` folder.

The staff check needs no data and runs in under a minute:

```sh
python staff/check_pyro_training.py
```

## Data

- **Source:** the BIWI Walking Pedestrians recordings (Pellegrini et al., ICCV 2009, ETH Zurich) and the Crowds-by-Example recordings (Lerner et al., Computer Graphics Forum 2007, University of Cyprus), from fixed overhead cameras, in the text form prepared for Social GAN and redistributed in the [Trajectron++](https://github.com/StanfordASL/Trajectron-plus-plus) repository under its MIT license. Rows are frame, pedestrian id, x and y in metres, one annotated frame per 0.4 seconds.
- **Terms:** published for research use; cite the original papers. Nothing is committed here. One 5,492-row ETH file and a tensor of ETH test sequences from the first week of the project remain in the history.
- **How to get it:** [data/README.md](data/README.md).
- **Scenes and splits:** the scripts train and validate on ETH, Hotel, Univ, Zara1 and Students001. The write-up's table lists eight scenes with Zara2 held out, but Zara2, Zara3 and Students003 are simply unused, and no script evaluates on a test split. The notebook's test loader reads the six whole-scene files, 34,161 windows, four of them training scenes.
- **Windows:** every run of 20 consecutive annotated frames of one pedestrian, with a stride of one frame, so neighbouring windows share 19 frames. Observations are positions minus the first observed position; targets are positions minus the last observed position. These are positions in a shifted frame, not step-to-step displacements, which matters below.

## Project structure

| Path | Contents |
|---|---|
| `REPORT.md` | The authors' write-up: dataset, the three models with diagrams and code excerpts, the safety framework with its plots, conclusions, usage. |
| `src/data_loader.py` | Reading the text files, cutting 8+12 windows, the relative-position normalisation, and loaders per scene list for train, val and test. |
| `baseline/baseline_lstm.py` | The deterministic encoder-decoder LSTM, training loop and evaluation. |
| `mc_dropout/mcmc.py` | The MC dropout LSTM, `mc_predict` with dropout kept on, training and evaluation. Despite the file name there is no Markov chain Monte Carlo in it. |
| `variational_bnn/bnn.py` | The LSTM with a Pyro model and guide over the output layer, SVI training, sampling-based prediction. |
| `*/results/training_results.csv` | Validation metrics every 5 epochs for each model. |
| `*/figures/` | Architecture, intuition and training-history figures for each model. |
| `safety/safety_analysis.ipynb`, `safety/plots/` | The safety framework on one batch of 32 windows, with its seven plots. |
| `staff/check_pyro_training.py` | Course staff: does SVI train the LSTM? |
| `data/README.md` | Course staff: how to rebuild the data folder. |

## Method

**Baseline.** A batch of (8, 2) observations goes through a two-layer LSTM with 128 hidden units; the top layer's final hidden and cell state seed an LSTM cell, whose first input is the last observed position. Twelve steps are decoded, each passed through a linear layer to (x, y) and fed back as the next input. The loss is the mean over batch and steps of the Euclidean distance between predicted and true relative position. Adam at 1e-3, a plateau scheduler on the training loss (patience 5, factor 0.5), gradient clipping at 1.0, batch size 32, 100 epochs, seed 42, validation every 5 epochs, checkpoint on the best final displacement error.

**MC dropout.** The same architecture with a dropout layer at 0.3 applied to the top-layer hidden state before decoding and to the decoder's hidden state at every step; the encoder also has PyTorch's inter-layer dropout of 0.3 during training. `mc_predict` puts the model in evaluation mode, switches every `nn.Dropout` module back to training mode, runs 50 forward passes and returns their mean, variance and the samples. Validation error uses the 50-pass mean; the reported uncertainty is the variance averaged over steps, coordinates and the batch.

**Variational output layer.** The Pyro model draws the output weights W (2 × 128) and bias b from Normal(0, 1), runs the LSTM encoder and decoder with dropout 0.1, applies the sampled weights functionally with `F.linear`, and observes the targets as Normal(prediction, 0.1) inside a plate over the batch. The guide is a factorized Normal with learnable means and softplus-transformed scales initialised at −5, a standard deviation of about 0.007. Trained with `Trace_ELBO` and Pyro's Adam at 1e-3 for 100 epochs. Prediction traces the guide 50 times and reruns the network with each weight sample.

**Safety framework.** For each window, the variance is averaged over steps and coordinates; the score is 1 minus that average over an upper threshold, clamped to [0, 1]. Labels are SAFE below 0.020, CAUTION to 0.040 and UNSAFE above for MC dropout, and 0.002 and 0.004 for the variational model, chosen to match its scale. The fan plots draw the 50 sample paths per window; the calibration plot bins the 32 windows into deciles of variance against mean error; the crossing scenario takes the least and the most uncertain window in the batch and draws a straight car path with a conflict circle of three final-step standard deviations.

### What is Bayesian here

MC dropout rests on Gal and Ghahramani's result that a network trained with dropout before every weight layer is doing approximate variational inference: the dropout masks define a variational distribution over the weights, and the dropout-regularised objective is an evidence lower bound up to constants. Keeping dropout on at test time samples from that approximate posterior, so the mean of 50 passes is an approximate posterior predictive mean and their spread is the epistemic part of the predictive uncertainty. Here dropout is applied to the hidden state at two points rather than before every weight layer, and the inter-layer dropout inside `nn.LSTM` stays off at inference, because `model.eval()` disables it and only `nn.Dropout` modules are switched back on. The approximation is loose, but it is the right idea and the sampling is done correctly.

The variational output layer is a genuine, if small, Bayesian model: a prior over 258 weights, a variational family, an ELBO with the KL term to the prior and a Gaussian likelihood, optimised by SVI, and prediction by sampling the posterior. The observation noise of 0.1 m is fixed rather than learned, so there is no aleatoric term, and the data term weights each squared metre of error by 50. Because the rest of the network is deterministic, the output weights are the only source of predictive spread, and with the posterior scales starting at 0.007 and a strong likelihood pulling them tighter, a predictive variance of 0.003 m², a standard deviation of about 5 cm, is what this setup produces. It is also the part of the model that did not train, below.

The baseline is not Bayesian; it is the point-estimate comparison. The topic proposal listed Kalman and particle filters, hidden Markov models, a VAE and MCMC; none of those were built, and the archived branch's fuller Bayesian LSTM did not work.

## Results

Every number is on the validation windows of the five training scenes, read from the saved CSVs, evaluated every 5 epochs, best epoch by final displacement error. Nothing was evaluated on a test split.

| Model | Best ADE (m) | Best FDE (m) | Predictive variance at the best epoch (m²) |
|---|---|---|---|
| Baseline LSTM | 0.4207 (epoch 50) | 0.8956 (epoch 50) | none |
| MC dropout LSTM | 0.4187 (epoch 50) | 0.8892 (epoch 50) | 0.0251 |
| Variational output layer | 0.5813 (epoch 95) | 1.0315 (epoch 75) | 0.0034 |

ADE is the mean distance between predicted and true position over the 12 steps; FDE is the distance at the last step. The write-up's own table gives the MC dropout's best ADE as 0.4209 at epoch 95; the CSV and the training-history figure say 0.4187 at epoch 50.

**Safety framework**, on the first 32 windows of the ETH scene file. MC dropout: 22 SAFE, 2 CAUTION, 8 UNSAFE, mean score 0.654, batch variance 0.0314. Variational model: 23, 2 and 7, mean score 0.485, batch variance 0.0032. The MC dropout variance grows from 0.0031 at step 1 to 0.0691 at step 12, a 22-fold increase; the variational model's grows from near zero to about 0.011. The crossing scenario's confident pedestrian has a variance of 0.00012 and the uncertain one 0.129.

In standard deviations, MC dropout's 0.025 m² is 16 cm, and its final-step 0.069 m² is 26 cm, against an average error of 42 cm and a final error of 89 cm.

## What to learn from this project

Course staff added this section. The pipeline is the team's own, the MC dropout model is sound, and the write-up is unusually readable; the points below are about checking what the code does before writing about it.

- **Check that your Bayesian model trained.** The Pyro model uses the LSTM encoder and decoder without registering them with `pyro.module`, so the SVI optimiser only knows about the four tensors in Pyro's parameter store, and the LSTM's 267,000 weights never move from their random initialisation. `staff/check_pyro_training.py` shows it: zero change in any LSTM weight across 30 SVI steps, and movement as soon as the module is registered. The "Variational BNN" results are those of a Bayesian linear read-out on top of an untrained recurrent network, which is why it is 16 cm worse than a baseline whose architecture it shares, and why its variance is so small. The write-up's diagnosis, posterior collapse because the prior is too strong, and the conclusion that MC dropout wins, both rest on this. One line fixes it. A model that is worse than its deterministic twin, a checkpoint whose weights equal their initialisation, a loss that barely moves: any one of these is the cue to look.
- **"Held out" has to be true in the code.** The write-up and the notebook report results "on the ZARA2 test set". No script loads a test split; every number is the validation set of the five training scenes, and that set also chose the checkpoints. The notebook's test loader reads six whole-scene files, and for ETH the whole scene is exactly the training file plus the validation file, row for row, with the other training scenes laid out the same way. Its batch of 32, on which every safety plot is made, is the first 32 windows of the ETH file: training data. The leave-one-scene-out protocol that the write-up names is the standard for this benchmark, and the loader can do it; it was never run.
- **Plot in the coordinates your data are in.** The loader produces positions relative to the first and last observation, not step-to-step displacements; the write-up describes the inputs as displacements throughout, and the notebook takes a cumulative sum before plotting "to recover absolute positions". The fans therefore show cumulative sums of positions: paths 50 m long for 4.8 seconds of walking, observed and predicted segments that both start at the origin because they are in two different frames, and a conflict circle whose radius is in metres drawn on axes that are not. The classification, calibration and growth numbers are computed before the cumulative sum and are unaffected; the pictures are not of anything a pedestrian did.
- **Compare the spread to the error.** MC dropout's standard deviation is 16 cm on average and 26 cm at the final step; the error is 42 cm on average and 89 cm at the final step. The 50 paths in a fan therefore rarely reach the true one, which is what the calibration scatter shows and what the proposal's own success criterion, 90 percent intervals covering 90 percent of trajectories, would have made plain. It was never computed. Dropout variance is only the epistemic part; there is no aleatoric term, and the fixed 0.1 m noise in the Pyro model is not learned from anything. A calibration curve from 32 points in 10 bins cannot be read either way.
- **Ask what your uncertainty is tracking.** The SAFE pedestrian in the crossing scenario barely moves (variance 0.00012); the UNSAFE one walks far. With outputs that are positions relative to the last observation, a faster walker has larger outputs, and dropout noise on the hidden state scales with them. The thresholds were set by eye to the MC dropout's variance scale and then divided by ten for the other model so that the two produce similar counts, which makes the agreement between their classifications circular. Whether the score separates ambiguous situations from fast ones is the question the framework was built to answer, and it does not.
- **Keep illustrations apart from results.** The intuition figures with ±0.1 m and ±2.0 m of uncertainty are drawn scenarios, not model output, but the write-up describes them as what the 50 passes produce. Diagrams are useful; label them as diagrams. A fragment of an AI model's name sits at the end of the comparison section, a sign that drafted text went out unread.
- **Small things.** The MC dropout file is called `mcmc.py`. The encoder's inter-layer dropout is off at MC inference. The default scene list in `get_train_loader` names a scene that does not exist. The install instructions clone `your-repo`. The write-up gives the MC dropout's mean uncertainty as 0.024 in one table and 0.031 in another. There is no license.

What the project does well: a compact, readable implementation of a sequence-to-sequence LSTM and of MC dropout, with a prediction function that does the one thing MC dropout requires; a Pyro model written by hand, with an explicit model and guide rather than an automatic one; results saved as CSVs, so that every number in the write-up can be checked, and was; honest reporting that the more principled method did worse, even if the reason was misdiagnosed; a seeded training setup that checkpoints on validation error; and a write-up that explains MC dropout and variational inference to a reader who has never met them, with an evaluation design, leave one scene out, ADE and FDE, calibration, that is the right one even though it was not carried out.

## About this project

This project was built in Bayesian Machine Learning with Generative AI Applications (ADSP 32014), part of the Master of Science in Applied Data Science at the University of Chicago, in Winter 2026. It is shown here as an example for later cohorts.

## Changes by course staff

Course staff forked the authors' repository and made these edits. The authors' code, notebook and figures are unchanged.

- Renamed the authors' README to `REPORT.md`, unchanged, and added this README with the program banner.
- Added `staff/check_pyro_training.py`, which runs the authors' Pyro model for 30 SVI steps on random tensors, with and without `pyro.module`, and prints how much the LSTM weights change.
- Added `data/README.md` with the steps to rebuild the data folder from Trajectron++, and tidied `.gitignore` so that data, checkpoints and editor files stay out.

## License

The authors have not added a license, so their work is shown here under GitHub's terms for public repositories. Ask them before reusing it. The datasets have their own terms, above.
