# Uncertainty quantification for radiomics using RadMLBench

This repository contains the code for a large scale analysis of uncertainty estimation in radiomics datasets with direct connection to [radMLBench](https://pubmed.ncbi.nlm.nih.gov/39270457/). The code is packaged as a straightforward package where radMLBench datasets - or local datasets (see below!) - can be specified for model training. 

This library is not prepared for inference as the work developed here is strictly used to estimate how different model specifications, training regimes and uncertainty estimation methods have an impact in i) performance and in ii) performance upon sample deferral. The expected input is always a dataset, and the expected output is the predictions for the test set.

# Usage

`run-all.sh` contains a sample of how to run these models, but we explain these below.

## With radMLBench

1. `uv run script/split_data.py` can be used to split for a given dataset x seed combination. This also happens within `script/workflow.py`, but given that we wanted to parallelize this (using GPU `parallel`(https://www.gnu.org/software/parallel/)), we chose to separate data splitting from model training.
2. `uv run script/workflow.py` is then used to train models. This can take as specifications the machine-learning methods (`--methods <method>`), whether CatBoost should use posterior sampling during training (`--posterior-sampling`), the random seed (`--random-seed <number>`), the output directory for the results (`--output-dir <path>`), the datasets to be used for training (`--datasets <radmlbench-id-or-all>`, either dataset IDs for radMLBench), the number of virtual ensembles (`--virtual-ensembles-count <number>` for CatBoost), whether hyperparameter optimization should be performed (`--optimize`) and whether Platt scaling calibration should be applied to the predictions (`--calibration`). Hyperparameter optimization includes a small panel of hyperparameters for different models and the number of features that should be kept after feature selection using `SelectKBest` with F metrics.

## With local datasets

To train with local datasets, the workflow is practically identical: the exception is that `--datasets` should be the path to a local CSV file with feature, `id` and `class` columns. Any column that is not either `id` or `class` is assumed to be a feature. 

## Citation

This work is currently under revision, so we hope to have a citation ready soon :-)