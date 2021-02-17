# _k_-Anonymity in Practice: How Generalisation and Suppression Affect Machine Learning Classifiers

This repository contains the Python code for applying different _k_-anonymisation algorithms, i.e., Optimal Lattice Anonymization (OLA), Mondrian, Top-Down Greedy Anonymisation (TDG), k-NN Clustering-Based (CB) Anonymisation,  on datasets and measuring their effects on machine learning classifiers as presented in [_k_-Anonymity in Practice: How Generalisation and Suppression Affect Machine Learning Classifiers](https://arxiv.org/abs/2102.04763).

```bibtex
@misc{slijepčević2021kanonymity,
    title        = {$k$-Anonymity in Practice: How Generalisation and Suppression Affect Machine Learning Classifiers},
    author       = {Djordje Slijepčević and
                    Maximilian Henzl and
                    Lukas Daniel Klausner and
                    Tobias Dam and
                    Peter Kieseberg and
                    Matthias Zeppelzauer},
    year         = 2021,
    eprint       = {2102.04763},
    archiveprefix = {arXiv},
    primaryclass = {cs.LG}
}
```

## Setup

In order to install the necessary requirements either use `pipenv install` or `pip3 install -r requirements.txt`.
Then activate the virtual environment, e.g. with `pipenv shell`.

## Code & Usage
The code is written in Python 3 and conducts following steps for each experiment:

- read specified dataset
- measure specified ML algorithm performance using original dataset
- anonymise dataset with specified algorithm and current value _k_ for _k_-anonymity
- measure specified ML algorithm performance using anonymised dataset
- repeat previous steps for other configured values of _k_

The different parameters (dataset, ML algorithm, _k_-anonymisation algorithm, _k_) are defined via arguments as follows:

```txt
usage: baseline_with_repetitions.py [-h] [--start-k START_K] [--stop-k STOP_K] [--step-k STEP_K] [--debug] [--verbose] [{cmc,mgm,adult,cahousing}] [{rf,knn,svm,xgb}] {mondrian,ola,tdg,cb} ...

Anonymize data utilising different algorithms and analyse the effects of the anonymization on the data

positional arguments:
  {cmc,mgm,adult,cahousing}
                        the dataset used for anonymization
  {rf,knn,svm,xgb}      machine learning classifier
  {mondrian,ola,tdg,cb}
    mondrian            mondrian anonyization algorithm
    ola                 ola anonyization algorithm
    tdg                 tdg anonyization algorithm
    cb                  cb anonyization algorithm

optional arguments:
  -h, --help            show this help message and exit
  --start-k START_K     initial value for k of k-anonymity
  --stop-k STOP_K       last value for k of k-anonymity
  --step-k STEP_K       step for increasing k of k-anonymity
  --debug, -d           enable debugging
  --verbose, -v
```

The _k_-anonymisation algorithms "_k_-NN Clustering-Based Anonymisation", "Mondrian" and "Top-Down Greedy Anonymisation" located in the folders `clustering_based`, `basic_mondrian` and `top_down_greedy` are based on the open-source implementation of [Qiyuan Gong](mailto:qiyuangong@gmail.com).

The original reporitories can be found on github.com:

- [Clustering Based k-Anonymization](https://github.com/qiyuangong/Clustering_based_K_Anon)
- [Basic Mondrian](https://github.com/qiyuangong/Basic_Mondrian)
- [Top Down Greedy Anonymization](https://github.com/qiyuangong/Top_Down_Greedy_Anonymization)

Our changes included the migration of Python 2 to Python 3, the option to leave non-QID attributes and the target variable non-anonymised, the ability to handle float numbers in datasets, removal and cleanup of files and code that were irrelevant to our project.

## Data
The repository contains following locations for data:

- `datasets`
  - contains all available datasets in separate folders
- `generalization/hierarchies`
  - contains our defined generalization hierarchies per attribute and dataset
- `results`
  - all computed results (anonymised datasets, ML performance, etc.) are stored inside a folder structure inside `results` for each experiment
