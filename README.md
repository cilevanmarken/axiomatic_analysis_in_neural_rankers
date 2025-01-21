# Reproducing Axiomatic Causal Interventions for Reverse Engineering Relevance Computation in Neural Retrieval Models
Authors: Catherine Chen, Jack Merullo, and Carsten Eickhoff
By Cile van Marken, University of Amsterdam

<!-- This code corresponds to the paper: __Axiomatic Causal Interventions for Reverse Engineering Relevance Computation in Neural Retrieval Models__, in _Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’24)_, July 14–18, 2024, Washington, DC, USA. [Link to paper](https://arxiv.org/abs/2405.02503) -->

## Repository
```
ProjAI/
├── data/
│   ├── TFC1-I/
│   ├── TFC1-R/
│   ├── TFC2/
│   ├── baseline.json
│   ├── QIDs_with_text.csv
│   ├── queries.json
│   ├── selected_query_terms.csv
├── figures/
│   ├── TFC1-I/
│   ├── TFC1-R/
│   └── TFC2/
├── results/
│   ├── TFC1-I/
│   ├── TFC1-R/
│   └── TFC2/
├── TransformerLens/
├── .gitignore
├── create_datasets.py
├── environment.yml
├── experiment.py
├── helpers.py
├── LICENSE
├── patching_helpers.py
├── plot_results.py
├── README.md
└── retrieval_patching_demo.py
```

## Setup
To install an environment with all neccesary packages, run the following commands:
```
conda create --name ACI_env python=3.9
conda activate ACI_env
pip install ./TransformerLens
pip install -r requirements.txt
```

This code uses a copy of the [TransformerLens](https://github.com/neelnanda-io/TransformerLens) package, with additional changes to support activation patching in a retrieval setting. Changes made to the original TransformerLens package to support activation patching for retrieval (TAS-B) can be found in the following files:

- `components.py`
- `loading_from_pretrained.py`
- `HookedEncoder.py`


## Data and diagnostic datasets
To reproduce the experiments in the papers, first download the data from Google Drive --> TODO. 
<!-- [download the data from Google Drive](https://drive.google.com/file/d/1duqXgx2iqPyoom0Nui3nwy33rqjt5pll/view?usp=drive_link) -->
Then run the following command to create the baseline and perturbed datasets for the experiment from the original documents.

```
python create_datasets.py --dataset DATASET --TFC1_I_perturb_type TFC1_I_PERTURB_TYPE --TFC2_K [1, 2, 5, 10, 50]
```

`DATASET`:
- `TFC1-I`: Experiment from the original paper, where a query term is appended or prepended to the orginal document.
- `TFC1-R`: Experiment from the original paper, where all occurences of a query term in the original document are replaced by a filler token .
- `TFC2`: Extension of the original paper, where K query terms are appended to the document text as a baseline, and K + 1 query terms are appended to the document text as perturbation.

`TFC1_I_PERTURB_TYPE` (for TFC1-I):
- `append`: target query term added to the end of a document
- `prepend`: target query term added to the beginning of a document

`TFC2_K` (for TFC2): list, possibly containing
- 1: run experiment with baseline document with 1 appended query terms and perturbed document with 2 appended query terms.
- 2: run experiment with baseline document with 2 appended query terms and perturbed document with 3 appended query terms.
- 5: run experiment with baseline document with 5 appended query terms and perturbed document with 6 appended query terms.
- 10: run experiment with baseline document with 10 appended query terms and perturbed document with 11 appended query terms.
- 50: run experiment with baseline document with 11 appended query terms and perturbed document with 51 appended query terms.


## Experiments

To reproduce the experiments in the papers, first [download the data from Google Drive](https://drive.google.com/file/d/1duqXgx2iqPyoom0Nui3nwy33rqjt5pll/view?usp=drive_link). 

To run the patching experiments:

```
python experiment.py --dataset DATASET --experiment_type EXPERIMENT_TYPE --TFC1_I_perturb_type TFC1_I_PERTURB_TYPE --TFC2_K TFC2_K
```

The patching experiments are currently designed to be run on a single GPU, and depending on the experiment, can take several hours to days to complete. To save time, we would suggest setting up separate jobs for each type of patching experiment.


## Visualization
To create the plots from the papers, run the following command:
```
python plot_results.py --dataset DATASET --experiment_type EXPERIMENT_TYPE --TFC1_I_perturb_type TFC1_I_PERTURB_TYPE --TFC2_K TFC2_K
```

