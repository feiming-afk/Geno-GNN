# Characterization of SARS-CoV-2 fitness dynamics across immune backgrounds via graph representation learning
We developed Geno-GNN, a graph representation learning model, to predict the ACE2 binding affinity of a given SARS-CoV-2 RBD sequence and its immune escape potential against various specific immune types (e.g., WT convalescent, WT vaccine, BA.1+BTI convalescent, BA.2+BTI convalescent, and BA.5+BTI convalescent). After validation on external datasets, we utilized Geno-GNN to characterize the viral fitness dynamics of SARS-CoV-2 during the COVID-19 pandemic and to reveal potential evolutionary trajectories of the virus through virtual mutation scanning. The folders here contain the code for the model and the main analyses of the article.
## Components
1. The model structure, training process: `./geno_gnn/`
2. External testing analysis of Geno-GNN: `./analysis/external_test/`
3. SARS-CoV-2 fitness dynamics and immune background-related fitness variations revealed by Geno-GNN: `./analysis/time_vary/`
4. Exploration of the evolutionary trajectory of SARS-CoV-2 and quantification of mutation adaptation effects by Geno-GNN: `./analysis/mutation_space/`
## Data availability
ACE2 affinity datasets for training and testing the Geno-GNN model are sourced from [Starr et al., 2022](https://doi.org/10.1371/journal.ppat.1010951), [Taylor et al., 2024](https://doi.org/10.1093/ve/veae067), and [Moulana et al., 2022](https://doi.org/10.1038/s41467-022-34506-z).</br>Immune escape datasets are obtained from [Cao et al., 2022a](https://doi.org/10.1038/s41586-022-04980-y), [Cao et al., 2022b](https://doi.org/10.1038/s41586-021-04385-3), and [Cao et al., 2023](https://doi.org/10.1038/s41586-022-05644-7).</br>The GISAID dataset, requiring registration and authentication, is not provided here but can be accessed via [GISAID](https://www.gisaid.org).

## Dependencies
Our code was executed using `Conda 23.3.1`, `Python 3.7.16`, and `CUDA 11.7` on `CentOS Linux release 7.9.2009`. The required Python dependencies are specified in the `environment.yaml` file.
### Automatic Installation
1. Create an environment using the `environment.yaml` file:
```
conda env create -f environment.yaml
```
2. Activate the new environment:
```
conda activate geno_gnn
```
### If Installation Fails
1. Create a new Python environment manually:
```
conda create -n geno_gnn python=3.7.16
```
2. Activate the new environment:
```
conda activate geno_gnn
```
3. Manually Install Dependencies:
   - Review `environment.yaml` for the list of required packages and versions.
   - For any package that fails to install via `pip` or `conda`, download its wheel file (links may be in `environment.yaml` or sourced from trusted sites).
   - Install each downloaded wheel file with:</br>
     (Update the path to the actual location of the downloaded file.)
   ```
   pip install /path/to/downloaded_wheel.whl
   ```
   - Repeat until all dependencies from `environment.yaml` are installed.
