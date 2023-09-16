# Reproducing the Experiments

This directory contains the Jupyter notebooks required to reproduce the experiments and results presented in our paper. Below, you'll find a brief description of each notebook and its purpose:

## Notebooks

### 1. `bootstrapping_real_sample.ipynb`
- **Purpose:** Generates averaged metrics for the real dataset with bootstrapping.
- **Output:** Creates the `metrics_bootstrap_real.pkl` file which contains the averaged metrics based on the real dataset. 

### 2. `evaluation.ipynb`
- **Purpose:** Evaluates synthetic data from all experiments and compares them to metrics from the real data.
- **Output:** Creates the `agg_df` dataframe used to generate all results.

### 3. `paper_experiments.ipynb`
- **Purpose:** Provides the code necessary to reproduce all tables in the paper.

### 4. `content_analysis.ipynb`
- **Purpose:** Includes the qualitative content analysis describe in section 4 of the paper.
