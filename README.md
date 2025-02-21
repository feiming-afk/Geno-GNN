# Characterization of SARS-CoV-2 fitness dynamics across immune backgrounds via graph representation learning
We developed Geno-GNN, a graph representation learning model, to predict the ACE2 binding affinity of a given SARS-CoV-2 RBD sequence and its immune escape potential against various specific immune types (e.g., WT convalescent, WT vaccine, BA.1+BTI convalescent, BA.2+BTI convalescent, and BA.5+BTI convalescent). After validation on external datasets, we utilized Geno-GNN to characterize the viral fitness dynamics of SARS-CoV-2 during the COVID-19 pandemic and to reveal potential evolutionary trajectories of the virus through virtual mutation scanning. The folders here contain the code for the model and the main analyses of the article.

1. The model structure, training process, and the model used for the article analysis: `./geno_gnn/`
2. External testing analysis of Geno-GNN: `./analysis/external_test/`
3. SARS-CoV-2 fitness dynamics and immune background-related fitness variations revealed by Geno-GNN: `./analysis/time_vary/`
4. Exploration of the evolutionary trajectory of SARS-CoV-2 and quantification of mutation adaptation effects by Geno-GNN: `./analysis/mutation_space/`
