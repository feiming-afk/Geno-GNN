# Characterization of SARS-CoV-2 fitness dynamics across immune backgrounds via graph representation learning
We developed Geno-GNN to predict the ACE2 affinity for a given RBD sequence, as well as the overall antibody escape levels specific to these cohorts, and to analyze the evolutionary trajectory of COVID-19. The folders here contain the code for the model and the main analyses of the article.

1. The model structure, training process, and the model used for the article analysis: `./geno_gnn/`
2. External testing analysis of Geno-GNN: `./analysis/external_test/`
3. The time dynamics of SARS-CoV-2 fitness and the immune background-related fitness differences revealed by Geno-GNN: `./analysis/time_vary/`
4. Exploration of the evolutionary trajectory of SARS-CoV-2 and quantification of mutation adaptation effects by Geno-GNN: `./analysis/mutation_space/`
