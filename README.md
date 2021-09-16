# Protein Stability Predictor 
for cDNA display / protease project @ Rocklin Lab

## Structure of the repo

      ├── data
      ├── notebooks
      │   ├── run_model.ipynb
      ├── requirements.txt
      ├── results
      │   ├── hyperparameters
      │   ├── losses
      │   ├── saved_weights
      │   └── stability_scores
      └── src_
          ├── config.py
          ├── evals
          │   ├── data_processing.py
          │   ├── hyperparameter_search.py
          │   ├── run_model.py
          │   └── stability_score.py
          ├── models
          │   ├── convnet1d.py
          │   ├── convnet2d.py
          │   ├── evaluator_model.py
          │   ├── losses.py
          │   └── wrapper.py
          └── utils
              ├── general.py
              └── plotting.py
