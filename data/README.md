# Data

`data/raw/` in this copy holds small sample PDFs only.

CUAD is not included. `scripts/build_training_data_from_cuad.py` downloads it on first use from:

https://huggingface.co/datasets/theatticusproject/cuad/resolve/main/CUAD_v1/CUAD_v1.json

The file is saved at `data/raw/cuad/CUAD_v1.json`. CUAD is CC BY 4.0. Cite Hendrycks et al. 2021.

Trained weights are not included. Training writes them under `data/models/`.
