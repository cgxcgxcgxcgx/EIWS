

## Overview

This project proposes a hybrid modeling approach that combines physics-informed methods with data-driven machine learning techniques to estimate the phase speeds of ocean internal waves using remote sensing imagery. The approach integrates physical oceanographic principles with advanced neural networks and ensemble learning to improve accuracy and robustness in phase speed estimation from satellite data.

---

## Repository Structure

- `data_utils.py` — Utilities for loading and preprocessing data.
- `base_models.py` — Training and saving classical machine learning base models.
- `nn_model.py` — Building, training, fine-tuning, and saving neural network models, plus loss visualization.
- `meta_model.py` — Training and saving meta-level stacking model.
- `stacking_pipeline.py` — Pipeline orchestrating model training, fine-tuning, and evaluation.
- `main.py` — Entry script to run the full training and fine-tuning workflow.

---

## Requirements

- Python 3.7+
- TensorFlow 2.x
- scikit-learn
- xgboost
- pandas
- numpy
- matplotlib
- joblib

Install dependencies via:

```bash
pip install tensorflow scikit-learn xgboost pandas numpy matplotlib joblib
````

---

## Usage

1. Prepare your training and fine-tuning datasets in CSV format with features and labels.
2. Adjust paths and hyperparameters in `main.py` as needed.
3. Run the main script:

```bash
python main.py
```

This will train base models, train and fine-tune the neural network, train the meta model, and save all models and loss plots.

---

## Citation


> **A Hybrid Physics-Informed and Data-Driven Model for Estimating Ocean Internal Wave Phase Speeds from Remote Sensing Imagery**
> Remote Sensing of Environment, \[Year], \[Volume], \[Pages].
> DOI: \[]


````
