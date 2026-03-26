# Heart Disease Risk Predictor

A complete Flask web application for heart disease risk prediction using multiple machine learning models, fuzzy logic, feature selection, data augmentation, and an ensemble stacking layer.

## What This Project Does

The app lets a user enter patient information and predicts the probability of heart disease. It also shows:

- an overall risk percentage
- a risk level badge
- a model comparison chart
- a radar chart against a healthy profile
- feature impact scores
- a medical guidance message

The project also includes a website training lab where you can:

- add new labeled rows from the web UI
- save partial rows with missing values automatically filled
- browse the datasets used by the app
- run fast retrain or full retrain from the website
- remove added rows and choose whether to use no retrain, fast retrain, or full retrain afterward

## Models Used

This project uses 4 base models and 1 ensemble layer:

1. Artificial Neural Network (ANN)
2. Random Forest
3. Fuzzy Logic
4. Support Vector Machine (SVM)
5. Stacking Meta-Learner using Logistic Regression

It also uses:

- Genetic Algorithm for feature selection
- SMOTE and RandomOverSampler for data balancing and augmentation
- MinMaxScaler for feature scaling

## How It Works

The full training pipeline is:

1. Download the UCI Cleveland Heart Disease dataset if it is not already present.
2. Clean the data by handling `?`, converting types, filling missing values, binarizing the target, removing duplicates, and capping outliers.
3. Merge any website-added labeled rows.
4. Augment the dataset to a balanced 1100-row training set using SMOTE and RandomOverSampler.
5. Scale all 13 input features with MinMaxScaler.
6. Select the most useful features using a Genetic Algorithm.
7. Train the base models: ANN, Random Forest, Fuzzy Logic, and SVM.
8. Train a stacking meta-model on top of the base model probabilities.
9. Save all artifacts for web inference.

At prediction time, the app:

1. accepts full or partial patient input
2. auto-fills missing values when needed
3. scales the input using the saved scaler
4. keeps only the selected GA features
5. gets probabilities from ANN, Random Forest, Fuzzy Logic, and SVM
6. sends those 4 probabilities into the saved stacking model
7. returns the final risk prediction and chart data to the frontend

## Dataset

- Source: UCI Cleveland Heart Disease Dataset
- Raw rows: 303
- Cleaned rows: 303
- Augmented rows: 1100

## Latest Saved Metrics

The current saved metrics in `models/metrics.json` are:

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| ANN | 0.9136 | 0.9027 | 0.9273 | 0.9148 |
| Random Forest | 0.9864 | 0.9908 | 0.9818 | 0.9863 |
| Fuzzy Logic | 0.4864 | 0.4920 | 0.8364 | 0.6195 |
| SVM | 0.9773 | 0.9907 | 0.9636 | 0.9770 |
| Stacked Ensemble | 0.9727 | 0.9815 | 0.9636 | 0.9725 |

Notes:

- Random Forest is currently the strongest single base model in the saved artifacts.
- The Stacked Ensemble is the final web prediction layer.
- Metrics may change slightly after future retraining, especially if website-added rows are included.

## Project Structure

```text
heart-disease-detection/
|- data/
|  |- raw/
|  |- processed/
|  `- user/
|- models/
|- results/
|- src/
|  |- ann_model.py
|  |- data_augmentation.py
|  |- data_cleaning.py
|  |- data_download.py
|  |- evaluate.py
|  |- fuzzy_logic.py
|  |- genetic_algorithm.py
|  |- random_forest.py
|  |- stacking_model.py
|  |- svm_model.py
|  `- user_data.py
|- static/
|  |- css/
|  `- js/
|- templates/
|- app.py
|- requirements.txt
`- train.py
```

## Requirements

Tested with:

- Python 3.13
- Windows PowerShell locally

Main Python packages:

- Flask
- NumPy
- Pandas
- scikit-learn
- imbalanced-learn
- TensorFlow / Keras
- scikit-fuzzy
- matplotlib
- seaborn
- joblib

Install everything with:

```powershell
pip install -r requirements.txt
```

## Quick Start

If you already have Python installed, this PowerShell one-liner sets up the environment, installs dependencies, trains the models, and starts the app:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt; python train.py full; python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Common Commands

Install dependencies:

```powershell
pip install -r requirements.txt
```

Full training:

```powershell
python train.py full
```

Fast training:

```powershell
python train.py fast
```

Start the web app:

```powershell
python app.py
```

## Full Retrain vs Fast Retrain

### Full Retrain

Full retrain runs the entire pipeline:

- GA feature selection again
- full Random Forest tuning
- full SVM tuning
- full stacking workflow

Use it when:

- you want the most complete rebuild
- you have added a meaningful amount of new data
- you changed training logic

### Fast Retrain

Fast retrain is a speed-focused mode:

- reuses saved GA-selected features
- reuses previously tuned RF and SVM parameters
- trains ANN with fewer epochs
- uses a lighter stacking path

Use it when:

- you only added a few rows from the website
- you want a much quicker refresh
- you do not need a full exhaustive rebuild

## Website Features

The web app includes:

- patient prediction form
- partial-input prediction with automatic value filling
- gauge, bar, radar, and feature impact charts
- sample generation buttons
- dataset explorer
- save-row training lab
- fast retrain and full retrain buttons
- remove added rows with retrain mode choice

## Troubleshooting

If predictions fail:

- make sure `models/` contains the saved model files
- run `python train.py full` again

If the website says artifacts are not ready:

- retrain the models
- restart `python app.py`

If you remove website-added rows:

- choose `fast`, `full`, or `none` depending on whether you want the current model to forget them immediately

## License

MIT
