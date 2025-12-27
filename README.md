# Intro ML Capstone — Housing Price Prediction

Capstone project for a **graduate-level Introductory Machine Learning** course at UNC Charlotte.  
This project compares **classical regression models** with **research-inspired ensemble methods** on the Ames Housing dataset (Kaggle).

The focus is on **preprocessing, model comparison, and generalization**, not leaderboard optimization.

---

## Overview

- Dataset: Ames Housing (79 features, mixed numeric + categorical)
- Task: Predict house sale prices
- Evaluation: RMSE, MAE, MSE, R²
- Emphasis: reproducibility, interpretability, and analysis

---

## Models Implemented

**Classical models**
- Linear Regression  
- Polynomial Regression (degree 2)  
- Ridge Regression (L2 regularization)

**Research-inspired models**
- Random Forest  
  - baseline
  - paper-reported hyperparameters (2024)
- XGBoost  
  - baseline
  - paper configuration
  - randomized cross-validation tuning

---

## Preprocessing (High Level)

Implemented using `scikit-learn` Pipelines and ColumnTransformers:

- log(1 + x) transform for skewed features and target
- median imputation for missing values
- robust scaling for linear models
- one-hot encoding for categorical features

(Tree-based models use a simplified, scale-invariant pipeline.)

---

## Results (Summary)

- Linear and Ridge Regression performed strongly on validation data due to clear linear structure in the dataset.
- Polynomial Regression underperformed due to increased variance.
- Random Forest models were sensitive to hyperparameter choices.
- XGBoost showed the best generalization on the test set.

Overall takeaway: **simple models + good preprocessing can be very competitive**, while more complex models require careful tuning to generalize well.

---

## Repository Structure

Each file is a standalone `.py` script derived from the original notebooks.  
Some scripts include multiple configurations or variants of the same model.

