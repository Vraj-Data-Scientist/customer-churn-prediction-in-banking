
# Customer Churn Prediction in Banking

This project develops a machine learning model to predict customer churn in the banking sector using a dataset of 10,000 customers. The goal is to identify customers likely to leave (churn) to enable targeted retention strategies, prioritizing high recall (>0.75) and reasonable precision (>0.50) to balance customer retention with campaign costs.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering](#feature-engineering)
- [Model Results](#model-results)
- [Industry Relevance](#industry-relevance)
- [Key Learnings](#key-learnings)
- [Installation and Usage](#installation-and-usage)
- [References](#references)

## Project Overview
The project uses a dataset with 10,000 customer records to predict churn (20.37% churn rate). Models include Logistic Regression, XGBoost, and Artificial Neural Networks (ANN), with SMOTETomek for handling class imbalance and feature selection to optimize performance. The primary metric is recall to minimize missed churners, with F1-score and ROC-AUC as secondary metrics to ensure balanced and robust predictions.

## Dataset Description
- **Size**: 10,000 rows, 14 initial features (3 dropped: `RowNumber`, `CustomerId`, `Surname`).
- **Target**: `Exited` (binary: 0 = not churned, 1 = churned; 20.37% churn rate).
- **Features**: 13 initial features (e.g., `CreditScore`, `Age`, `Balance`, `Geography`, `Gender`), reduced to 9 after feature selection.
- **Challenges**: 
  - Low mutual information (max MI=0.074), indicating weak feature-target relationships.
  - 36% zero-valued `Balance` introduces noise.
  - Limited features (no transactional data) cap recall at ~0.75.

## Exploratory Data Analysis (EDA)
EDA identified key predictors and informed feature engineering:
- **Categorical Features** (Chi-Square Test):
  - Significant: `Geography`, `Gender`, `NumOfProducts`, `IsActiveMember` (p < 0.05).
  - Non-significant: `Tenure`, `HasCrCard` (p > 0.05), kept for potential interactions.
- **Continuous Features** (T-Test):
  - Significant: `CreditScore`, `Age`, `Balance` (p < 0.05).
  - Non-significant: `EstimatedSalary` (p > 0.05), kept for ratios.
- **Correlation Analysis**:
  - No multicollinearity (all correlations < 0.8).
  - Engineered features (e.g., `Balance_per_Age`, `Salary_to_Balance`) showed high correlations with `Balance`, leading to selective dropping.
- **Mutual Information**:
  - Top features: `Age` (0.074), `NumOfProducts` (0.072), `Age_Binned` (0.051).
  - Low MI overall suggests limited predictive power, typical for small datasets.

## Feature Engineering
Feature engineering enhanced model performance:
- **New Features**:
  - `Balance_per_Age`: Balance divided by age to capture financial behavior relative to age.
  - `Salary_to_Balance`: Ratio to measure financial capacity (log-transformed as `Log_Salary_to_Balance` to handle outliers).
  - `Tenure_NumProducts`: Interaction to capture engagement patterns.
  - `Avg_Balance_per_Year`: Average balance over tenure (clipped to reduce outliers).
  - `Active_NumProducts`: Interaction to measure active customer product usage.
  - `Age_Binned`: Age grouped into bins (0–30, 30–45, 45+).
  - `CreditScore_Binned`: Credit score grouped into bins (0–600, 600–700, 700–850).
  - `Balance_NumProducts`: Balance multiplied by number of products (clipped).
- **Encoding**:
  - One-hot encoding for `Geography` (`Geo_Germany`, `Geo_Spain`).
  - Binary encoding for `Gender` (Male=0, Female=1).
- **Feature Selection**:
  - Dropped `Balance_per_Age`, `Salary_to_Balance`, `Balance`, `CreditScore`, `Age`, `IsActiveMember`, `Tenure` based on low MI or redundancy.
  - Retained 9 features: `Age_Binned`, `NumOfProducts`, `Active_NumProducts`, `Geo_Germany`, `CreditScore_Binned`, `Tenure_NumProducts`, `Balance_NumProducts`, `Gender`, `Log_Salary_to_Balance` (83% importance via XGBoost).

## Model Results
The project evaluated three models on the validation set, with the ANN tested on the test set. Results are summarized below:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Threshold |
|-------|----------|-----------|--------|----------|---------|-----------|
| Logistic Regression | 0.6995 | 0.3650 | 0.6405 | 0.4650 | 0.7446 | 0.50 |
| XGBoost (Default) | 0.8454 | 0.6504 | 0.5229 | 0.5797 | 0.8278 | 0.50 |
| XGBoost (Tuned, SMOTETomek) | 0.8115 | 0.5297 | 0.6699 | 0.5916 | 0.8227 | 0.35 |
| ANN (Default) | 0.8068 | 0.5189 | 0.7190 | 0.6027 | 0.8436 | 0.50 |
| ANN (Test Set) | 0.7760 | 0.4675 | 0.7059 | 0.5625 | 0.8421 | 0.50 |

**Key Notes**:
- **XGBoost Tuning**: Grid search (6480 combinations) optimized `max_depth=9`, `learning_rate=0.1`, `n_estimators=200`, `subsample=0.8`, `colsample_bytree=1.0`, `gamma=0.1`. Threshold=0.35 improved recall to 0.6699.
- **ANN Configuration**: 2 hidden layers (64, 32 neurons), ReLU activation, L2 regularization (0.01), dropout (0.3), trained for 100 epochs. No threshold improved recall > 0.75 while maintaining precision > 0.50 and F1 ≥ 0.60.
- **Test Set**: ANN maintained strong performance (recall=0.7059, ROC-AUC=0.8421), confirming robustness.

## Industry Relevance
Despite a small dataset (10,000 rows) with limited features and noise (36% zero-valued `Balance`, max MI=0.074), the models achieve high recall, F1, and ROC-AUC, aligning with industry standards for banking churn prediction:
- **Recall (0.65–0.75)**: ANN recall=0.7190 (validation) and 0.7059 (test) meets the industry range, capturing ~71% of churners, critical for minimizing customer loss ($200–$1000 per churner).
- **F1-Score (0.60–0.65)**: ANN F1=0.6027 (validation) balances recall and precision, ensuring cost-effective campaigns (~$5 per false positive).
- **ROC-AUC**: ANN ROC-AUC=0.8436 (validation) and 0.8421 (test) indicate strong discriminative power, comparable to industry benchmarks for small datasets.
- **Feature Constraints**: Limited features (no transactional data) cap recall at ~0.75, consistent with industry challenges in small, sparse datasets. Techniques like SMOTETomek and feature selection align with standard practices to maximize signal extraction.

## Key Learnings
- **Data Imbalance**: SMOTETomek balanced the 20.37% minority class, boosting recall (e.g., XGBoost from 0.5229 to 0.6699), but synthetic samples can’t overcome low feature signal (MI=0.074).
- **Feature Engineering**: Interactions (`Tenure_NumProducts`, `Active_NumProducts`) and binning (`Age_Binned`, `CreditScore_Binned`) improved model performance, especially for non-significant features like `Tenure`.
- **Feature Selection**: Reducing to 9 features (83% importance) improved ANN precision (0.5189) and F1 (0.6027), preventing overfitting in a small dataset.
- **Model Choice**: ANN outperformed XGBoost in recall (0.7190 vs. 0.6699) and F1 (0.6027 vs. 0.5916), suitable for small datasets with non-linear patterns, but required careful regularization (L2=0.01, dropout=0.3).
- **Threshold Tuning**: Lowering XGBoost threshold to 0.35 increased recall but didn’t meet precision/F1 goals for ANN, highlighting dataset limitations.

## Installation and Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Vraj-Data-Scientist/customer-churn-prediction-in-banking
   cd customer-churn-prediction
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imblearn`, `tensorflow`, `xgboost`.
3. **Run the Notebook**:
   - Open `customer-churn-prediction-in-banking.ipynb` in Jupyter Notebook.
   - Ensure `Churn_Modelling.csv` is in the project directory.
   - Execute cells sequentially to preprocess data, train models, and evaluate results.
4. **Hardware Note**: XGBoost uses GPU (`tree_method='hist'`, `device='cuda'`); ANN training is slow without GPU (no GPU detected in notebook).

## References
- Reichheld, F. F. (2001). *The Loyalty Effect*. Harvard Business Review.
- XGBoost Documentation, 2025. https://xgboost.readthedocs.io/en/stable/
- TensorFlow Documentation, 2025. https://www.tensorflow.org/
- Goodfellow, I., et al. (2016). *Deep Learning*. MIT Press.
- Hastie, T., et al. (2009). *The Elements of Statistical Learning*. Springer.
