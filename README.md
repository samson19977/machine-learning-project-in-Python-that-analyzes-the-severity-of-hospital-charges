# Hospital Charges Prediction

This project predicts medical costs (hospital charges) using patient demographic and health data. It trains and compares multiple regression models: Linear Regression, Random Forest, Gradient Boosting, and XGBoost. The best model is saved for later use.

## Dataset

The script expects a CSV file with the following columns (similar to the [Medical Cost Personal Dataset](https://www.kaggle.com/mirichoi0218/insurance)):

- `age`: age of primary beneficiary
- `sex`: insurance contractor gender (female, male)
- `bmi`: Body Mass Index
- `children`: number of children covered by health insurance
- `smoker`: smoking status (yes, no)
- `region`: beneficiary's residential area (northeast, southeast, southwest, northwest)
- `charges`: individual medical costs billed by health insurance (target)

If you have a different dataset, adjust the column names in the configuration.

## Requirements

- Python 3.8+
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- xgboost
- joblib

All dependencies are listed in `requirements.txt`.

