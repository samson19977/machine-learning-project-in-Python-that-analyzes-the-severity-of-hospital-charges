#!/usr/bin/env python3
"""
Hospital Charges Prediction

This script predicts medical costs (hospital charges) using patient data.
It trains and compares several regression models: Linear Regression,
Random Forest, Gradient Boosting, and XGBoost. The best model is saved
for later use. Supports training, evaluation, and prediction modes.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------- Configuration --------------------
CONFIG = {
    # Data
    'data_path': Path('data/hospital_charges.csv'),   # default dataset location
    'test_size': 0.2,
    'random_state': 42,

    # Features
    'categorical_features': ['sex', 'region', 'smoker'],
    'numerical_features': ['age', 'bmi', 'children'],
    'interaction_features': [('bmi', 'smoker'), ('age', 'children')],   # pairs for interaction
    'target': 'charges',

    # Models
    'models': {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(objective='reg:squarederror', random_state=42)
    },

    # Outputs
    'output_dir': Path('./output'),
    'models_dir': Path('./models'),
    'plots_dir': Path('./plots'),
    'best_model_name': 'best_model.pkl',
    'results_file': 'model_comparison.csv',
}

# Create directories
for d in [CONFIG['output_dir'], CONFIG['models_dir'], CONFIG['plots_dir']]:
    d.mkdir(parents=True, exist_ok=True)

# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# -------------------- Data Loading and Preprocessing --------------------
def load_data(file_path):
    """Load dataset from CSV."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)
    logger.info(f"Loaded data from {file_path}, shape: {df.shape}")
    return df

def preprocess_data(df, config):
    """Handle missing values, create interaction features, and prepare for modeling."""
    # Drop missing values
    initial_len = len(df)
    df = df.dropna()
    if len(df) < initial_len:
        logger.warning(f"Dropped {initial_len - len(df)} rows with missing values")

    # Create interaction features
    for f1, f2 in config['interaction_features']:
        # Convert smoker to numeric if needed
        if f2 == 'smoker' and df[f2].dtype == 'object':
            df[f'{f1}_{f2}'] = df[f1] * (df[f2] == 'yes').astype(int)
        else:
            df[f'{f1}_{f2}'] = df[f1] * df[f2]
        logger.info(f"Created interaction feature: {f1}_{f2}")

    # Update numerical features list to include interaction features
    numerical_features = config['numerical_features'].copy()
    for f1, f2 in config['interaction_features']:
        numerical_features.append(f'{f1}_{f2}')

    return df, numerical_features

def create_preprocessor(numerical_features, categorical_features):
    """Build column transformer for preprocessing."""
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    return preprocessor

# -------------------- Exploratory Data Analysis --------------------
def perform_eda(df, target, output_dir):
    """Generate exploratory plots and save them."""
    logger.info("Performing exploratory data analysis...")

    # Distribution of target
    plt.figure(figsize=(8, 6))
    sns.histplot(df[target], bins=30, kde=True)
    plt.title("Distribution of Hospital Charges")
    plt.xlabel("Charges ($)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / 'target_distribution.png', dpi=300)
    plt.show()

    # Correlation heatmap (only numeric)
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        plt.figure(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300)
        plt.show()

    # Boxplot by smoker
    if 'smoker' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='smoker', y=target, data=df)
        plt.title("Hospital Charges by Smoking Status")
        plt.tight_layout()
        plt.savefig(output_dir / 'boxplot_by_smoker.png', dpi=300)
        plt.show()

# -------------------- Model Training and Evaluation --------------------
def train_and_evaluate(models, preprocessor, X_train, y_train, X_test, y_test, output_dir):
    """Train all models, evaluate, and return results."""
    results = {}
    best_model = None
    best_r2 = -np.inf
    best_name = None

    for name, model in models.items():
        logger.info(f"Training {name}...")
        pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'MSE': mse, 'R2': r2}
        logger.info(f"{name} - MSE: {mse:.2f}, R2: {r2:.2f}")

        # Plot actual vs predicted
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel("Actual Charges")
        plt.ylabel("Predicted Charges")
        plt.title(f"{name} - Actual vs Predicted")
        plt.tight_layout()
        plt.savefig(output_dir / f'actual_vs_predicted_{name.replace(" ", "_")}.png', dpi=300)
        plt.show()

        # Track best model
        if r2 > best_r2:
            best_r2 = r2
            best_model = pipeline
            best_name = name

    return results, best_model, best_name

def plot_model_comparison(results_df, output_dir):
    """Plot bar chart of model R² scores."""
    plt.figure(figsize=(10, 6))
    sns.barplot(x=results_df.index, y='R2', data=results_df)
    plt.title("Model Performance Comparison (R² Score)")
    plt.ylabel("R² Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300)
    plt.show()

# -------------------- Prediction --------------------
def predict_new_data(model_path, new_data, preprocessor, numerical_features, categorical_features):
    """Load saved pipeline and predict for new data."""
    import joblib
    pipeline = joblib.load(model_path)
    # Ensure new_data has the same features as training
    # Preprocess new_data (interaction features should already be present)
    prediction = pipeline.predict(new_data)
    return prediction[0]

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description='Hospital Charges Prediction')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'evaluate', 'predict'],
                        help='Mode: train, evaluate, or predict')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to CSV data (for train/evaluate)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to saved model (for evaluate/predict)')
    parser.add_argument('--new_data', type=str, default=None,
                        help='Path to CSV with new data for prediction (must have same columns as training)')
    args = parser.parse_args()

    if args.mode == 'train':
        # Load data
        data_path = Path(args.data_path) if args.data_path else CONFIG['data_path']
        df = load_data(data_path)

        # Preprocess
        df, numerical_features = preprocess_data(df, CONFIG)
        # Separate features and target
        X = df.drop(CONFIG['target'], axis=1)
        y = df[CONFIG['target']]

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=CONFIG['test_size'], random_state=CONFIG['random_state']
        )
        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # EDA
        perform_eda(df, CONFIG['target'], CONFIG['plots_dir'])

        # Preprocessor
        preprocessor = create_preprocessor(numerical_features, CONFIG['categorical_features'])

        # Train models
        results, best_model, best_name = train_and_evaluate(
            CONFIG['models'], preprocessor, X_train, y_train, X_test, y_test, CONFIG['plots_dir']
        )

        # Save comparison results
        results_df = pd.DataFrame(results).T
        results_df.to_csv(CONFIG['output_dir'] / CONFIG['results_file'])
        logger.info(f"Model comparison saved to {CONFIG['output_dir'] / CONFIG['results_file']}")

        # Plot comparison
        plot_model_comparison(results_df, CONFIG['plots_dir'])

        # Save best model
        import joblib
        best_model_path = CONFIG['models_dir'] / CONFIG['best_model_name']
        joblib.dump(best_model, best_model_path)
        logger.info(f"Best model ({best_name}) saved to {best_model_path}")

        print("\n=== Best Model ===")
        print(f"Model: {best_name}")
        print(f"R² Score: {results[best_name]['R2']:.3f}")

    elif args.mode == 'evaluate':
        if not args.model_path:
            logger.error("Please provide --model_path for evaluation mode.")
            sys.exit(1)
        if not args.data_path:
            logger.error("Please provide --data_path for evaluation mode.")
            sys.exit(1)

        # Load data and preprocess
        df = load_data(Path(args.data_path))
        df, numerical_features = preprocess_data(df, CONFIG)
        X = df.drop(CONFIG['target'], axis=1)
        y = df[CONFIG['target']]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=CONFIG['test_size'], random_state=CONFIG['random_state']
        )

        # Load preprocessor (should be part of saved model)
        import joblib
        pipeline = joblib.load(Path(args.model_path))
        y_pred = pipeline.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Model evaluation on test set:")
        print(f"  MSE: {mse:.2f}")
        print(f"  R²:  {r2:.3f}")

    elif args.mode == 'predict':
        if not args.model_path:
            logger.error("Please provide --model_path for prediction mode.")
            sys.exit(1)
        if not args.new_data:
            logger.error("Please provide --new_data CSV file.")
            sys.exit(1)

        # Load new data
        new_df = pd.read_csv(Path(args.new_data))
        logger.info(f"Loaded new data: {new_df.shape}")

        # Preprocess new data (must have same columns as training)
        # Note: The model's preprocessing pipeline expects the same features.
        # We assume the new data already contains interaction features (or we recreate them)
        # For simplicity, we'll use the same preprocessing function.
        new_df, _ = preprocess_data(new_df, CONFIG)   # will add interaction features
        # Ensure the columns match the ones the pipeline expects
        # We'll just pass the new_df as is; the pipeline will handle it if we saved the full pipeline.

        import joblib
        pipeline = joblib.load(Path(args.model_path))
        predictions = pipeline.predict(new_df)
        print("Predictions:")
        for i, pred in enumerate(predictions):
            print(f"Sample {i+1}: ${pred:.2f}")

if __name__ == '__main__':
    main()
