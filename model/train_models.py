"""
Machine Learning Classification Models for Bank Marketing Dataset
This module implements 6 classification models:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

Each model is evaluated with: Accuracy, AUC, Precision, Recall, F1, MCC
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data(train_path='bank-additional-train.csv', 
                              test_path='bank-additional-test.csv'):
    """
    Load and preprocess the bank marketing dataset.
    Returns preprocessed train and test data.
    """
    # Load data
    train_df = pd.read_csv(train_path, sep=';')
    test_df = pd.read_csv(test_path, sep=';')
    
    # Combine for consistent encoding
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    # Identify categorical columns
    categorical_cols = combined.select_dtypes(include=['object']).columns.tolist()
    categorical_cols.remove('y')  # Remove target variable
    
    # Label encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))
        label_encoders[col] = le
    
    # Encode target variable
    target_encoder = LabelEncoder()
    combined['y'] = target_encoder.fit_transform(combined['y'])
    
    # Split back into train and test
    train_processed = combined.iloc[:len(train_df)]
    test_processed = combined.iloc[len(train_df):]
    
    # Separate features and target
    X_train = train_processed.drop('y', axis=1)
    y_train = train_processed['y']
    X_test = test_processed.drop('y', axis=1)
    y_test = test_processed['y']
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save preprocessors
    joblib.dump(label_encoders, 'model/label_encoders.pkl')
    joblib.dump(target_encoder, 'model/target_encoder.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, X_train.columns.tolist()


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a model and return all metrics.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    metrics = {
        'Model': model_name,
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'AUC': round(roc_auc_score(y_test, y_pred_proba), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall': round(recall_score(y_test, y_pred), 4),
        'F1': round(f1_score(y_test, y_pred), 4),
        'MCC': round(matthews_corrcoef(y_test, y_pred), 4)
    }
    
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return metrics, cm, report


def train_all_models(X_train, X_test, y_train, y_test):
    """
    Train all 6 classification models and return results.
    """
    results = []
    confusion_matrices = {}
    classification_reports = {}
    
    # 1. Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    joblib.dump(lr_model, 'model/logistic_regression.pkl')
    metrics, cm, report = evaluate_model(lr_model, X_test, y_test, 'Logistic Regression')
    results.append(metrics)
    confusion_matrices['Logistic Regression'] = cm
    classification_reports['Logistic Regression'] = report
    
    # 2. Decision Tree
    print("Training Decision Tree...")
    dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt_model.fit(X_train, y_train)
    joblib.dump(dt_model, 'model/decision_tree.pkl')
    metrics, cm, report = evaluate_model(dt_model, X_test, y_test, 'Decision Tree')
    results.append(metrics)
    confusion_matrices['Decision Tree'] = cm
    classification_reports['Decision Tree'] = report
    
    # 3. K-Nearest Neighbors
    print("Training K-Nearest Neighbors...")
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    joblib.dump(knn_model, 'model/knn.pkl')
    metrics, cm, report = evaluate_model(knn_model, X_test, y_test, 'kNN')
    results.append(metrics)
    confusion_matrices['kNN'] = cm
    classification_reports['kNN'] = report
    
    # 4. Naive Bayes (Gaussian)
    print("Training Naive Bayes (Gaussian)...")
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    joblib.dump(nb_model, 'model/naive_bayes.pkl')
    metrics, cm, report = evaluate_model(nb_model, X_test, y_test, 'Naive Bayes')
    results.append(metrics)
    confusion_matrices['Naive Bayes'] = cm
    classification_reports['Naive Bayes'] = report
    
    # 5. Random Forest (Ensemble)
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, 'model/random_forest.pkl')
    metrics, cm, report = evaluate_model(rf_model, X_test, y_test, 'Random Forest')
    results.append(metrics)
    confusion_matrices['Random Forest'] = cm
    classification_reports['Random Forest'] = report
    
    # 6. XGBoost (Ensemble)
    print("Training XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=100, 
        max_depth=6, 
        learning_rate=0.1, 
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, 'model/xgboost.pkl')
    metrics, cm, report = evaluate_model(xgb_model, X_test, y_test, 'XGBoost')
    results.append(metrics)
    confusion_matrices['XGBoost'] = cm
    classification_reports['XGBoost'] = report
    
    return pd.DataFrame(results), confusion_matrices, classification_reports


def main():
    """
    Main function to train and evaluate all models.
    """
    print("=" * 60)
    print("Bank Marketing Classification - Model Training")
    print("=" * 60)
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()
    
    # Save feature names
    joblib.dump(feature_names, 'model/feature_names.pkl')
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")
    
    # Train all models
    print("\n" + "=" * 60)
    print("Training Models...")
    print("=" * 60)
    
    results_df, confusion_matrices, classification_reports = train_all_models(
        X_train, X_test, y_train, y_test
    )
    
    # Save results
    results_df.to_csv('model/model_comparison.csv', index=False)
    joblib.dump(confusion_matrices, 'model/confusion_matrices.pkl')
    joblib.dump(classification_reports, 'model/classification_reports.pkl')
    
    # Display results
    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    # Find best model for each metric
    print("\n" + "=" * 60)
    print("BEST MODEL FOR EACH METRIC")
    print("=" * 60)
    
    for metric in ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']:
        best_idx = results_df[metric].idxmax()
        best_model = results_df.loc[best_idx, 'Model']
        best_value = results_df.loc[best_idx, metric]
        print(f"{metric}: {best_model} ({best_value})")
    
    print("\n" + "=" * 60)
    print("Training complete! Models saved in 'model/' directory.")
    print("=" * 60)
    
    return results_df


if __name__ == "__main__":
    results = main()
