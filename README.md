# Bank Marketing Classification - ML Assignment 2

## 💳 Problem Statement

The goal of this project is to predict whether a client will subscribe to a term deposit based on their demographic information, past campaign interactions, and economic indicators. This is a **binary classification problem** where the target variable is whether the client subscribed to a term deposit (`yes` or `no`).

The business objective is to help the bank identify potential customers who are more likely to subscribe to term deposits, enabling more efficient and targeted marketing campaigns.

## �️ Dataset Description

### Source
- **Dataset:** Bank Marketing Dataset (bank-additional-full.csv)
- **Source:** UCI Machine Learning Repository
- **Domain:** Banking/Marketing

### Dataset Statistics
- **Total Instances:** 41,188
- **Training Set:** 32,950 (80%)
- **Test Set:** 8,238 (20%)
- **Number of Features:** 20
- **Target Variable:** `y` (yes/no - term deposit subscription)

### Class Distribution
| Class | Training Set | Test Set |
|-------|-------------|----------|
| No (0) | 29,238 | 7,310 |
| Yes (1) | 3,712 | 928 |

**Note:** The dataset is imbalanced with approximately 88.7% negative class and 11.3% positive class.

### Feature Description

| Feature | Type | Description |
|---------|------|-------------|
| **age** | Numeric | Client's age |
| **job** | Categorical | Type of job (admin, blue-collar, entrepreneur, etc.) |
| **marital** | Categorical | Marital status (married, single, divorced) |
| **education** | Categorical | Education level (basic.4y, high.school, university.degree, etc.) |
| **default** | Categorical | Has credit in default? (yes, no, unknown) |
| **housing** | Categorical | Has housing loan? (yes, no, unknown) |
| **loan** | Categorical | Has personal loan? (yes, no, unknown) |
| **contact** | Categorical | Contact communication type (cellular, telephone) |
| **month** | Categorical | Last contact month of year |
| **day_of_week** | Categorical | Last contact day of the week |
| **duration** | Numeric | Last contact duration in seconds |
| **campaign** | Numeric | Number of contacts during this campaign |
| **pdays** | Numeric | Days since client was last contacted (999 = never contacted) |
| **previous** | Numeric | Number of contacts before this campaign |
| **poutcome** | Categorical | Outcome of previous marketing campaign |
| **emp.var.rate** | Numeric | Employment variation rate (quarterly) |
| **cons.price.idx** | Numeric | Consumer price index (monthly) |
| **cons.conf.idx** | Numeric | Consumer confidence index (monthly) |
| **euribor3m** | Numeric | Euribor 3 month rate (daily) |
| **nr.employed** | Numeric | Number of employees (quarterly) |

## ⚙️ Models Used

Six classification models were implemented and evaluated:

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.9139 | 0.9370 | 0.7002 | 0.4127 | 0.5193 | 0.4956 |
| Decision Tree | 0.9147 | 0.8920 | 0.6347 | 0.5711 | 0.6012 | 0.5546 |
| kNN | 0.9053 | 0.8617 | 0.6267 | 0.3944 | 0.4841 | 0.4491 |
| Naive Bayes | 0.8536 | 0.8606 | 0.4024 | 0.6175 | 0.4872 | 0.4189 |
| Random Forest (Ensemble) | 0.9212 | 0.9519 | 0.7025 | 0.5216 | 0.5986 | 0.5636 |
| XGBoost (Ensemble) | 0.9227 | 0.9549 | 0.6907 | 0.5679 | 0.6233 | 0.5841 |

### Best Model by Metric
- **Accuracy:** XGBoost (0.9227)
- **AUC:** XGBoost (0.9549)
- **Precision:** Random Forest (0.7025)
- **Recall:** Naive Bayes (0.6175)
- **F1:** XGBoost (0.6233)
- **MCC:** XGBoost (0.5841)

## � Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Provides a strong baseline with good AUC (0.9370), indicating excellent discriminative ability. High precision (0.7002) but lower recall (0.4127) suggests it's conservative in predicting positive class, missing many actual subscribers. Suitable when false positives are costly. The linear decision boundary works reasonably well for this dataset. |
| **Decision Tree** | Shows balanced performance between precision and recall. Achieves the second-best recall (0.5711) among non-Bayesian models. The model is interpretable and captures non-linear relationships. However, lower AUC (0.8920) compared to ensemble methods indicates some overfitting tendencies despite max_depth constraint. |
| **kNN** | Lowest overall performance among the models. The instance-based approach struggles with the high-dimensional feature space (20 features) and class imbalance. Recall (0.3944) is particularly low, indicating difficulty in identifying positive cases. Performance could potentially improve with feature selection and different k values. |
| **Naive Bayes** | Achieves the highest recall (0.6175) among all models, making it effective at identifying potential subscribers. However, this comes at the cost of lowest precision (0.4024) and accuracy (0.8536), resulting in many false positives. Best suited when maximizing identification of potential customers is priority over precision. The independence assumption may not perfectly hold for correlated features. |
| **Random Forest (Ensemble)** | Strong ensemble performance with best precision (0.7025) and second-best overall metrics. The bagging approach effectively reduces variance and handles feature interactions well. Achieves excellent AUC (0.9519), indicating robust probability estimates. Good balance between bias and variance, making it a reliable choice for production deployment. |
| **XGBoost (Ensemble)** | Best overall performer across most metrics. Achieves highest accuracy (0.9227), AUC (0.9549), F1 (0.6233), and MCC (0.5841). The gradient boosting approach effectively handles class imbalance and captures complex patterns. Provides the best balance between precision (0.6907) and recall (0.5679). Recommended as the primary model for deployment due to its superior generalization ability. |

## ▶️ Streamlit App Features

The deployed Streamlit application includes:

1. **Dataset Upload Option (CSV)**
   - Upload custom test data for predictions
   - Option to use default test data
   - Data preview functionality

2. **Model Selection Dropdown**
   - Select from 6 trained classification models
   - Easy switching between models for comparison

3. **Display of Evaluation Metrics**
   - Real-time calculation of all 6 metrics
   - Visual comparison charts across all models
   - Interactive metric cards

4. **Confusion Matrix and Classification Report**
   - Heatmap visualization of confusion matrix
   - Detailed classification report with precision, recall, and F1 per class
   - Interpretation of results

## � Project Structure

```
project-folder/
│-- app.py                          # Main Streamlit application
│-- requirements.txt                # Python dependencies
│-- README.md                       # Project documentation
│-- bank-additional-full.csv        # Original dataset
│-- bank-additional-train.csv       # Training dataset (80%)
│-- bank-additional-test.csv        # Test dataset (20%)
│-- data_split.py                   # Script to split data
│-- model/
    │-- train_models.py             # Model training script
    │-- logistic_regression.pkl     # Trained Logistic Regression
    │-- decision_tree.pkl           # Trained Decision Tree
    │-- knn.pkl                     # Trained KNN
    │-- naive_bayes.pkl             # Trained Naive Bayes
    │-- random_forest.pkl           # Trained Random Forest
    │-- xgboost.pkl                 # Trained XGBoost
    │-- label_encoders.pkl          # Feature encoders
    │-- target_encoder.pkl          # Target variable encoder
    │-- scaler.pkl                  # Feature scaler
    │-- feature_names.pkl           # Feature column names
    │-- model_comparison.csv        # Model comparison results
    │-- confusion_matrices.pkl      # Stored confusion matrices
    │-- classification_reports.pkl  # Stored classification reports
```

## ⚒️ Installation & Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <project-folder>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train models (if needed):
```bash
python model/train_models.py
```

4. Run the Streamlit app locally:
```bash
streamlit run app.py
```

## 🔗 Deployment

The application is deployed on **Streamlit Community Cloud**.

**Live App Link:** [Insert Streamlit App URL Here]

## ⊞ Technologies Used

- **Python 3.x**
- **Scikit-learn** - Machine Learning algorithms
- **XGBoost** - Gradient Boosting
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Matplotlib/Seaborn** - Visualization
- **Streamlit** - Web application framework
- **Joblib** - Model serialization

## ✎ Author
Rahul Bhave
## Developed for 
**M.Tech (AIML/DSE) - Machine Learning Assignment 2**  
BITS Pilani - Work Integrated Learning Programmes Division

---

## ⌘ References

1. [UCI Machine Learning Repository - Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
2. [Scikit-learn Documentation](https://scikit-learn.org/stable/)
3. [XGBoost Documentation](https://xgboost.readthedocs.io/)
4. [Streamlit Documentation](https://docs.streamlit.io/)
