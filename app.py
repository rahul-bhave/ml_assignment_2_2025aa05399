"""
Streamlit Web Application for Bank Marketing Data Classification
This app allows users to:
1. Upload test data (CSV)
2. Select classification model
3. View evaluation metrics
4. View confusion matrix and classification report
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')


# Page configuration
st.set_page_config(
    page_title="Bank Marketing Dataset Classifier",
    page_icon="💳",
    layout="wide"
)


# Custom CSS for better readability
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Force Streamlit metric styling */
    [data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 2px solid #1f77b4 !important;
        padding: 15px 20px !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.15) !important;
    }
    
    [data-testid="stMetric"] label {
        color: #333333 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #1f77b4 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #28a745 !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    
    /* DataFrame/Table styling - force dark text */
    .stDataFrame, .dataframe {
        font-size: 1rem !important;
    }
    
    [data-testid="stDataFrame"] div[data-testid="stDataFrameGlideCanvas"] {
        color: #000000 !important;
    }
    
    /* Subheader styling */
    h3 {
        color: #1f77b4 !important;
        font-weight: 600 !important;
    }
    
    /* Custom metric card - used for HTML metrics */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 2px solid #1f77b4;
        border-radius: 12px;
        padding: 20px 15px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 5px 0;
    }
    
    .metric-card .metric-label {
        color: #555555;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    
    .metric-card .metric-value {
        color: #1f77b4;
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Best model card with gradient */
    .best-model-card {
        background: linear-gradient(135deg, #1f77b4 0%, #2e86de 100%);
        border-radius: 12px;
        padding: 20px 15px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        margin: 5px 0;
    }
    
    .best-model-card .metric-label {
        color: #e8f4fd;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 5px;
    }
    
    .best-model-card .metric-model {
        color: #ffffff;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 3px;
    }
    
    .best-model-card .metric-value {
        color: #ffd700;
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Prediction metric cards */
    .pred-metric-card {
        background: #ffffff;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 15px 10px;
        text-align: center;
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
    }
    
    .pred-metric-card .metric-label {
        color: #444444;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 5px;
    }
    
    .pred-metric-card .metric-value {
        color: #28a745;
        font-size: 1.6rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


def render_metric_card(label, value, card_type="default"):
    """Render a custom metric card with HTML for better visibility."""
    if card_type == "prediction":
        return f"""
        <div class="pred-metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """
    else:
        return f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """


def render_best_model_card(label, model_name, value):
    """Render a best model card with HTML."""
    return f"""
    <div class="best-model-card">
        <div class="metric-label">{label}</div>
        <div class="metric-model">{model_name}</div>
        <div class="metric-value">{value}</div>
    </div>
    """


@st.cache_resource
def load_models():
    """Load all trained models and preprocessors."""
    models = {
        'Logistic Regression': joblib.load('model/logistic_regression.pkl'),
        'Decision Tree': joblib.load('model/decision_tree.pkl'),
        'kNN': joblib.load('model/knn.pkl'),
        'Naive Bayes': joblib.load('model/naive_bayes.pkl'),
        'Random Forest': joblib.load('model/random_forest.pkl'),
        'XGBoost': joblib.load('model/xgboost.pkl')
    }
    
    label_encoders = joblib.load('model/label_encoders.pkl')
    target_encoder = joblib.load('model/target_encoder.pkl')
    scaler = joblib.load('model/scaler.pkl')
    feature_names = joblib.load('model/feature_names.pkl')
    
    return models, label_encoders, target_encoder, scaler, feature_names


@st.cache_data
def load_comparison_results():
    """Load pre-computed model comparison results."""
    results_df = pd.read_csv('model/model_comparison.csv')
    confusion_matrices = joblib.load('model/confusion_matrices.pkl')
    classification_reports = joblib.load('model/classification_reports.pkl')
    return results_df, confusion_matrices, classification_reports


def preprocess_data(df, label_encoders, target_encoder, scaler, feature_names):
    """Preprocess uploaded data for prediction."""
    df_processed = df.copy()
    
    # Get categorical columns (same as training)
    categorical_cols = [col for col in label_encoders.keys()]
    
    # Label encode categorical variables
    for col in categorical_cols:
        if col in df_processed.columns:
            le = label_encoders[col]
            # Handle unseen labels
            df_processed[col] = df_processed[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    # Encode target if present
    has_target = 'y' in df_processed.columns
    if has_target:
        y = df_processed['y'].apply(
            lambda x: target_encoder.transform([x])[0] if x in target_encoder.classes_ else -1
        ).values
        df_processed = df_processed.drop('y', axis=1)
    else:
        y = None
    
    # Ensure columns are in correct order
    X = df_processed[feature_names]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    return X_scaled, y


def plot_confusion_matrix(cm, model_name):
    """Create confusion matrix visualization."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Subscription', 'Subscription'],
                yticklabels=['No Subscription', 'Subscription'])
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14)
    return fig


def plot_metrics_comparison(results_df):
    """Create bar chart comparing all models."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    colors = plt.cm.Set2(np.linspace(0, 1, len(results_df)))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        bars = ax.bar(results_df['Model'], results_df[metric], color=colors)
        ax.set_title(metric, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


def main():
    """Main function to run the Streamlit app."""
    
    # Header
    st.markdown('<h1 class="main-header">💳 Bank Marketing Dataset Classifier</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align: center; font-size: 1.1rem;">
    This application predicts whether a client will subscribe to a term deposit based on their characteristics.
    </p>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Load models and data
    try:
        models, label_encoders, target_encoder, scaler, feature_names = load_models()
        results_df, confusion_matrices, classification_reports = load_comparison_results()
        models_loaded = True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please ensure model files are in the 'model/' directory.")
        models_loaded = False
    
    if models_loaded:
        # Sidebar
        st.sidebar.title("🔧 Configuration")
        
        # Model Selection
        st.sidebar.subheader("Select Model")
        selected_model = st.sidebar.selectbox(
            "Choose a classification model:",
            options=list(models.keys()),
            index=5,  # Default to XGBoost
            help="Select a machine learning model for predictions"
        )
        
        st.sidebar.divider()
        
        # Main content - tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "⚖️ Model Comparison", 
            "📤 Upload & Predict", 
            "🎯 Evaluation Metrics",
            "🔍 Model Observations",
            "📋 About"
        ])
        
        # Tab 1: Model Comparison
        with tab1:
            st.subheader("Model Performance Comparison")
            
            # Display comparison table with improved styling
            st.markdown("### Performance Metrics Table")
            
            # Create HTML table for better font control
            metrics_html = '''
            <style>
                .metrics-table {
                    width: 100%;
                    border-collapse: collapse;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 20px 0;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    border-radius: 8px;
                    overflow: hidden;
                }
                .metrics-table th {
                    background: linear-gradient(135deg, #1f77b4 0%, #2e86de 100%);
                    color: white;
                    font-size: 16px;
                    font-weight: 600;
                    padding: 15px 12px;
                    text-align: center;
                    border: none;
                }
                .metrics-table td {
                    padding: 14px 12px;
                    text-align: center;
                    font-size: 15px;
                    font-weight: 500;
                    color: #333333;
                    background-color: #ffffff;
                    border-bottom: 1px solid #e0e0e0;
                }
                .metrics-table tr:hover td {
                    background-color: #f5f9ff;
                }
                .metrics-table td.model-name {
                    font-weight: 600;
                    text-align: left;
                    padding-left: 20px;
                    color: #1f77b4;
                }
                .metrics-table td.best-value {
                    background-color: #d4edda;
                    color: #155724;
                    font-weight: 700;
                }
            </style>
            <table class="metrics-table">
                <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>AUC</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1</th>
                    <th>MCC</th>
                </tr>
            '''
            
            # Find best values for highlighting
            best_vals = {
                'Accuracy': results_df['Accuracy'].max(),
                'AUC': results_df['AUC'].max(),
                'Precision': results_df['Precision'].max(),
                'Recall': results_df['Recall'].max(),
                'F1': results_df['F1'].max(),
                'MCC': results_df['MCC'].max()
            }
            
            for _, row in results_df.iterrows():
                metrics_html += f'<tr><td class="model-name">{row["Model"]}</td>'
                for metric in ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']:
                    cell_class = 'best-value' if row[metric] == best_vals[metric] else ''
                    metrics_html += f'<td class="{cell_class}">{row[metric]:.4f}</td>'
                metrics_html += '</tr>'
            
            metrics_html += '</table>'
            st.markdown(metrics_html, unsafe_allow_html=True)
            
            # Metrics visualization
            st.markdown("### Visual Comparison")
            fig = plot_metrics_comparison(results_df)
            st.pyplot(fig)
            
            # Best model summary
            st.markdown("### �️ Best Model Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                best_acc = results_df.loc[results_df['Accuracy'].idxmax()]
                st.markdown(render_best_model_card("Best Accuracy", best_acc['Model'], f"{best_acc['Accuracy']:.4f}"), unsafe_allow_html=True)
                
            with col2:
                best_auc = results_df.loc[results_df['AUC'].idxmax()]
                st.markdown(render_best_model_card("Best AUC", best_auc['Model'], f"{best_auc['AUC']:.4f}"), unsafe_allow_html=True)
                
            with col3:
                best_f1 = results_df.loc[results_df['F1'].idxmax()]
                st.markdown(render_best_model_card("Best F1 Score", best_f1['Model'], f"{best_f1['F1']:.4f}"), unsafe_allow_html=True)
        
        # Tab 2: Upload & Predict
        with tab2:
            st.subheader("Upload Test Data for Prediction")
            
            st.info("""
            � **Instructions:**
            - Upload a CSV file with the same structure as the training data
            - The file should be semicolon-separated (;)
            - Include the target column 'y' for evaluation
            - Or use the default test data for quick testing
            """)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                uploaded_file = st.file_uploader(
                    "Choose a CSV file",
                    type=['csv'],
                    help="Upload test data in CSV format"
                )
            
            with col2:
                use_default = st.checkbox("Use default test data", value=True)
            
            if use_default or uploaded_file is not None:
                try:
                    if use_default:
                        df = pd.read_csv('bank-additional-test.csv', sep=';')
                        st.success("☑️ Default test data loaded successfully!")
                    else:
                        df = pd.read_csv(uploaded_file, sep=';')
                        st.success("☑️ File uploaded successfully!")
                    
                    # Show data preview
                    with st.expander("Preview Data", expanded=False):
                        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
                        st.dataframe(df.head(10), use_container_width=True)
                    
                    # Preprocess and predict
                    X_test, y_test = preprocess_data(df, label_encoders, target_encoder, scaler, feature_names)
                    
                    # Make predictions
                    model = models[selected_model]
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                    st.markdown(f"### Predictions using **{selected_model}**")
                    
                    # If we have true labels, show evaluation
                    if y_test is not None:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            acc = accuracy_score(y_test, y_pred)
                            st.markdown(render_metric_card("Accuracy", f"{acc:.4f}", "prediction"), unsafe_allow_html=True)
                        
                        with col2:
                            auc = roc_auc_score(y_test, y_pred_proba)
                            st.markdown(render_metric_card("AUC Score", f"{auc:.4f}", "prediction"), unsafe_allow_html=True)
                        
                        with col3:
                            f1 = f1_score(y_test, y_pred)
                            st.markdown(render_metric_card("F1 Score", f"{f1:.4f}", "prediction"), unsafe_allow_html=True)
                        
                        # Additional metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            prec = precision_score(y_test, y_pred)
                            st.markdown(render_metric_card("Precision", f"{prec:.4f}", "prediction"), unsafe_allow_html=True)
                        
                        with col2:
                            rec = recall_score(y_test, y_pred)
                            st.markdown(render_metric_card("Recall", f"{rec:.4f}", "prediction"), unsafe_allow_html=True)
                        
                        with col3:
                            mcc = matthews_corrcoef(y_test, y_pred)
                            st.markdown(render_metric_card("MCC Score", f"{mcc:.4f}", "prediction"), unsafe_allow_html=True)
                        
                        # Confusion Matrix
                        st.markdown("### Confusion Matrix")
                        cm = confusion_matrix(y_test, y_pred)
                        fig = plot_confusion_matrix(cm, selected_model)
                        st.pyplot(fig)
                        
                        # Classification Report
                        st.markdown("### Classification Report")
                        report = classification_report(y_test, y_pred, 
                                                       target_names=['No Subscription', 'Subscription'],
                                                       output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.round(4), use_container_width=True)
                    
                    else:
                        st.warning("No target column 'y' found. Showing predictions only.")
                        
                        # Show predictions
                        pred_df = df.copy()
                        pred_df['Prediction'] = ['Yes' if p == 1 else 'No' for p in y_pred]
                        pred_df['Probability'] = y_pred_proba
                        
                        st.dataframe(pred_df[['Prediction', 'Probability']].head(20))
                
                except Exception as e:
                    st.error(f"Error processing data: {e}")
        
        # Tab 3: Detailed Evaluation Metrics
        with tab3:
            st.subheader(f"Detailed Evaluation: {selected_model}")
            
            # Get metrics for selected model
            model_metrics = results_df[results_df['Model'] == selected_model].iloc[0]
            
            # Display metrics using custom HTML cards
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                st.markdown(render_metric_card("Accuracy", f"{model_metrics['Accuracy']:.4f}"), unsafe_allow_html=True)
            with col2:
                st.markdown(render_metric_card("AUC", f"{model_metrics['AUC']:.4f}"), unsafe_allow_html=True)
            with col3:
                st.markdown(render_metric_card("Precision", f"{model_metrics['Precision']:.4f}"), unsafe_allow_html=True)
            with col4:
                st.markdown(render_metric_card("Recall", f"{model_metrics['Recall']:.4f}"), unsafe_allow_html=True)
            with col5:
                st.markdown(render_metric_card("F1", f"{model_metrics['F1']:.4f}"), unsafe_allow_html=True)
            with col6:
                st.markdown(render_metric_card("MCC", f"{model_metrics['MCC']:.4f}"), unsafe_allow_html=True)
            
            st.divider()
            
            # Confusion Matrix
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Confusion Matrix")
                cm = confusion_matrices[selected_model]
                fig = plot_confusion_matrix(cm, selected_model)
                st.pyplot(fig)
            
            with col2:
                st.markdown("### Classification Report")
                report = classification_reports[selected_model]
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.round(4), use_container_width=True)
                
                # Interpretation
                st.markdown("### Interpretation")
                st.markdown(f"""
                - **True Negatives:** {cm[0,0]} (Correctly predicted No Subscription)
                - **False Positives:** {cm[0,1]} (Incorrectly predicted Subscription)
                - **False Negatives:** {cm[1,0]} (Missed Subscriptions)
                - **True Positives:** {cm[1,1]} (Correctly predicted Subscription)
                """)
        
        # Tab 4: Model Performance Observations
        with tab4:
            st.subheader("� Model Performance Observations")
            
            st.markdown("""
            The following table presents detailed observations on the performance of each machine learning model 
            trained on the Bank Marketing dataset. These observations are based on evaluation metrics including 
            Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
            """)
            
            # Model observations data
            observations_data = {
                'ML Model Name': [
                    'Logistic Regression',
                    'Decision Tree',
                    'kNN',
                    'Naive Bayes',
                    'Random Forest (Ensemble)',
                    'XGBoost (Ensemble)'
                ],
                'Observation about model performance': [
                    'Achieves strong accuracy (91.39%) and excellent AUC (0.9370), indicating good discriminative ability. '
                    'However, it has lower recall (41.27%), meaning it misses many positive cases. Best suited when '
                    'interpretability and stable performance are priorities. The linear decision boundary limits its '
                    'ability to capture complex non-linear relationships in the data.',
                    
                    'Shows balanced performance with accuracy of 91.47% and the highest recall among non-ensemble models (57.11%). '
                    'AUC of 0.8920 is lower than other models, suggesting it may not rank predictions as effectively. '
                    'Prone to overfitting but provides excellent interpretability through visualizable decision rules. '
                    'Good choice when understanding the decision process is important.',
                    
                    'Delivers moderate performance with 90.53% accuracy but struggles with recall (39.44%), indicating '
                    'difficulty identifying positive class instances. AUC of 0.8617 is the lowest among all models. '
                    'Performance is sensitive to the choice of k and feature scaling. The instance-based approach '
                    'makes it computationally expensive for large datasets and less effective with imbalanced data.',
                    
                    'Lowest accuracy (85.36%) but achieves the highest recall (61.75%) among all models, making it '
                    'excellent at identifying potential subscribers. The assumption of feature independence may not '
                    'hold for this dataset, limiting overall performance. Best suited when minimizing false negatives '
                    'is critical and when quick probabilistic predictions are needed.',
                    
                    'Strong performer with 92.12% accuracy and excellent AUC (0.9519). Provides good balance between '
                    'precision (70.25%) and recall (52.16%). The ensemble approach reduces overfitting compared to '
                    'single decision trees. Robust to outliers and handles the class imbalance reasonably well. '
                    'Offers feature importance insights while maintaining predictive power.',
                    
                    'Best overall performer with highest accuracy (92.27%), AUC (0.9549), and MCC (0.5841). '
                    'Achieves the best F1 score (0.6233), indicating optimal precision-recall balance. '
                    'Gradient boosting effectively handles class imbalance and captures complex feature interactions. '
                    'The sequential learning approach corrects errors from previous iterations, leading to superior '
                    'generalization on this banking dataset.'
                ]
            }
            
            observations_df = pd.DataFrame(observations_data)
            
            # Create HTML table for better font control
            obs_html = '<style>'
            obs_html += '.obs-table { width: 100%; border-collapse: collapse; font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif; margin: 20px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }'
            obs_html += '.obs-table th { background: linear-gradient(135deg, #1f77b4 0%, #2e86de 100%); color: white; font-size: 16px; font-weight: 600; padding: 16px 15px; text-align: left; border: none; }'
            obs_html += '.obs-table td { padding: 16px 15px; font-size: 15px; color: #333333; background-color: #ffffff; border-bottom: 1px solid #e0e0e0; vertical-align: top; line-height: 1.6; }'
            obs_html += '.obs-table tr:hover td { background-color: #f5f9ff; }'
            obs_html += '.obs-table td.model-name { font-weight: 700; font-size: 15px; color: #1f77b4; background-color: #f8f9fa; min-width: 180px; white-space: nowrap; }'
            obs_html += '.obs-table td.observation { font-weight: 400; text-align: justify; }'
            obs_html += '</style>'
            obs_html += '<table class="obs-table">'
            obs_html += '<tr><th style="width: 200px;">ML Model Name</th><th>Observation about model performance</th></tr>'
            
            for _, row in observations_df.iterrows():
                obs_html += f'<tr><td class="model-name">{row["ML Model Name"]}</td><td class="observation">{row["Observation about model performance"]}</td></tr>'
            
            obs_html += '</table>'
            st.markdown(obs_html, unsafe_allow_html=True)
            
            # Key insights summary
            st.markdown("### � Key Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Best for Accuracy & Overall Performance:**
                - ① XGBoost (92.27% accuracy, 0.9549 AUC)
                - ② Random Forest (92.12% accuracy, 0.9519 AUC)
                - ③ Decision Tree (91.47% accuracy)
                
                **Best for Identifying Subscribers (Recall):**
                - ① Naive Bayes (61.75% recall)
                - ② Decision Tree (57.11% recall)
                - ③ XGBoost (56.79% recall)
                """)
            
            with col2:
                st.markdown("""
                **Best for Precision (Avoiding False Positives):**
                - ① Random Forest (70.25% precision)
                - ② Logistic Regression (70.02% precision)
                - ③ XGBoost (69.07% precision)
                
                **Best Balance (F1 Score):**
                - ① XGBoost (0.6233)
                - ② Decision Tree (0.6012)
                - ③ Random Forest (0.5986)
                """)
            
            st.markdown("---")
            st.markdown("""
            **▶ Recommendation:** For this bank marketing dataset with class imbalance (88.7% negative, 11.3% positive), 
            **XGBoost** is recommended as the primary model due to its superior performance across all metrics. 
            However, if interpretability is crucial, **Decision Tree** or **Logistic Regression** may be preferred.
            If minimizing missed potential subscribers is the priority, consider **Naive Bayes** for its high recall.
            """)
        
        # Tab 5: About
        with tab5:
            st.subheader("About This Application")
            
            st.markdown("""
            ### �️ Bank Marketing Dataset
            
            This application uses the **Bank Marketing Dataset** from UCI Machine Learning Repository.
            The data is related to direct marketing campaigns (phone calls) of a Portuguese banking institution.
            
            **Target Variable:** Whether the client subscribed to a term deposit (yes/no)
            
            ### ▣ Features (20 attributes)
            
            | Category | Features |
            |----------|----------|
            | **Client Info** | age, job, marital, education, default, housing, loan |
            | **Campaign Info** | contact, month, day_of_week, duration, campaign, pdays, previous, poutcome |
            | **Economic Indicators** | emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed |
            
            ### ⚙️ Models Implemented
            
            1. **Logistic Regression** - Linear classifier for binary outcomes
            2. **Decision Tree** - Rule-based tree classifier
            3. **K-Nearest Neighbors (kNN)** - Instance-based learning
            4. **Naive Bayes (Gaussian)** - Probabilistic classifier
            5. **Random Forest** - Ensemble of decision trees
            6. **XGBoost** - Gradient boosting ensemble
            
            ### ◈ Evaluation Metrics
            
            - **Accuracy** - Overall correctness
            - **AUC** - Area Under ROC Curve
            - **Precision** - Positive predictive value
            - **Recall** - Sensitivity/True positive rate
            - **F1 Score** - Harmonic mean of precision and recall
            - **MCC** - Matthews Correlation Coefficient
            
            ### ⊕ Developed For
            
            M.Tech (AIML) - Machine Learning Assignment 2  
            BITS Pilani - Work Integrated Learning Programmes October 2025 batch
                        
            
            """)


if __name__ == "__main__":
    main()
