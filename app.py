# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# Title of the Streamlit App
st.title("🔍 Anomaly Detection Web Application")
st.write("""
Upload your dataset and select a model to detect anomalies in the data.
""")

# File uploader to upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

# Sidebar for model selection
st.sidebar.header("Select Model for Anomaly Detection")
model_choice = st.sidebar.selectbox("Choose Model", ["Local Outlier Factor", "Isolation Forest", "One-Class SVM"])

# Load dataset
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("✅ **Dataset Loaded Successfully!**")

    # Display a preview of the data
    st.write("### Preview of Dataset")
    st.dataframe(data.head())

    # Select features and target variable
    if 'Class' in data.columns:
        X = data.drop('Class', axis=1)
        Y = data['Class']
    else:
        st.error("❌ 'Class' column not found in the dataset. Please make sure to include the target column.")
        st.stop()

    # Define models
    classifiers = {
        "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=0.01),
        "Isolation Forest": IsolationForest(n_estimators=100, contamination=0.01, random_state=42),
        "One-Class SVM": OneClassSVM(nu=0.01, kernel="rbf", gamma=0.1)
    }

    # Fit and predict using selected model
    clf = classifiers[model_choice]

    st.write(f"### Model Selected: {model_choice}")
    if model_choice == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_prediction = clf.negative_outlier_factor_
    elif model_choice == "One-Class SVM":
        clf.fit(X)
        y_pred = clf.predict(X)
    else:
        clf.fit(X)
        y_pred = clf.predict(X)

    # Convert predictions to 0 for Normal and 1 for Fraud
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    # Calculate errors
    n_errors = (y_pred != Y).sum()

    # Display model performance
    st.write("### Model Performance")
    st.write(f"**Number of Errors:** {n_errors}")
    st.write(f"**Accuracy Score:** {accuracy_score(Y, y_pred):.4f}")

    # Show classification report
    st.write("### Classification Report")
    report = classification_report(Y, y_pred, output_dict=True)
    st.write(pd.DataFrame(report).transpose())

    # Plot Heatmap
    st.write("### Heatmap of Correlation")
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)

    # Plot Anomalies
    st.write("### Anomaly Visualization")
    plt.figure(figsize=(12, 6))
    normal = X[Y == 0]
    anomalies = X[Y == 1]
    plt.scatter(normal.iloc[:, 0], normal.iloc[:, 1], label='Normal Transactions', c='blue')
    plt.scatter(anomalies.iloc[:, 0], anomalies.iloc[:, 1], label='Anomalies', c='red')
    plt.title(f"Anomaly Detection using {model_choice}")
    plt.xlabel('V11')
    plt.ylabel('V12')
    plt.legend()
    st.pyplot(plt)

else:
    st.write("📂 Please upload a CSV file to proceed.")
