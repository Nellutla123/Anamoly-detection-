
# 🕵️‍♂️ Credit Card Fraud Detection Using Anomaly Detection Techniques

## 📚 **Project Overview**
This project focuses on detecting fraudulent credit card transactions using anomaly detection techniques. The dataset used is obtained from [Kaggle's Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

The dataset has been pre-processed using **Principal Component Analysis (PCA)** and appropriate feature engineering techniques. The primary goal is to analyze the dataset and detect anomalies (outliers), representing fraudulent transactions, using three different algorithms:

- 📡 **Local Outlier Factor (LOF)**
- 🌲 **Isolation Forest (IF)**
- ⚡️ **One-Class SVM (OCSVM)**

---

## 📊 **Dataset Information**
- **Source:** [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Fraudulent Transactions:** 492 (0.17% of the data)
- **Number of Features:** 30 (28 PCA components + Time + Amount)
- **Class Distribution:**
  - 0: Normal transactions
  - 1: Fraudulent transactions

---

## 🚀 **Goal and Objectives**
- Detect fraudulent transactions by identifying anomalies.
- Compare the performance of different anomaly detection techniques.
- Evaluate model performance using classification metrics.

---

## 🛠️ **Techniques Used for Anomaly Detection**

### 📡 1. Local Outlier Factor (LOF)
- Identifies anomalies by comparing the density of a point with its neighbors.
- **Advantage:** Fast for large datasets and effective at capturing local anomalies.

### 🌲 2. Isolation Forest (IF)
- Randomly partitions data and isolates anomalies through fewer splits.
- **Advantage:** Handles high-dimensional data efficiently and detects global anomalies.

### ⚡️ 3. One-Class SVM (OCSVM)
- Fits a hyperplane to separate normal data points from anomalies.
- **Advantage:** Suitable for high-dimensional data and can capture complex relationships.



## 📊 **Model Performance and Results**

### ✅ **1. Model Accuracy and Errors**
The models were evaluated using accuracy, precision, recall, and F1-score.

| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Local Outlier Factor | 0.99     | 0.82      | 0.76   | 0.79     |
| Isolation Forest     | 0.98     | 0.75      | 0.68   | 0.71     |
| One-Class SVM        | 0.97     | 0.70      | 0.63   | 0.66     |


### 📈 **3. Visualizing Anomalies**
Here’s an example visualization of anomalies detected using different models:

- **Red Points:** Anomalies (Fraudulent Transactions)
- **Blue Points:** Normal Transactions

---

## 📝 **Key Findings and Conclusion**
- **LOF** performed the best with the highest accuracy and F1-score.
- **Isolation Forest** is faster and works better for larger datasets.
- **One-Class SVM** is computationally intensive and may take longer for large datasets.

✅ **Final Verdict:** For credit card fraud detection, LOF offers the best performance in terms of precision and recall.


## 📚 **References**
1. [Kaggle - Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. [Local Outlier Factor (LOF)](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)
3. [Isolation Forest Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
4. [One-Class SVM Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html)





