# 🚀 Customer Segmentation System (End-to-End Unsupervised ML + Deployment)

This project implements a production-ready **Customer Segmentation System** using Unsupervised Machine Learning, following a complete ML pipeline with data ingestion, preprocessing, model training, evaluation, and deployment using Streamlit.

The system groups customers into meaningful clusters based on their demographics, spending behavior, purchase history, and engagement levels.  
This segmentation enables targeted marketing, personalized offers, and improved business insights.

---

## 🧠 Problem Statement
Businesses need deep understanding of customer behavior to tailor marketing efforts and maximize ROI.  
Instead of manually categorizing customers, this project uses **K-Means clustering** to automatically segment customers into high-impact behavioral groups.

---

## 📁 Dataset Overview

The dataset contains detailed customer demographic and behavioral attributes used to build segmentation clusters.

### 🔹 Dataset Columns

| Column | Description |
|--------|-------------|
| Income | Annual income of the customer |
| Recency | Days since last purchase |
| MntWines / MntMeat / MntFish / MntSweet / MntGold | Spending in each category |
| NumDealsPurchases | Purchases made using discounts |
| NumWeb / Catalog / Store Purchases | Channel-wise purchases |
| NumWebVisitsMonth | Website visits |
| Kidhome / Teenhome | Household composition |
| Education | Education level |
| Marital_Status | Marital status |
| Response | Campaign response indicator |
| **Age** | Engineered feature (2025 - Year_Birth) |
| **Family_Size** | Engineered (Adults + Kids) |
| **Total_Spending** | Sum of all product category spending |
| **Total_Purchases** | Sum of all purchase channels |
| **Customer_For_Days** | Tenure with company |

---

## 🎯 Target Output
Since this is an **unsupervised problem**, the goal is to assign a **Cluster ID** to each customer.

Example clusters:
- **Cluster 0:** High spenders, loyal customers  
- **Cluster 1:** Low-income, low-engagement users  
- **Cluster 2:** Digital-first medium spenders  
- **Cluster 3:** Family-focused discount buyers  

---

## ⚙️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Feature Engineering | Custom transformations |
| Scaling + Encoding | StandardScaler, OneHotEncoder |
| Clustering Model | K-Means |
| Visualization | Matplotlib, Seaborn, PCA |
| Deployment | Streamlit |
| Version Control | Git, GitHub |
| Logging | Custom Logger |
| Error Handling | Custom Exceptions |
| Serialization | Joblib |

---

## 🔄 ML Pipeline Workflow

### ✅ **Data Ingestion**
- Reads cleaned dataset  
- Saves dataset into `artifacts/raw.csv`, `train.csv`, `test.csv`

### ✅ **Data Transformation**
Handles:
- Scaling numerical features  
- One-hot encoding categorical features  
- Saves preprocessor → `artifacts/preprocessor.pkl`

### ✅ **Model Training (K-Means)**
- Uses **Elbow Method** + **Silhouette Score**  
- Trains final K-Means model  
- Saves model → `artifacts/kmeans.pkl`

### ✅ **Prediction Pipeline**
- Accepts user input  
- Applies preprocessing  
- Predicts the customer’s cluster  
- Returns:
  - Cluster ID  
  - Confidence score (inverse-distance metric)

---

## 📊 Model Performance

| Metric | Value |
|--------|--------|
| Best K (clusters) | **4** |
| PCA Variance Explained | **58.5% (PC1 + PC2)** |
| Key Drivers | Spending, Income, Visits, Recency |

---

## 🌐 Streamlit Web Application

The UI allows users to:

- Input customer attributes manually  
- Receive:
  - Cluster assignment  
  - Confidence score  

---

### 🏆 Key Highlights

* Full end-to-end ML pipeline

*  Real-world customer segmentation use case

* Feature engineering for deeper insights

* PCA visualization for dimensionality reduction

* Production-grade structure

* Logging + exception handling

* Clean modular architecture

* Simple, user-friendly Streamlit deployment 

---
### 🚀 Future Improvements

Automated cluster naming

Hierarchical & DBSCAN clustering

Recommendation engine per cluster

Deployment on AWS/GCP

Cluster drift monitoring


---

### 👨‍💻 Author

Girish K S
Data Scientist