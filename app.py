import streamlit as st
import pandas as pd
import numpy as np
import os

from src.utils import load_object
from src.exception import CustomException

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("Customer Segmentation  KMeans Model")
st.write("Upload customer data and the model will assign a cluster to each record.")

 
@st.cache_resource
def load_artifacts():
    try:
        preprocessor = load_object("artifacts/preprocessor.pkl")
        kmeans = load_object("artifacts/kmeans.pkl")
        return preprocessor, kmeans
    except Exception as e:
        raise CustomException(e, None)

try:
    preprocessor, kmeans = load_artifacts()
except:
    st.error(" Could not load model or preprocessor.")
    st.stop()

 
def predict_clusters(df):
    X = preprocessor.transform(df)
    if hasattr(X, "toarray"):
        X = X.toarray()

    labels = kmeans.predict(X)

    distances = kmeans.transform(X)
    confidence = 1 / (1 + distances.min(axis=1))

    output = df.copy()
    output["Cluster"] = labels
    output["Confidence"] = np.round(confidence, 4)
    return output

 
st.subheader(" Upload CSV File")

file = st.file_uploader("Upload customer data (.csv)", type=["csv"])

if file:
    try:
        df = pd.read_csv(file)
        st.write("### Preview of uploaded data")
        st.dataframe(df.head(), use_container_width=True)

        if st.button(" Predict Clusters"):
            results = predict_clusters(df)

            st.success("Prediction Complete!")
            st.write("### Results")
            st.dataframe(results.head(50), use_container_width=True)

             
            st.write("### Cluster Distribution")
            st.bar_chart(results["Cluster"].value_counts().sort_index())

    except Exception as e:
        st.error("Error processing the file. Please ensure your CSV matches training schema.")
        st.exception(e)
 
