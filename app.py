import streamlit as st
import pandas as pd
import joblib
import json
import plotly.express as px

@st.cache_data
def load_raw():
    return pd.read_csv("data/raw_data.csv")

@st.cache_data
def load_clean():
    return pd.read_csv("data/clean_data.csv")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl"), joblib.load("scaler.pkl"), joblib.load("label_encoder.pkl")

@st.cache_data
def load_metrics():
    return json.load(open("model_metrics.json"))

raw_df = load_raw()
clean_df = load_clean()
model, scaler, le = load_model()
metrics = load_metrics()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Data", "EDA", "Model", "Predict"])

if page == "Overview":
    st.title("Brewery Type Classifier")
    st.write("Predicts brewery type from location data. Source: openbrewerydb.org")
    st.metric("Total rows", len(clean_df))
    st.metric("Classes", len(le.classes_))
    st.metric("Best model", metrics["best_model"])

if page == "Data":
    st.title("Data Overview")
    st.write("Raw shape:", raw_df.shape)
    st.write(raw_df.isna().sum())
    st.dataframe(raw_df.head())
    st.write("Clean shape:", clean_df.shape)
    st.write(clean_df.isna().sum())
    st.dataframe(clean_df.head())

if page == "EDA":
    st.title("EDA")
    types = clean_df["brewery_type"].unique()
    picked = st.sidebar.multiselect("Filter types", types, default=types)
    filtered = clean_df[clean_df["brewery_type"].isin(picked)]
    counts = filtered["brewery_type"].value_counts().reset_index()
    counts.columns = ["brewery_type", "count"]
    st.plotly_chart(px.bar(counts, x="brewery_type", y="count"))
    corr = filtered[["longitude", "latitude"]].corr()
    st.plotly_chart(px.imshow(corr))

if page == "Model":
    st.title("Model Performance")
    st.dataframe(pd.DataFrame(metrics["all_model_results"]))
    st.plotly_chart(px.imshow(metrics["confusion_matrix"], x=metrics["class_labels"], y=metrics["class_labels"]))
    imp = pd.DataFrame(metrics["feature_importance"].items(), columns=["feature", "importance"])
    st.plotly_chart(px.bar(imp, x="feature", y="importance"))

if page == "Predict":
    st.title("Predict")
    longitude = st.number_input("Longitude", value=0.0)
    latitude = st.number_input("Latitude", value=0.0)
    countries = [c.replace("country_", "") for c in metrics["features"] if c.startswith("country_")]
    country = st.selectbox("Country", countries)

if st.button("Predict"):
        row = dict.fromkeys(metrics["features"], 0)
        row["longitude"] = longitude
        row["latitude"] = latitude
        row["country_" + country] = 1
        input_df = pd.DataFrame([row])[metrics["features"]]
        scaled = scaler.transform(input_df)
        pred = le.inverse_transform(model.predict(scaled))[0]
        st.success("Predicted: " + pred)
        probs = model.predict_proba(scaled)[0]
        st.plotly_chart(px.bar(x=le.classes_, y=probs))

st.sidebar.write("Source: Open Brewery DB API")
st.sidebar.write("Last updated: July 2026")