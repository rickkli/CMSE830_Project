import streamlit as st
import pandas as pd
import plotly.graph_objects as go


# load cleaned datasets
@st.cache_data
def load_data():
    disaster_df = pd.read_csv('datasets/disaster_df.csv')
    indicators_df_imputed = pd.read_csv('datasets/indicators_df_imputed.csv')
    return disaster_df, indicators_df_imputed

disaster_df, indicators_df_imputed = load_data()


st.title("Global Disaster and Economic Impact Explorer (2020 - 2025)")

# ----- testing ------
st.header("Dataset Overview")
st.write("This page shows a general overview of the data.")

st.write("### Climate Events Dataset")
st.dataframe(disaster_df.head())

st.write("### Economic Indicators Dataset")
st.dataframe(indicators_df_imputed.head())