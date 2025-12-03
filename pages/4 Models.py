import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ------------------------------
# PAGE CONFIGURATION
# ------------------------------
st.set_page_config(
   page_title="Models",
   layout="wide"
)

# ------------------------------
# MAIN TITLE
# ------------------------------
st.title("Modeling: Predictive and Clustering Approaches")

# ------------------------------
# LOAD DATA
# ------------------------------
@st.cache_data
def load_data():
   disaster_df = pd.read_csv('datasets/disaster_df.csv')
   indicators_df_imputed = pd.read_csv('datasets/indicators_df_imputed.csv')
   demographic_df_imputed = pd.read_csv('datasets/demographic_df_imputed.csv')
   return disaster_df, indicators_df_imputed, demographic_df_imputed

disaster_df, indicators_df_imputed, demographic_df_imputed = load_data()

# ------------------------------
# Prepare Data for Modeling
# ------------------------------
disaster_df = disaster_df.rename(columns={
    'country': 'Country',
    'year': 'Year',
})

indicators_df_imputed = indicators_df_imputed.rename(columns={
    'country_name': 'Country',
    'year': 'Year'
})

demographic_df_imputed = demographic_df_imputed.rename(columns={
    'Name': 'Country'
})

# ------------------------------
# TAB SETUP
# ------------------------------
tab1, tab2 = st.tabs([
    "Disaster Event Severity Prediction",
    "Country Disaster Vulnerability Clustering",
])

# ------------------------------
# TAB 1
# ------------------------------
with tab1:
    st.subheader("Disaster Event Severity Prediction using Random Forest Regression")

    st.markdown("""
    This section provides an interactive Random Forest regression model designed to predict the severity index (1–10) of global disaster events. 
    
    The model leverages a combination of disaster event characteristics to estimate how severe an event is likely to be.

    This page allows you to:
    - Tune Random Forest hyperparameters using interactive controls.
    - Observe real-time updates to model performance.
    - Explore feature importance to understand which variables contribute the most to severity prediction.
    - Compare model behavior under different parameter configurations using interactive visualizations.

    By experiment with the Random Forest model, we can gain deeper insight into how disaster-related factors influence severity outcomes.
    """)

    # --- Prepare dataset for modeling ---
    # create copy of disaster dataset
    df = disaster_df.copy()

    # one-hot encode event_type
    df = pd.get_dummies(df, columns=['event_type'], prefix='event')

    # Target
    y = df['severity']

    # Features
    X = df.drop(columns=['severity', 'Country', 'date'])

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    st.divider()

    # ---------------------------------
    # Interactive Sidebar Controls
    # ---------------------------------

    with st.expander("Configure Random Forest Model Parameters", expanded=True):

        col1, col2, col3 = st.columns(3)

        # ---- n_estimators ----
        with col1:
            n_estimators = st.number_input(
                "Number of Trees",
                min_value=50, max_value=500, value=100, step=25
            )

        # ---- max_depth ----
        with col2:
            # Number input always visible
            max_depth_value = st.number_input(
                "Max Depth",
                min_value=10,
                max_value=100,
                value=10,
                step=10,
                disabled=False
            )

            # Checkbox BELOW number input
            use_unlimited_depth = st.checkbox("Unlimited Depth (None)", value=False)

            # Logic for actual parameter
            max_depth = None if use_unlimited_depth else max_depth_value

        # ---- max_features ----
        with col3:

            # Initialize keys
            for key in ("sqrt", "log2", "all_features", "fraction_input"):
                if key not in st.session_state:
                    st.session_state[key] = False if key != "fraction_input" else 0.3

            # Callback: user changes fraction → uncheck all boxes
            def _on_fraction_change():
                st.session_state["sqrt"] = False
                st.session_state["log2"] = False
                st.session_state["all_features"] = False

            # Callback: user selects a checkbox → clear the other two
            def _select_only(selected_key):
                for key in ("sqrt", "log2", "all_features"):
                    if key != selected_key:
                        st.session_state[key] = False

            # Fraction input (horizontal)
            fraction_value = st.number_input(
                "Max Features",
                min_value=0.1,
                max_value=1.0,
                value=st.session_state["fraction_input"],
                step=0.1,
                key="fraction_input",
                on_change=_on_fraction_change
            )

            # Checkboxes (mutually exclusive)
            st.checkbox(
                "sqrt",
                key="sqrt",
                on_change=lambda: _select_only("sqrt")
            )

            st.checkbox(
                "log2",
                key="log2",
                on_change=lambda: _select_only("log2")
            )

            st.checkbox(
                "All Features (None)",
                key="all_features",
                on_change=lambda: _select_only("all_features")
            )

            # Final logic
            if st.session_state["all_features"]:
                max_features = None
            elif st.session_state["sqrt"]:
                max_features = "sqrt"
            elif st.session_state["log2"]:
                max_features = "log2"
            else:
                max_features = fraction_value

    st.markdown("<br><br>", unsafe_allow_html=True)

        
    # ----------------------------------------
    # Train Model with Selected Parameters
    # ----------------------------------------
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        random_state=1,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Cross-validation
    cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring="neg_mean_squared_error")
    cv_mse_scores = -cv_scores
    cv_mse_mean = cv_mse_scores.mean()

    # --------------------------------
    # Display Metrics
    # --------------------------------
    st.markdown("##### Model Performance")

    colA, colB, colC = st.columns(3)

    with colA:
        st.metric("MSE", f"{mse:.3f}")

    with colB:
        st.metric("R² Score", f"{r2:.3f}")

    with colC:
        st.metric("Mean CV MSE", f"{cv_mse_mean:.3f}")
    
    st.markdown("<br><br>", unsafe_allow_html=True)

    # ----------------------------------
    # Interactive Feature Importance + CV Plot
    # ----------------------------------

    # Shared layout template to force matching chart shape
    shared_layout = dict(
        height=500,
        margin=dict(l=80, r=40, t=80, b=60),
        template="plotly_white",
        title=dict(
            x=0.0,
            xanchor="left",
            font=dict(size=20)
        ),
        xaxis=dict(
            title_font=dict(size=14),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title_font=dict(size=14),
            tickfont=dict(size=12)
        )
    )

    # ---- Feature Importance Plot ----
    feat_imp = (
            pd.DataFrame({
                "feature": X.columns,
                "importance": rf_model.feature_importances_
            })
            .sort_values("importance", ascending=True)
        )

    fig_feat = go.Figure(
        go.Bar(
            x=feat_imp["importance"],
            y=feat_imp["feature"],
            orientation="h",
            hovertemplate="%{x:.4f}<extra></extra>",
        )
    )
    fig_feat.update_layout(**shared_layout)
    fig_feat.update_layout(title=dict(text="Random Forest Feature Importance"))


    # ---- Cross-Fold MSE Plot ----
    fold_numbers = list(range(1, len(cv_mse_scores) + 1))
    mean_mse = np.mean(cv_mse_scores)

    fig_cv = go.Figure()
    fig_cv.add_trace(go.Scatter(
        x=fold_numbers,
        y=cv_mse_scores,
        mode="lines+markers",
        line=dict(width=2),
        marker=dict(size=7),
        name="Fold MSE"
    ))

    fig_cv.add_trace(go.Scatter(
        x=fold_numbers,
        y=[mean_mse] * len(fold_numbers),
        mode="lines",
        line=dict(width=2, dash="dash"),
        name=f"Mean MSE ({mean_mse:.4f})"
    ))

    fig_cv.update_layout(**shared_layout)
    fig_cv.update_layout(title=dict(text="Cross-Fold MSE Results"))

    # Display side by side
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig_feat, use_container_width=True)

    with col2:
        st.plotly_chart(fig_cv, use_container_width=True)

# ------------------------------
# TAB 2
# ------------------------------
with tab2:

    # ---------------------------------------------------
    # TITLE + INTRO
    # ---------------------------------------------------
    st.subheader("Country Disaster Vulnerability Clustering using KMeans")

    st.markdown("""
    This section groups countries into vulnerability clusters based on the country's disaster event impact, economic indicators, and demographic characteristics.
    
    The purpose of this clustering model is to identify patterns of vulnerability across countries, enabling:

    - Identification of countries with similar disaster risk profiles.
    - Exploration of how socioeconomic and demographic conditions relate to disaster vulnerability.
    - Visual comparison of clusters using PCA-based dimensionality reduction.

    Use the controls below to configure and explore the clustering model.
    """)

    st.divider()

    # ---------------------------------------------------
    # PREPROCESS DATA
    # ---------------------------------------------------
    combined_df = (
        disaster_df
        .merge(indicators_df_imputed, on=['Country', 'Year'], how='left')
        .merge(demographic_df_imputed, on=['Country', 'Year'], how='left')
    )

    numeric_cols = combined_df.select_dtypes(include='number').columns

    country_df = (
        combined_df
        .groupby('Country')[numeric_cols]
        .mean()
        .reset_index()
    )

    selector = VarianceThreshold()
    selector.fit(country_df.select_dtypes(include='number'))
    country_clean = country_df[country_df.select_dtypes(include='number').columns[selector.get_support()]]

    X_scaled = StandardScaler().fit_transform(country_clean.values)

    # ---------------------------------------------------
    # ELBOW METHOD SECTION
    # ---------------------------------------------------
    st.subheader("KMeans Clustering & Elbow Method")
    st.markdown("""
    We use KMeans clustering to group countries based on their average disaster, economic, and demographic indicators.

    KMeans partitions countries into k clusters such that each country belongs to the cluster with the nearest mean.  

    To determine the appropriate number of clusters (k), we use the Elbow Method, which involves plotting the Sum of Squared Errors (SSE) for different k values.  
    The “elbow” point indicates a good tradeoff between low SSE and minimal complexity.
    """)

    # Compute SSE for k
    sse = []
    K_range = range(2, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=1)
        km.fit(X_scaled)
        sse.append(km.inertia_)

    # Plot SSE
    fig_sse = go.Figure()
    fig_sse.add_trace(go.Scatter(
        x=list(K_range),
        y=sse,
        mode="lines+markers",
        marker=dict(size=8),
        line=dict(width=2),
        hovertemplate="k=%{x}<br>%{y:.2f}<extra></extra>"
    ))
    fig_sse.update_layout(
        title=dict(text="Elbow Method: SSE by Number of Clusters", x=0.0),
        xaxis_title="k (Clusters)",
        yaxis_title="SSE",
        template="plotly_white",
        height=450,
        margin=dict(l=60, r=40, t=80, b=60)
    )

    st.plotly_chart(fig_sse, use_container_width=True)
    st.markdown("<br><br>", unsafe_allow_html=True)

    # ---------------------------------------------------
    # PCA CLUSTERING SECTION
    # ---------------------------------------------------
    st.subheader("PCA Visualization of Clusters")
    st.markdown("""
    For visualization, we reduce the feature space to two dimensions using PCA (Principal Component Analysis).  
    
    PCA helps us plot high-dimensional data while preserving the most important variance.

    Use the slider below to select the number of clusters for KMeans, and observe how countries are grouped.                
    """)

    st.markdown("<br>", unsafe_allow_html=True)


    # Slider for number of clusters
    with st.expander("Configue KMeans Parameters", expanded=True):
        selected_k = st.slider(
            "Select Number of Clusters (k)",
            min_value=2, max_value=10, value=7, step=1
        )

    # Fit KMeans
    kmeans = KMeans(n_clusters=selected_k, random_state=1)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # PCA 2D coords
    coords = PCA(n_components=2).fit_transform(X_scaled)

    # Scatter plot of PCA clusters
    fig_pca = go.Figure()
    fig_pca.add_trace(go.Scatter(
        x=coords[:,0],
        y=coords[:,1],
        mode="markers",
        marker=dict(
            size=10,
            color=cluster_labels,
            colorscale="Viridis",
            showscale=True
        ),
        text=country_df['Country'],
        hovertemplate="<b>%{text}</b><br>Cluster %{marker.color}<extra></extra>"
    ))

    fig_pca.update_layout(
        title=dict(text="PCA Clustering of Country Disaster Vulnerability", x=0.0),
        xaxis_title="PC1",
        yaxis_title="PC2",
        template="plotly_white",
        height=500,
        margin=dict(l=60, r=40, t=80, b=60)
    )

    st.plotly_chart(fig_pca, use_container_width=True)