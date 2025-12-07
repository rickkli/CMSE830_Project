import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score, davies_bouldin_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px


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

    # --------------------------------
    # Random Forest Regressin Model
    # --------------------------------

    # ---------------------------------
    # Interactive Sidebar Controls
    # ---------------------------------
    with st.expander("Configure Random Forest Model Parameters", expanded=True):

        col1, col2, col3 = st.columns(3)

        # ---- n_estimators ----
        with col1:
            n_estimators = st.number_input(
                "Number of Trees",
                min_value=50, max_value=500, value=50, step=25
            )

        # --- max depth ---
        with col2:
            # --- Initialize session state ---
            if "max_depth_value" not in st.session_state:
                st.session_state.max_depth_value = 10

            if "use_unlimited_depth" not in st.session_state:
                st.session_state.use_unlimited_depth = False


            # --- Callbacks ---
            def on_change_max_depth_value():
                # Anytime number is changed, disable unlimited mode
                if st.session_state.use_unlimited_depth:
                    st.session_state.use_unlimited_depth = False


            def on_change_unlimited_depth():
                # No special logic needed here; number input stays enabled
                pass

            # Number Input (always enabled)
            max_depth_value = st.number_input(
                "Max Depth",
                min_value=10,
                max_value=100,
                step=10,
                key="max_depth_value",
                on_change=on_change_max_depth_value
            )

            # Unlimited checkbox
            use_unlimited_depth = st.checkbox(
                "Unlimited Depth (None)",
                key="use_unlimited_depth",
                on_change=on_change_unlimited_depth
            )


            # --- Final parameter ---
            max_depth = None if st.session_state.use_unlimited_depth else st.session_state.max_depth_value

        # ---- max_features ----
        with col3:

            # Initialize keys
            for key in ("sqrt", "log2", "all_features", "fraction_input"):
                if key not in st.session_state:
                    st.session_state[key] = False if key != "fraction_input" else 0.1

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
                #value=st.session_state["fraction_input"],
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
    st.subheader("Model Performance")

    colA, colB, colC = st.columns(3)

    # Shared style for all three metric boxes
    metric_card = """
        border:2px solid #636efa;
        border-radius:10px;
        padding:20px;
        text-align:center;
        width:100%;
    """

    title_style = "margin:0; color:#636efa;"
    value_style = "font-size:24px; margin:5px 0;"

    with colA:
        st.markdown(
            f"""
            <div style="{metric_card}">
                <h3 style="{title_style}">Test MSE</h3>
                <p style="{value_style}">{mse:.3f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with colB:
        st.markdown(
            f"""
            <div style="{metric_card}">
                <h3 style="{title_style}">Test R²</h3>
                <p style="{value_style}">{r2:.3f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with colC:
        st.markdown(
            f"""
            <div style="{metric_card}">
                <h3 style="{title_style}">Mean CV MSE</h3>
                <p style="{value_style}">{cv_mse_mean:.3f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
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
            hovertemplate="%{x:.3f}<extra></extra>",
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
        hovertemplate="%{y:.3f}<extra></extra>",
        name="Fold MSE"
    ))

    fig_cv.add_trace(go.Scatter(
        x=fold_numbers,
        y=[mean_mse] * len(fold_numbers),
        mode="lines",
        line=dict(width=2, dash="dash"),
        name=f"Mean MSE ({mean_mse:.3f})"
    ))

    fig_cv.update_layout(**shared_layout)
    fig_cv.update_layout(title=dict(text="Cross-Fold MSE Results"))

    # Display side by side
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig_feat, use_container_width=True)

    with col2:
        st.plotly_chart(fig_cv, use_container_width=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ----------------------------------
    # Hyperparameter Tuning
    # ----------------------------------
    with st.expander("Hyperparameter Tuning Analysis", expanded=True):
        st.subheader("Hyperparameter Tuning Analysis")
        st.markdown("""
        To further improve the performance of the baseline Random Forest model, we performed hyperparameter tuning using a Randomized Search approach. 
        
        Random Forests contain several important hyperparameters (tree depth, number of estimators, and feature sampling strategy) that significantly influence predictive accuracy and model stability.

        **Tuning Approach**
                    
        We used `RandomizedSearchCV` with:
        - 20 sampled hyperparameter combinations.
        - 5-fold cross-validation.
        - MSE minimization (`neg_mean_squared_error`) as the scoring metric.
        
        The search grid covered:
        - Number of estimators (`200–500`).
        - Maximum tree depth (`None` or between `10–40`).
        - Feature sampling strategy (`sqrt`, `log2`, or `None`).

        **Best-Fit Model**
                    
        The tuning procedure returned a set of optimized hyperparameters that were used to refit the model on the training set.
                    
        The optimized Random Forest outperformed the baseline (`n_estimators=100`, `max_depth=None` , `max_features='sqrt'`) across both:
        - Hold-out test metrics (MSE, R²). 
        - 5-fold cross-validation MSE.

        This provides strong evidence that tuning improved both accuracy and generalization stability.
        Together, these findings confirm that hyperparameter tuning meaningfully improved the severity prediction task.
        """)

        st.markdown("<br><br>", unsafe_allow_html=True)

        # ---------------------------------------------------
        # Baseline Model Performance
        # ---------------------------------------------------
        rf_base = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        max_features='sqrt',
        random_state=1,
        n_jobs=-1
        )

        rf_base.fit(X_train, y_train)

        y_pred_base = rf_base.predict(X_test)
        mse_base = mean_squared_error(y_test, y_pred_base)
        r2_base = r2_score(y_test, y_pred_base)

        # --------------------------------------------
        # Optimized Random Forest Performance
        # --------------------------------------------
        best_params = {
        "n_estimators": 500,
        "max_features": "sqrt",
        "max_depth": 30
        }

        best_rf = RandomForestRegressor(
            n_estimators=best_params["n_estimators"],
            max_features=best_params["max_features"],
            max_depth=best_params["max_depth"],
            random_state=1,
            n_jobs=-1
        )

        best_rf.fit(X_train, y_train)
        y_pred_opt = best_rf.predict(X_test)

        mse_opt = mean_squared_error(y_test, y_pred_opt)
        r2_opt = r2_score(y_test, y_pred_opt)

        # ---------------------------------------------------
        # User-Selected Model Performance
        # ---------------------------------------------------
        y_pred_user = rf_model.predict(X_test)
        mse_user = mean_squared_error(y_test, y_pred_user)
        r2_user = r2_score(y_test, y_pred_user)

        # ---------------------------------------------------
        # 5-Fold CV for All Models
        # ---------------------------------------------------
        kf = KFold(n_splits=5, shuffle=True, random_state=1)

        cv_mse_base = -cross_val_score(rf_base, X_train, y_train,
                                    scoring="neg_mean_squared_error", cv=kf, n_jobs=-1)

        cv_mse_user = -cross_val_score(rf_model, X_train, y_train,
                                    scoring="neg_mean_squared_error", cv=kf, n_jobs=-1)

        cv_mse_opt = -cross_val_score(best_rf, X_train, y_train,
                                    scoring="neg_mean_squared_error", cv=kf, n_jobs=-1)

        # -------------------------------
        # Optimal Parameter Display Box
        # -------------------------------
        st.subheader("Optimal Random Forest Parameters")
        st.markdown("""
        The hyperparameter tuning process identified the following optimal parameters for the Random Forest model.
        """)

        st.info(
            f"""
            **Optimal Parameters Identified:**
            - `n_estimators`: **{best_params['n_estimators']}**
            - `max_depth`: **{best_params['max_depth']}**
            - `max_features`: **{best_params['max_features']}**
            """
        )
        
        st.markdown("<br><br>", unsafe_allow_html=True)

        # -------------------------------
        # Model Performance Breakdown
        # -------------------------------
        def metric_box(title, value, color):
            return f"""
            <div style="
                border:2px solid {color};
                border-radius:10px;
                padding:18px;
                text-align:center;
                width:100%;
                margin-bottom:12px;
            ">
                <h3 style="margin:0; color:{color}; font-size:20px;">{title}</h3>
                <p style="font-size:24px; margin:6px 0;">{value}</p>
            </div>
            """
        
        st.subheader("Model Performance Comparison")
        st.markdown("""
        Additionally, we can compare the performances of the baseline model, the user-configured model, and the optimized model.
                    
        By evaluating each model using consistent cross-validation metrics, we can assess how parameter tuning influences predictive accuracy and determine whether the optimized configuration offers a meaningful improvement. 
        
        The visualized results highlight differences in error reduction, model stability across folds, and overall predictive performance.
        """)

        # Metric options
        metric_options = {
            "Test MSE": (
                f"{mse_base:.3f}",
                f"{mse_user:.3f}",
                f"{mse_opt:.3f}"
            ),
            "Test R²": (
                f"{r2_base:.3f}",
                f"{r2_user:.3f}",
                f"{r2_opt:.3f}"
            ),
            "CV Mean MSE": (
                f"{cv_mse_base.mean():.3f}",
                f"{cv_mse_user.mean():.3f}",
                f"{cv_mse_opt.mean():.3f}"
            ),
            "CV Std MSE": (
                f"{cv_mse_base.std():.3f}",
                f"{cv_mse_user.std():.3f}",
                f"{cv_mse_opt.std():.3f}"
            ),
        }

        # Select which metric to view
        selected_metric = st.selectbox("Select a Metric to Compare", list(metric_options.keys()))

        st.markdown("<br>", unsafe_allow_html=True)

        # Assign colors
        red = "#ef553b"
        blue = "#636efa"
        green = "#00cc96"

        # Retrieve metric values for Baseline, User, Optimized
        base_val, user_val, opt_val = metric_options[selected_metric]

        # 3 columns for display
        colA, colB, colC = st.columns(3)

        with colA:
            st.markdown(metric_box(f"Baseline", base_val, red), unsafe_allow_html=True)

        with colB:
            st.markdown(metric_box(f"User Model", user_val, blue), unsafe_allow_html=True)

        with colC:
            st.markdown(metric_box(f"Optimized", opt_val, green), unsafe_allow_html=True)

# ------------------------------
# TAB 2
# ------------------------------
with tab2:

    # ---------------------------------------------------
    # TITLE + INTRO
    # ---------------------------------------------------
    st.subheader("Country Disaster Vulnerability Clustering using K-Means")

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
    # PCA CLUSTERING
    # ---------------------------------------------------
    st.subheader("PCA Visualization of Clusters")
    st.markdown("""
    This section explores the underlying structure of country-level disaster vulnerability by applying K-Means clustering to group countries based on socioeconomic, demographic, and disaster-related indicators.

    To visualize the clusters, we first reduce the high-dimensional feature space into two principal components using Principal Component Analysis (PCA).            

    This approach enables us to visualize patterns and cluster separation in 2D while retaining the most important information.

    **PCA Scatter Plot**  
    - Shows each country projected onto the top two principal components, colored by cluster membership.  
    - Helps reveal how distinct or overlapping the clusters are.
    
    **Cluster Feature Comparison**
    - Allows you to select a feature and visualize how each cluster differs.
    - This supports interpretation of what truly distinguishes each cluster.

    This enables you to explore how different choices of `k` affect cluster structure and interpretability.
                
    Use the slider below to select the number of clusters (`k`) for K-Means.  
    """)

    st.markdown("<br>", unsafe_allow_html=True)


    # Slider for number of clusters
    with st.expander("Configue K-Means Parameters", expanded=True):
        selected_k = st.slider(
            "Select Number of Clusters (k)",
            min_value=2, max_value=10, value=2, step=1
        )

    # Fit KMeans
    kmeans = KMeans(n_clusters=selected_k, random_state=1)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # PCA 2D coords
    coords = PCA(n_components=2).fit_transform(X_scaled)

    # ---------------------------------------------------
    # TWO EQUAL HEIGHT COLUMNS
    # ---------------------------------------------------
    col_left, col_right = st.columns(2, vertical_alignment="top")

    # Define fixed heights for both column containers
    PLOT_HEIGHT = 550
    RIGHT_TOTAL_HEIGHT = 550  # dropdown + plot should match this


    # ---------------------------------------------------
    # LEFT COLUMN → PCA PLOT (MATCHED HEIGHT)
    # ---------------------------------------------------
    with col_left:
        left_container = st.container()
        with left_container:

            fig_pca = go.Figure()
            fig_pca.add_trace(go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers",
                marker=dict(
                    size=10,
                    color=cluster_labels,
                    colorscale="Viridis",
                    showscale=True,
                ),
                text=country_df["Country"],
                hovertemplate="<b>%{text}</b><br>Cluster %{marker.color}<extra></extra>",
            ))

            fig_pca.update_layout(
                title=dict(text="PCA Clustering of Country Disaster Vulnerability", x=0.0),
                xaxis_title="PC1",
                yaxis_title="PC2",
                template="plotly_white",
                height=PLOT_HEIGHT,
                margin=dict(l=60, r=40, t=80, b=60)
            )

            st.plotly_chart(fig_pca, use_container_width=True)

    # ---------------------------------------------------
    # RIGHT COLUMN → FEATURE BAR CHART
    # ---------------------------------------------------
    with col_right:
        right_container = st.container()
        with right_container:
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("**Cluster Feature Comparison**")

            # Dropdown
            feature_cols = country_clean.columns[2:]
            selected_feature = st.selectbox(
                "Select a Feature",
                options=feature_cols,
            )

            # Generate consistent Viridis colors for each cluster
            viridis = px.colors.sequential.Viridis
            num_clusters = selected_k

            # Sample evenly from Viridis
            cluster_colors = [viridis[int(i * (len(viridis) - 1) / (num_clusters - 1))] for i in range(num_clusters)]

            # Compute cluster means
            country_df["Cluster"] = cluster_labels
            cluster_means = (
                country_df.groupby("Cluster")[selected_feature]
                .mean()
                .reset_index()
                .rename(columns={selected_feature: "Value"})
            )

            # Bar chart
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=cluster_means["Cluster"].astype(str),
                y=cluster_means["Value"],
                marker=dict(
                    color=[cluster_colors[c] for c in cluster_means["Cluster"]],
                ),
                hovertemplate="%{y:.2f}<extra></extra>"
            ))

            fig_bar.update_layout(
                xaxis_title="Cluster",
                yaxis_title=selected_feature,
                template="plotly_white",
                height=PLOT_HEIGHT - 150,  # adjust to compensate for dropdown height
                margin=dict(l=20, r=20, t=40, b=40),
            )

            st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # ---------------------------------------------------
    # CLUSTERING EVALUTATION
    # ---------------------------------------------------
    with st.expander("K-Means Clustering Evaluation", expanded=True):
        st.subheader("K-Means Clustering Evaluation")
        st.markdown("""
        The goal of this section is to identify an appropriate number of clusters (`k`) and assess whether those clusters are well-formed, well-separated, and evenly distributed.
                    
        To determine how well the countries group into meaningful clusters, we evaluate the clustering performance using a combination of diagnostic tools.
                    
        **Elbow Method**
        - Helps identify a reasonable value of `k` by examining how the Sum of Squared Errors (SSE) decreases as more clusters are added.
        - The “elbow point” indicates where adding more clusters no longer yields substantial improvement.
                    
        **Cluster Size Distribution**
        - Visualizes how many countries fall into each cluster for the selected `k`.
        - Helps identify if any clusters are disproportionately large or small.
                    
        **Silhouette Score**
        - Measures how similar countries are to their own cluster compared to other clusters.
        - Values range from -1 to 1, with higher values indicating better-defined clusters.
                    
        **Davies–Bouldin Index (DBI)**
        - Evaluates averge similarity between clusters.
        - Lower values indicate more distinct and well-separated clusters.
                    
        By combining these diagnostics, gain a comprehensive understanding of the apporipate number of clusters and their quality.
                    
        These insights help ensure the clustering results reflect real underlying patterns in countries’ disaster vulnerability characteristics.
        """)

        # ------------------------------------
        # Compute SSE for Elbow Method
        # ------------------------------------
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
            title=dict(text="Elbow Method", x=0.0),
            xaxis_title="k (Clusters)",
            yaxis_title="SSE",
            template="plotly_white",
            height=450,
            margin=dict(l=60, r=40, t=80, b=60)
        )

        st.plotly_chart(fig_sse, use_container_width=True)

        st.info("""
        **Optimal Number of Clusters Identified:**
                
        - k = 7
        - the Elbow Method shows that the decrese in SSE slows significantly after k = 7, indicating diminishing improvements in model fit with more clusters
        """)

        st.markdown("<br><br>", unsafe_allow_html=True)

        # ------------------------------------
        # Fit KMeans for k = 7
        # ------------------------------------
        k_opt = 7
        kmeans = KMeans(n_clusters=k_opt, random_state=1)
        labels = kmeans.fit_predict(X_scaled)

        # Evaluation metrics
        sil_score = silhouette_score(X_scaled, labels)
        dbi_score = davies_bouldin_score(X_scaled, labels)

        # ------------------------------------
        # Two Columns Below Elbow
        # ------------------------------------
        col1, col2 = st.columns([2, 1])

        # ------------------------------------
        # LEFT: Donut Chart — Cluster Size Distribution
        # ------------------------------------
        with col1:
            st.markdown("**Cluster Size Distribution (k = 7)**")

            cluster_counts = pd.Series(labels).value_counts().sort_index()
            
            # Generate exactly 7 Viridis colors for the clusters
            viridis = px.colors.sequential.Viridis
            # cluster_colors = [viridis[int(i * (len(viridis) - 1) / (k_opt - 1))] for i in range(k_opt)]

            cluster_counts = pd.Series(labels).value_counts().sort_index()

            viridis = px.colors.sequential.Viridis
            cluster_colors = [viridis[int(i * (len(viridis) - 1) / (k_opt - 1))] for i in range(k_opt)]

            fig_donut = go.Figure(data=[go.Pie(
                labels=[f"Cluster {i}" for i in cluster_counts.index],
                values=cluster_counts.values,
                hole=0.55,
                marker=dict(colors=cluster_colors),
                hovertemplate="%{label}<br>Count: %{value}<extra></extra>",
                textinfo='label+percent',
                textposition='inside',
                automargin=True,
                sort=False
            )])

            fig_donut.update_layout(
                showlegend=True,
                template="plotly_white",
                height=500,
                margin=dict(l=20, r=20, t=30, b=30),
                legend=dict(
                    x=0.8,             # move legend horizontally closer to donut
                    y=0.5,             # vertically centered
                    xanchor='center',
                    yanchor='middle',
                    orientation='v',   # vertical stack
                    font=dict(size=14), # slightly bigger font
                    traceorder='normal'
                )
            )

            st.plotly_chart(fig_donut, use_container_width=True)

            st.info("""
            - distribution of cluster sizes shows that clusters are reasonably proportional, with no extreme outliers in size
            - indicates that the chosen clustering solution does not suffer from overly small or dominant clusters
            """)

        # ------------------------------------
        # RIGHT: Silhouette & DBI
        # ------------------------------------
        with col2:
            st.markdown("**Clustering Quality Metrics**")

            # Create HTML styled boxes for metrics
            metric_html = f"""
            <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:500px; gap:20px;">
                <div style="border:2px solid #636efa; border-radius:10px; padding:20px; text-align:center; width:90%;">
                    <h3 style="margin:0; color:#636efa;">Silhouette Score</h3>
                    <p style="font-size:24px; margin:5px 0;">{sil_score:.4f}</p>
                </div>
                <div style="border:2px solid #636efa; border-radius:10px; padding:20px; text-align:center; width:90%;">
                    <h3 style="margin:0; color:#636efa;">Davies–Bouldin Index (DBI)</h3>
                    <p style="font-size:24px; margin:5px 0;">{dbi_score:.4f}</p>
                </div>
            </div>
            """

            st.markdown(metric_html, unsafe_allow_html=True)

            st.markdown(" \
            ")

            st.info("""
            - low positive silhouette score value indicates that clusters are weakly separated but still form distinguishable groupings
            - low DBI value indicates moderate cluster compactness, separation, and overall cluster structure quality
            
            - taken together, these metrics support the choice of k = 7 as a balanced solution, as clusters are distinguishable, not excessively overlapping, and reasonably compact
            """)