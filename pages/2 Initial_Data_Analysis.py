import streamlit as st

# ------------------------------
# PAGE CONFIGURATION
# ------------------------------
st.set_page_config(
   page_title="Initial Data Analysis",
   layout="wide"
)

# ------------------------------
# MAIN TITLE
# ------------------------------
st.title("Initial Data Analysis")

# ------------------------------
# TAB SETUP
# ------------------------------
tab1, tab2, tab3 = st.tabs([
   "Cleaning and Preprocessing",
   "Missingness Analysis",
   "Imputation Strategy and Evaluation"
])

# ------------------------------
# TAB 1 – CLEANING AND PREPROCESSING
# ------------------------------
with tab1:
   
   st.subheader("Data Cleaning and Preprocessing")
   st.markdown("""
   Each dataset underwent a series of cleaning, standardization, and preprocessing steps to ensure consistency and interoperability.
   The following sections summarize the key procedures applied to the disaster, economic, and demographic datasets.
   """)

   st.divider()

   # ------------------------------------------------------------
   # Natural Disasters Dataset
   # ------------------------------------------------------------
   st.subheader("Natural Disasters Dataset")
   st.markdown("""
   Cleaning and preparation focused on ensuring temporal consistency, standardizing categorical fields, and validating numerical values.
                
   **Cleaning and Standardization Steps**
   - Removed irrelevant or unused columns, including metadata identifiers.
   - Standardized date fields by converting all date strings into datetime objects.
   - Cleaned categorical variables such as event_type and country to ensure consistent naming.
   - Converted all numeric fields to numeric types.

   **Encoding**
   - Applied one-hot encoding to `event_type` to convert the categorical disaster types into numerical features.  
   - This encoding allows machine learning models to interpret each disaster type as an independent binary indicator without imposing an artificial ranking or ordinal relationship between categories.  
   - The resulting binary columns preserve interpretability and ensure that models do not assume any unintended ordering among disaster types.
   """)

   st.divider()

   # ------------------------------------------------------------
   # Economic Indicators Dataset
   # ------------------------------------------------------------
   st.subheader("Economic Indicators Dataset")
   st.markdown("""
   The economic indicators dataset was aligned with the structure and country definitions in the disaster dataset.

   **Cleaning and Standardization Steps**
   - Removed redundant identifiers and unused metadata columns.
   - Standardized monetary values such as GDP and gross national income into millions of USD.
   - Renamed variables to ensure consistent naming conventions and readable formatting.
   - Corrected numeric, percentage, and categorical field types to ensure reliable downstream analysis.
   """)

   st.divider()

   # ------------------------------------------------------------
   # Demographic Dataset
   # ------------------------------------------------------------
   st.subheader("Demographic Dataset")
   st.markdown("""
   Preprocessing for the demographic dataset ensured alignment with country names in the other datasets and standardized demographic metrics.

   **Cleaning and Standardization Steps**
   - Dropped unnecessary columns which did not provide meaningful analytic value.
   - Removed year separator rows to ensure a clean tabular structure.
   - Replaced missing value placeholders (e.g. "--") with NaNs for proper numeric processing.
   - Removed comma formatting in numeric fields to enable conversion to numeric types.
   - Standardized country names to match the disaster and economic datasets.
   """)

   st.divider()

   # ------------------------------------------------------------
   # Dataset Alignment and Harmonization
   # ------------------------------------------------------------
   st.subheader("Dataset Alignment and Harmonization")
   st.markdown("""
   To prepare the three datasets for integration, a structured alignment process was conducted to ensure all shared fields matched cleanly across sources. 
   This step was essential for merging and building a unified modeling dataset.

   **Alignment and Standardization Steps**
   1. **Country Filtering Based on Disaster Dataset**  
      - The disaster dataset served as the primary reference for included countries.  
      - Both the economic indicators and demographic datasets were filtered to retain only countries present in the disaster dataset, ensuring full cross-dataset compatibility.

   2. **Country Name Mapping and Normalization**  
      - Constructed a country-name mapping dictionary to resolve inconsistencies.
      - Applied the mapping across all datasets to enforce a single canonical naming scheme.

   3. **Year and Country Standardization**
      - Renamed the corresponding fields in all datasets to uniform column names (`Country`, `Year`)
      - This standardization ensured clean joins and removed any schema mismatches.

   4. **Temporal Alignment**
      - Filtered all datasets to include only years overlapping with the disaster dataset (2020–2025).
      - This prevented incomplete or non-overlapping records during integration.

   5. **Final Dataset Merge**
      - Performed a consistent merge across the three datasets using `Country` and `Year` as keys.
      - The output was a fully aligned, analysis-ready dataset containing disaster events, economic indicators, and demographic characteristics for each country-year pair.
   """)


# -------------------------------------
# TAB 2 - MISSINGNESS ANALYSIS
# -------------------------------------
with tab2:

   st.subheader("Missingness Analysis")
   st.markdown("""
   Missing data patterns play a critical role in understanding dataset reliability and selecting appropriate imputation strategies.
   This section evaluates missingness in the indicators dataset and highlights key structural patterns.
   """)

   st.divider()

   # ------------------------------------------------------------
   # Economic Indicators Missingness Overview
   # ------------------------------------------------------------
   st.subheader("Economic Indicators Dataset Missingness")

   # --- Image ---
   st.image(
    "images/indicators_missingness.png",
    caption="Missingness heatmap (left) and correlation between missingness indicators and observed values (right).",
    use_container_width=True
)

   # ------------------------------------------------------------
   # Interpretation
   # ------------------------------------------------------------
   st.markdown("""
   Several important patterns emerged from the Economic Indicators missingness visualizations:

   **1. Temporal and Variable-Level Patterns from Heatmap**  
   - Horizontal bands reveal groups of countries missing data across multiple indicators, particularly in later years such as 2024–2025 where reporting is incomplete.  
   - Vertical bands highlight indicators frequently missing across many countries, suggesting systematic gaps in data collection.

   **2. Correlation Between Missingness Indicators and Observed Values**  
   - Binary missingness indicators were correlated with the observed values of several variables.  
   - These moderate correlations indicate informative missingness, where the probability of missingness depends on economic conditions captured in the available data.

   **3. Missingness Mechanism**  
   - The observed structure supports a Missing At Random (MAR) mechanism assumption. Missingness appears to depend on observed characteristics, not on the missing values themselves.  
   - This supports the use of statistical imputation methods such as regression-based imputation or KNN without introducing substantial bias.

   **Key Takeaway**  
   The MAR-consistent pattern validates the use of multivariate imputation methods that leverage observed data relationships.
   """)

   st.divider()

   # ------------------------------------------------------------
   # Demographic Missingness Overview
   # ------------------------------------------------------------
   st.subheader("Demographics Dataset Missingness")

   # --- Image ---
   st.image(
    "images/demographics_missingness.png",
    caption="Missingness heatmap (left) and correlation between missingness indicators and observed values (right).",
    use_container_width=True
)

   # ------------------------------------------------------------
   # Interpretation
   # ------------------------------------------------------------
   st.markdown("""
   **1. Insights From the Missingness Heatmap**  
   - Short horizontal segments indicate that only a few specific country-year entries are missing for certain demographic indicators.
   - There are no long horizontal bands, suggesting that the same country is not consistently missing multiple demographic values across time.
   - No prominent vertical bands, suggesting that missingness is not concentrated in particular variables.

   **2. Correlation Between Missingness Indicators and Observed Values**  
   - Only few missingness indicators show any notable correlation with observed demographic variables, suggesting local irregularities rather than systematic patterns.
   - Most cells are near zero, indicating no meaningful relationship between a value being missing and other demographic measurements.

   **3. Missingness Mechanism**  
   - Missing values are rare, isolated, and not systematically tied to other observed variables.
   - The lack of strong correlation and absence of structural patterns supports a Missing At Random (MAR) mechanism.

   **Key Takeaway**  
   MAR implies that missingness is related to other observed variables, but not to the unobserved value itself, 
   a reasonable assumption here given the sparse, nonsystematic structure. Thus,
   standard imputation methods can be applied without significant risk of bias.
   """)

# -------------------------------------
# TAB 3 - IMPUTATION STRATEGY AND EVALUATION
# -------------------------------------
with tab3:
   st.subheader("Imputation Strategy")

   st.markdown("""
   Guided by the MAR (Missing At Random) assumptions identified in the missingness analysis, 
   we applied a unified two-stage imputation framework to both the economic indicators and demographic datasets.
   This approach ensures that temporal patterns within countries and similarity patterns across countries are both leveraged for reliable reconstruction of missing values.
   """)

   st.markdown("""
   **Step 1. Stochastic Regression Imputation**
   - For each variable (economic or demographic), we modeled its trajectory over time within each country using linear regression.
   - Missing values were replaced with regression-based predictions, with added stochastic noise to preserve natural variability.
   - Country–variable pairs with fewer than two valid observations were skipped in this step and imputed in Step 2.
   - This step captures temporal trends such as economic growth patterns or demographic shifts.
   """)

   st.markdown("""
   **Step 2. K-Nearest Neighbors (KNN) Imputation**
   - Remaining missing values were imputed using a KNN-based approach, leveraging similarity across countries.
   - For the indicators dataset, similarity was driven by economic profiles within each year.
   - For the demographic dataset, similarity was based on population structure and demographic characteristics.
   - Combining regression (temporal structure) with KNN (cross-sectional similarity) creates a robust hybrid strategy that accommodates both datasets’ patterns and ensures stable imputation.
   """)

   st.divider()

   # -------------------------------------
   # Post-Imputation Evaluation
   # -------------------------------------
   st.subheader("Post-Imputation Evaluation")

   st.markdown("""
   To assess the quality of our imputation strategy, we examined how well the multivariate correlation structure was preserved after imputation.  
   """)

   # --- Pre/Post Correlation Images (Indicators) ---
   st.image("images/indicators_correlation_structure_comparison.png",
            caption="Correlation structure before (left) and after (right) imputation for the Indicators dataset.",
            use_container_width=True)

   # --- Pre/Post Correlation Images (Demographics) ---
   st.image("images/demographics_correlation_structure_comparison.png",
            caption="Correlation structure before (left) and after (right) imputation for the Demographics dataset.",
            use_container_width=True)

   st.markdown("""
   Across both datasets, the pre and post imputation correlation matrices show highly consistent patterns:

   - Major correlation blocks remain aligned.  
   - Positive/negative relationships between variables were preserved.  
   - Only small local shifts occur, typically around variables with initially heavy missingness.  
   - No evidence of artificially strong or spurious correlations emerged.

   These results indicate that the imputation strategy did not distort the underlying statistical structure of the data. Therefore, 
   we can conclude that the hybrid imputation approach which combines stochastic temporal regression with KNN-based cross-sectional similarity,
   produced completed datasets that retain their original correlation structure.
   """)
