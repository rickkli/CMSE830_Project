## CMSE830 Project: Global Disaster and Economic Impact Dashboard (2020 - 2025)

### Overview
The Global Disaster and Economic Impact Dashboard is an interactive analytical tool designed to explore how natural disasters influence global economic stability between 2020 and 2025. Built using Streamlit, this project integrates real-world data from global organizations and provides a unified platform for visualizing the intersection between environmental and economic systems.

The core idea behind this project is to help researchers, policymakers, and data enthusiasts understand how patterns of natural disasters correlate with shifts in economic indicators such as GDP growth, inflation, and unemployment rates. By combining datasets on disaster occurrences with key macroeconomic metrics, the dashboard reveals how different regions have been impacted over time, and which economies appear most resilient or vulnerable to environmental shocks.

Through an interactive web interface, users can:

- Explore disaster frequencies and types across years to identify changing climate trends.
- Analyze economic indicators (GDP, inflation, interest rates, etc.) for individual countries or regions and observe long-term growth or decline.
- Visualize global disaster impact maps, revealing population exposure, affected regions, and intensity patterns.
- Compare cross-country trends, allowing deeper insights into the relationship between environmental crises and economic response.

The accompanying Jupyter Notebook documents the data cleaning, imputation methods, and exploratory data analysis, ensuring reproducibility and transparency. Together, the notebook and dashboard form a comprehensive analytical pipeline that bridges data science, environmental research, and economic modeling.

---

### Initial Data Analysis (IDA)
The initial data analysis (IDA) phase focused on understanding the structure, completeness, and alignment of the datasets before integration. While the disaster dataset was complete, the indicators dataset exhibited substantial missingness. 

To better understand these missing values, a missingness heatmap was generated to visually inspect the patterns of missing data across variables and years. 

Key observations:
- horizontal bands indicated specific countries had missing data across multiple variables (particularly for 2024 and 2025)
- vertical bands which indicated that some variables lacked data across many countries and years

From these observations, we could conclude that this structure aligned with a Missing At Random (MAR) mechanism, meaning missing values could be reasonably explained by observed data patterns.

---

### Data Cleaning and Preprocessing
Using the missingness mechanism we identified from the IDA step, a two-step imputation method was implemented to produce high quality, temporally consistent, and cross-sectionally comparable data.

**Step 1:** Stochastic Regression Imputation
- For each country-indicator pair, a simple linear regression model was fitted using year as the predictor and the indicator’s observed values as the response.
- Missing values were then predicted from the model, and random noise (based on residual standard deviation) was added to preserve realistic variability.
- Indicators with insufficient data (too few observations for regression) were left as NaN for Step 2.

This approach preserved temporal continuity within each country’s data.

**Step 2:** KNN Imputation (Per-Year Basis)
- For each year, a k-nearest neighbors imputer was applied to fill remaining missing values.
- Missing indicators were inferred from similar countries within the same year based on other available indicators.
  
Missing indicators for each country were inferred from similar countries in that same year, using other available indicators as features.

**Additional Cleaning Steps**
- Removed non-feature columns (event_id, country_id, ect.)
- Standardized values (converting current USD to million USD).
- Filtered economic data for only years 2020 - 2025 (to match disasters dataset).

---

### Exploratory Data Analysis (EDA)
After cleaning and imputation, exploratory data analysis (EDA) was conducted to uncover patterns and insights linking disasters to economic performance.

**Disaster Counts:**
- A boxplot visualization was created to summarize the yearly distribution of disaster counts, affected populations, and casualties across different disaster types.
- The plot revealed that drought and earthquakes were the most frequent types of disasters.

**Economic Indicators:**
- An interactive line plot was used to visualize trends in key economic indicators, such as GDP growth, inflation, unemployment, and interest rates, selectable via a dropdown menu.
- Inflation trends varied regionally. Some countries saw inflation spikes following major disasters, reflecting the short-term economic disruptions caused by recovery costs and supply shortages.
- The interactive component allows users to explore how each indicator evolved over time, comparing trends across countries and regions to assess resilience and volatility.

**Global Visualizations:**
- The choropleth map provided a global overview of disaster impact by visualizing countries colored according to their total economic impact for all years (2020-2025).
- This map highlighted regions with the highest exposure, showing dense concentrations of economic impact in North America, South/Central America, and Asia.

---

### Streamlit Web App Dashboard
The dashboard provides an intuitive platform for interacting with the processed datasets. Users can dynamically visualize disaster patterns and economic trends through linked visual components, including maps, charts, and selection filters.

The app is organized into four main pages:

1. Overview: Explains the goals and narrative of the project, as well as providing a summary of the global disaster and economic indicators datasets.

2. Disaster Counts: Bar chart displaying the counts of each type of disaster, affected population, and number of casualties.

3. Economic Indicators: Line plot visualization of the GDP, inflation, and other macroeconomic measures for countries.

4. Disaster Explorer: Interactive global map visualizing geographical locations of disasters along with affected population.

The Streamlit web app for this repo is already deployed through Streamlit's community cloud and can be accessed [here](https://cmse830project-3rbvj3nlqbxc4wdsnwa6du.streamlit.app/).

---

### Installation Instructions
1. Clone the repository.

2. Install required dependencies from the `requirements.txt` file.

3. (if planning to run jupyter notebook) Download original datasets into the `datasets` folder.
- Dataset 1: [Global Climate Events and Economic Impact Dataset](https://www.kaggle.com/datasets/uom190346a/global-climate-events-and-economic-impact-dataset/data)

- Dataset 2: [Global Economic Indicators (2010–2025) - World Bank](https://www.kaggle.com/datasets/tanishksharma9905/global-economic-indicators-20102025)

4. Run the Streamlit web dashboard through `App_Overview.py`.