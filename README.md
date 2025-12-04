## CMSE830 Project: Global Disaster, Economic, and Demographic Explorer (2020–2025)

### Overview
The Global Disaster, Economic, and Demographic Explorer Dashboard is an interactive Streamlit web application designed to explore how natural disasters affect global economies and populations between 2020 and 2025. By integrating three core datasets, the app provides a unified platform for analyzing the relationship between environmental shocks, economic performance, and human vulnerability.

The dashboard is intended for researchers, policymakers, analysts, and students who want to understand how climate-driven disasters shape economic and social outcomes across different regions. Users can investigate disaster frequency and severity, examine economic trends such as GDP growth and inflation, and assess demographic factors that influence a country’s resilience.

The goal of the project is to present a clear narrative of how environmental, economic, and demographic systems interact. Through visualizations, comparisons, and interactive filtering, the app highlights patterns such as which countries experience the greatest disruptions from disasters, how economic indicators vary before and after major events, and how demographic context influences risk and recovery. Ultimately, the dashboard serves as a tool for exploring global vulnerability and resilience relationships.

The accompanying Jupyter Notebooks document the data cleaning, imputation methods, and exploratory data analysis, ensuring reproducibility and transparency. Together, the notebook and dashboard form a comprehensive analytical pipeline that bridges data science, environmental research, and economic modeling.

---

### Initial Data Analysis (IDA)
The initial data analysis (IDA) phase focused on understanding the structure, completeness, and alignment of the datasets before integration. While the disaster dataset was complete, the indicators and demographics datasets exhibited missingness. 

To better understand these missing values, a missingness heatmap was generated to visually inspect the patterns of missing data across variables and years for both datasets.

Key observations:
- horizontal bands indicated specific countries had missing data across multiple variables
- vertical bands which indicated that some variables lacked data across many countries and years

From these observations, we could conclude that this structure aligned with a Missing At Random (MAR) mechanism, meaning missing values could be reasonably explained by observed data patterns.

---

### Data Cleaning and Preprocessing
Using the missingness mechanism we identified from the IDA step, a two-step imputation method was implemented to produce high quality, temporally consistent, and cross-sectionally comparable data.

**Step 1:** Stochastic Regression Imputation
- For each country-feature pair, a simple linear regression model was fitted using year as the predictor and the feature’s observed values as the response.
- Missing values were then predicted from the model, and random noise (based on residual standard deviation) was added to preserve realistic variability.
- Features with insufficient data (too few observations for regression) were left as NaN for Step 2.

**Step 2:** KNN Imputation (Per-Year Basis)
- For each year, a k-nearest neighbors imputer was applied to fill remaining missing values.
- Missing features were inferred from similar countries within the same year based on other available features.

**Additional Cleaning Steps**
- Removed non-feature columns (event_id, country_id, ect.)
- Standardized values (converting current USD to million USD).
- Filtered economic data for only years 2020 - 2025 (to match disasters dataset).

---

### Exploratory Data Analysis (EDA)
After cleaning and imputation, exploratory data analysis (EDA) was conducted to uncover patterns and insights linking disasters to economic performance.

**Disaster Event Counts:**
- A boxplot visualization was created to summarize the yearly distribution of disaster counts, affected populations, and casualties across different disaster types.
- The plot revealed that drought and earthquakes were the most frequent types of disasters.

**Economic Indicators:**
- An interactive line plot was used to visualize trends in key economic indicators, such as GDP growth, inflation, unemployment, and interest rates, selectable via a dropdown menu.
- Inflation trends varied regionally. Some countries saw inflation spikes following major disasters, reflecting the short-term economic disruptions caused by recovery costs and supply shortages.
- The interactive component allows users to explore how each indicator evolved over time, comparing trends across countries and regions to assess resilience and volatility.

**Choropleth Map:**
- The choropleth map provided a global overview of disaster impact by visualizing countries colored according to their total economic impact for all years.

**Heatmap:**
- This heatmap highlighted geographical locations with the highest frequency of disasters along with their intensity and impact on populations.

---

### Streamlit Web App Dashboard
The dashboard provides an intuitive platform for interacting with the processed datasets. Users can dynamically visualize disaster patterns and economic trends through linked visual components, including maps, charts, and selection filters.

The app is organized into four main pages:

1. App Overview: Explains the goals and narrative of the project, as well as providing a summary of the datasets.

2. Initial Data Analysis: Explains the cleaning and preprocessing steps performed on the datasets, as well as the missingness analysis and imputation techniques employed.

3. Exploratory Data Analysis: Contains interactive exploratory visualizations and statistical summaries into the three datasets.

4. Models: Interactive machine learning models for severity prediction and country diasaster vulnerability clustering.

The Streamlit web app for this repo is already deployed through Streamlit's community cloud and can be accessed [here](https://cmse830project-3rbvj3nlqbxc4wdsnwa6du.streamlit.app/).

---

### Installation Instructions
1. Clone the repository.

2. Install required dependencies from the `requirements.txt` file.

3. (if planning to run jupyter notebook) Download original datasets into the `datasets` folder.

- Dataset 1: [Global Climate Events and Economic Impact Dataset](https://www.kaggle.com/datasets/uom190346a/global-climate-events-and-economic-impact-dataset/data)

- Dataset 2: [Global Economic Indicators (2010–2025) - World Bank](https://www.kaggle.com/datasets/tanishksharma9905/global-economic-indicators-20102025)

- Dataset 3: [United States Census Bureau](https://www.census.gov/data-tools/demo/idb/#/table?dashboard_page=country&COUNTRY_YR_ANIM=2025&menu=tableViz&show_countries=y&CCODE=AR,AU,AT,BD,BE,BR,CA,CL,CN,CO,CZ,DK,EG,FI,FR,DE,GR,HU,IN,ID,IQ,IE,IL,IT,JP,KZ,KR,MY,MX,NL,NZ,NG,PK,PE,PH,PL,PT,QA,RO,RU,SA,SG,ZA,SE,CH,TH,AE,GB,US,VN,TR&TABLE_RANGE=2020,2025&TABLE_YEARS=2020,2021,2022,2023,2024,2025&TABLE_USE_RANGE=Y&TABLE_USE_YEARS=Y&TABLE_STEP=1&TABLE_ADD_YEARS=2025&quickReports=OVW)

4. Run the Streamlit web dashboard through `App_Overview.py`.