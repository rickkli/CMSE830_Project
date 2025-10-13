## CMSE830 Project

Dataset 1: [Climate Economics: Risks and Realities (2020 - 2025)](https://www.kaggle.com/datasets/mubeenshehzadi/climate-economics-risks-and-realities-2020-2025)

   Possibile Alternative for Dataset 1: [Global Climate Events and Economic Impact Dataset](https://www.kaggle.com/datasets/uom190346a/global-climate-events-and-economic-impact-dataset/data)
   
   Dataset 2: [Global Economic Indicators (2010–2025) - World Bank](https://www.kaggle.com/datasets/tanishksharma9905/global-economic-indicators-20102025)


Missing Values Notes:
- heatmap of indicies vs features
- the horizontal lines of missing represent missing values for many variables (2024, 2025) for each country
    - specific year for one country has many missing data for several variables
- the chunks of missing values (vertical) represent a variable that has missing values for many rows (years)
    - one (or multiple) variables have missing data for multiple years
- Conclusion:
The missing data structure from the heatmap suggests MAR, as missingness is related/explainable to observed factors

Imputation Notes:
- Two Step Imputation Method
- Step 1: Stochastic Regression
  - for each country and each indicator, a linear regression model is fitted using year as the predictor and the indicator's observed values as the response
  - after predicitng missing values, random noise is added based on the redsidual standard deviation from the fitted model
  - if variable couldn't be fit (missing too many observations), leave as NaN for step 2
  - preserves temporal continuity within countires
    
- Step 2: KNN (per-year)
  - remaining missing values filled using KNN imputer applied separately for each year
  - each country's missing indicator values in a given year are estimated from similar countries in that same year (based on other available indicators)
  - ensures imputed values come only from other countries in same year, maintaining year-specific relationships
  - captures cross sectional similairty between countries