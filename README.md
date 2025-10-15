## CMSE830 Project: Global Disaster and Economic Impact Dashboard (2020 - 2025)

### Overview
The Global Disaster and Economic Impact Dashboard is an interactive analytical tool designed to explore how natural disasters influence global economic stability between 2020 and 2025. Built using Streamlit, this project integrates real-world data from global organizations and provides a unified platform for visualizing the intersection between environmental and economic systems.

The core idea behind this project is to help researchers, policymakers, data enthusiasts, understand how patterns of natural disasters correlate with shifts in economic indicators such as GDP growth, inflation, and unemployment rates. By combining datasets on disaster occurrences with key macroeconomic metrics, the dashboard reveals how different regions have been impacted over time, and which economies appear most resilient or vulnerable to environmental shocks.

Through an interactive web interface, users can:

- Explore disaster frequencies and types across years to identify changing climate trends.
- Analyze economic indicators (GDP, inflation, interest rates, etc.) for individual countries or regions and observe long-term growth or decline.
- Visualize global disaster impact maps, revealing population exposure, affected regions, and intensity patterns.
- Compare cross-country trends, allowing deeper insights into the relationship between environmental crises and economic response.

The accompanying Jupyter Notebook documents the data cleaning, imputation methods, and exploratory data analysis, ensuring reproducibility and transparency. Together, the notebook and dashboard form a comprehensive analytical pipeline that bridges data science, environmental research, and economic modeling.

---
### Installation Instructions
1. Clone the repository

2. Install required dependencies from the `requirements.txt` file

3. (if planning to run jupyter notebook) Download original datasets into the `datasets` folder
- Dataset 1: [Global Climate Events and Economic Impact Dataset](https://www.kaggle.com/datasets/uom190346a/global-climate-events-and-economic-impact-dataset/data)

- Dataset 2: [Global Economic Indicators (2010–2025) - World Bank](https://www.kaggle.com/datasets/tanishksharma9905/global-economic-indicators-20102025)

4. Run the Streamlit web dashboard through `App_Overview.py`

### App Deployment
The Streamlit web app for this repo is already deployed through Streamlit's community cloud and can be accessed [here](https://cmse830project-3rbvj3nlqbxc4wdsnwa6du.streamlit.app/)
