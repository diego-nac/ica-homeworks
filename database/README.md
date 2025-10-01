
# Air Quality Dataset

This repository documents and organizes the **Air Quality** dataset, available at the [UCI Machine Learning Repository](https://doi.org/10.24432/C59K5F).
The dataset contains **responses from a chemical multisensor device** deployed in an Italian city, recording hourly measurements of air pollutants and environmental variables.

---

## 📌 General Information

- **Name:** Air Quality
- **Donation Date:** March 22, 2016
- **Subject Area:** Computer Science
- **Associated Tasks:** Regression
- **Data Characteristics:** Multivariate, Time-Series
- **Feature Types:** Real-valued and categorical
- **Number of Instances:** 9,358
- **Number of Features:** 15
- **Missing Values:** Yes (tagged with `-200`)
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

---

## 📊 Dataset Description

The dataset contains **9,358 instances** of hourly averaged responses from **five metal oxide chemical sensors**, monitoring atmospheric pollutants and environmental variables.

- Collection period: **March 2004 to February 2005**
- Location: Road-level in a heavily polluted area of an Italian city
- Sensors recorded:
  - Carbon Monoxide (CO)
  - Non-Methane Hydrocarbons (NMHC)
  - Benzene (C6H6)
  - Nitrogen Oxides (NOx, NO2)
- Additional environmental measurements: temperature, relative humidity, absolute humidity

⚠️ **Important Notes:**

- Evidence of **sensor drift** and **cross-sensitivity** is present.
- The dataset is intended for **research purposes only** (commercial use prohibited).

---

## 📑 Variables Structure

| Variable Name | Type        | Description                                       | Unit    | Missing Values |
| ------------- | ----------- | ------------------------------------------------- | ------- | -------------- |
| Date          | Date        | Date in DD/MM/YYYY format                         | -       | No             |
| Time          | Categorical | Time in HH.MM.SS format                           | -       | No             |
| CO(GT)        | Integer     | True hourly averaged CO concentration (reference) | mg/m³  | No             |
| PT08.S1(CO)   | Numeric     | Tin oxide sensor response (nominally CO)          | -       | Yes            |
| NMHC(GT)      | Integer     | True hourly averaged NMHC concentration           | µg/m³ | No             |
| C6H6(GT)      | Continuous  | True hourly averaged Benzene concentration        | µg/m³ | No             |
| PT08.S2(NMHC) | Numeric     | Titania sensor response (nominally NMHC)          | -       | Yes            |
| NOx(GT)       | Integer     | True hourly averaged NOx concentration            | ppb     | No             |
| PT08.S3(NOx)  | Numeric     | Tungsten oxide sensor response (nominally NOx)    | -       | Yes            |
| NO2(GT)       | Integer     | True hourly averaged NO2 concentration            | µg/m³ | No             |
| PT08.S4(NO2)  | Numeric     | Tungsten oxide sensor response (nominally NO2)    | -       | Yes            |
| PT08.S5(O3)   | Numeric     | Indium oxide sensor response (nominally O3)       | -       | Yes            |
| Temperature   | Continuous  | Ambient temperature                               | °C     | No             |
| RH            | Continuous  | Relative Humidity                                 | %       | No             |
| AH            | Continuous  | Absolute Humidity                                 | g/m³   | No             |

---

## 📂 Available Files

- **AirQualityUCI.xlsx** (1.2 MB)
- **AirQualityUCI.csv** (766.7 KB)

---

## 📚 Related Publications

- De Vito, S., Massera, E., Piga, M., Martinotto, L., & Francia, G. (2008).
  *On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario*.
  **Sensors and Actuators B: Chemical.**

### Citations using this dataset:

- Agarwal, N. et al. (2019). *Boosting for Dynamical Systems*. **ArXiv**.
- Jang, J.-G. et al. (2018). *Zoom-SVD: Fast and Memory Efficient Method for Extracting Key Patterns in an Arbitrary Time Range*. **CIKM '18**.
- Sundin, M. et al. (2015). *Combined modeling of sparse and dense noise for improvement of Relevance Vector Machine*.

---

## 🔧 How to Use

Install the `ucimlrepo` package to directly access this dataset:

```bash
pip install ucimlrepo
```
