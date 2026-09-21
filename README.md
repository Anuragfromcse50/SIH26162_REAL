# 🔥 SIH26162 REAL - Satellite-Based Fire Detection & Classification

## 📌 Project Overview

SIH26162 REAL is a satellite-based thermal source detection and classification prototype developed for the Smart India Hackathon problem statement SIH26162.

The project uses NASA FIRMS (Fire Information for Resource Management System) satellite observations to detect and analyse thermal anomalies using data processing, spatial persistence analysis, machine learning and GIS visualization.

The system provides an interactive Streamlit dashboard where users can:

• View satellite fire detections on an interactive GIS map
• Filter fire detections
• Analyse thermal properties
• Analyse persistence of thermal sources
• Click an individual map detection
• Predict its fire/thermal class using a Random Forest model
• Enter thermal parameters manually and get a prediction
• View prediction probabilities
• Download prediction results as CSV

---

## 🎯 Problem Statement

Satellite sensors can detect many thermal anomalies on the Earth's surface.

However, every thermal anomaly does not represent the same type of event.

A thermal detection may represent:

• Agricultural burning
• Forest or wildlife fire
• Industrial thermal activity
• Other thermal anomalies

The objective of this project is to build a prototype system that processes satellite thermal observations and provides a way to analyse and classify these detections.

---

## 💡 Proposed Solution

The system combines:

• NASA FIRMS satellite data
• Data preprocessing
• Feature engineering
• Spatial persistence analysis
• Rule-based prototype classification
• Random Forest machine learning
• Interactive GIS visualization
• Streamlit dashboard

### Overall System Flow

NASA FIRMS Satellite Data
        ↓
Data Preprocessing
        ↓
Data Validation
        ↓
Feature Engineering
        ↓
Spatial Persistence Analysis
        ↓
Prototype Classification
        ↓
Random Forest Model
        ↓
Processed Fire Data
        ↓
Streamlit Dashboard
        ↓
 ┌───────────────────────┐
 │                       │
 ▼                       ▼
GIS Dashboard       ML Prediction
 │                       │
 ▼                       ▼
Interactive Map      Manual Input
 │                       │
 ▼                       ▼
Click Detection      Feature Creation
 │                       │
 └──────────┬────────────┘
            ↓
      Random Forest
            ↓
    Fire Class Prediction
            ↓
 Prediction Probability
            ↓
      CSV Download

---

## 🛰️ Data Source

The project uses NASA FIRMS (Fire Information for Resource Management System) satellite fire detection data.

The main satellite dataset used in this prototype is:

VIIRS SNPP (Suomi NPP)

The downloaded prototype dataset contains approximately 7,615 satellite observations covering:

1 July 2026 to 19 September 2026

The dataset contains satellite observations of thermal anomalies along with their geographic and thermal properties.

---

## 📊 NASA FIRMS Data

Important fields available in the dataset include:

• LATITUDE - Latitude of the thermal detection
• LONGITUDE - Longitude of the thermal detection
• BRIGHTNESS - Satellite brightness temperature
• SCAN - Scan value
• TRACK - Track value
• ACQ_DATE - Acquisition date
• ACQ_TIME - Acquisition time
• SATELLITE - Satellite name
• INSTRUMENT - Satellite instrument
• CONFIDENCE - FIRMS detection confidence
• VERSION - FIRMS data version
• BRIGHT_T31 - Background brightness temperature
• FRP - Fire Radiative Power
• DAYNIGHT - Day or night observation
• geometry - Spatial geometry

---

## 🧹 Data Preprocessing

The raw NASA FIRMS data is first loaded and analysed.

The preprocessing stage includes:

• Loading the FIRMS dataset
• Checking available columns
• Checking missing values
• Checking coordinate system
• Checking numerical statistics
• Converting acquisition dates
• Preparing latitude and longitude
• Creating additional features
• Preparing the machine learning dataset

---

## 🧠 Feature Engineering

Additional features are created from the original NASA FIRMS observations.

### Temperature Difference

Temperature difference is calculated between the observed brightness temperature and background temperature.

TEMP_DIFF = BRIGHTNESS - BACKGROUND_TEMP

This represents how much hotter the detected location is compared with its background.

### FRP Log Transformation

FRP can contain a wide range of values.

Therefore a logarithmic transformation is used:

FRP_LOG = log(1 + FRP)

This reduces the effect of very large FRP values.

### Day/Night Feature

The original DAYNIGHT information is converted into a numerical feature.

Day = 0
Night = 1

The resulting feature is:

IS_NIGHT

### Confidence Encoding

FIRMS confidence values are converted into numerical values:

Low = 0
Nominal = 1
High = 2

The resulting feature is:

CONFIDENCE_NUM

---

## 🔁 Spatial Persistence Analysis

A thermal source that appears repeatedly at approximately the same location may be more persistent than a single observation.

To analyse this, latitude and longitude are rounded to two decimal places to create an approximate spatial grid.

LAT_GRID = rounded latitude
LON_GRID = rounded longitude

The number of unique observation dates for each approximate location is then calculated.

This creates:

PERSISTENCE_DAYS

### Persistence Flow

Latitude + Longitude
        ↓
Approximate Spatial Grid
        ↓
Group Observations
        ↓
Count Unique Observation Dates
        ↓
PERSISTENCE_DAYS
        ↓
Persistent Source Identification

The current prototype uses:

PERSISTENCE_DAYS >= 3

as the persistent-source threshold.

---

## 🔥 Prototype Fire Classes

The dashboard uses four conceptual fire or thermal-source classes:

🏭 Industrial Fire

🌾 Agricultural Fire

🌲 Forest / Wildlife Fire

🌡️ Other Thermal Anomaly

---

## ⚙️ Prototype Classification Logic

The current prototype creates classification labels using available satellite features and rule-based logic.

### Agricultural Fire

A prototype agricultural condition is:

PERSISTENCE_DAYS <= 2
AND
FRP >= 1

### Forest / Wildlife Fire

A prototype forest/wildlife condition is:

PERSISTENCE_DAYS >= 3
AND
FRP >= 3

### Other Thermal Anomaly

Detections that do not satisfy the prototype conditions are classified as:

Other Thermal Anomaly

### Industrial Fire

Industrial Fire classification is intended to use reliable industrial reference or contextual data.

The project does not treat arbitrary geographic points as verified industrial facilities.

---

## 🤖 Machine Learning

The project uses a Random Forest Classifier for prototype fire/thermal-source classification.

The trained model is saved as:

outputs/fire_model.pkl

The same trained model is used for:

• Manual ML prediction
• Map-based prediction

---

## 🧮 Machine Learning Features

The Random Forest model uses the following features:

• BRIGHTNESS
• BACKGROUND_TEMP
• FRP
• TEMP_DIFF
• FRP_LOG
• PERSISTENCE_DAYS
• IS_NIGHT
• CONFIDENCE_NUM

The same feature structure is used during prediction so that the input provided to the model remains consistent with the training data.

---

## 🌳 Random Forest Model

The machine learning model is a Random Forest Classifier.

Main configuration:

• Number of estimators: 300
• Random state: 42
• Class weight: balanced

Random Forest is used because it can work with multiple numerical features and can provide prediction probabilities for different classes.

---

## 🗺️ GIS Dashboard

The project provides an interactive GIS dashboard using Folium and Streamlit.

Each thermal detection is displayed on an interactive map using large circular markers.

Users can:

• View thermal detections
• Move around the map
• Zoom the map
• Apply filters
• Click a detection
• View detection information
• Run ML prediction
• View prediction confidence
• View class probabilities
• Download prediction results

---

## 📍 Map-Based Prediction Flow

User opens GIS map
        ↓
Thermal detections are displayed
        ↓
User clicks a detection
        ↓
Nearest FIRMS observation is identified
        ↓
Required ML features are prepared
        ↓
Random Forest model is used
        ↓
Fire class is predicted
        ↓
Prediction probability is calculated
        ↓
Result is displayed
        ↓
Prediction can be downloaded as CSV

---

## 🤖 Manual ML Prediction

The dashboard also provides a separate Random Forest ML prediction section.

The user can enter:

• Brightness
• Background Temperature
• FRP
• Persistence Days
• Day/Night
• FIRMS Confidence

The application automatically calculates:

• Temperature Difference
• Log FRP
• Night Indicator
• Numerical Confidence

The complete feature vector is then passed to the trained Random Forest model.

---

## 📊 Prediction Output

The dashboard displays:

• Predicted Fire Class
• Model Confidence
• Class Probabilities
• Input Information
• Downloadable Prediction Result

The prediction depends on the values entered by the user and the trained prototype model.

---

## 📥 Download Feature

The application allows users to download prediction results as CSV files.

Examples include:

• selected_fire_prediction.csv
• manual_ml_fire_prediction.csv

This allows prediction results to be saved and analysed separately.

---

## 🖥️ Streamlit Dashboard

The application is built using Streamlit.

The dashboard contains two main sections.

### 🗺️ GIS / Rule-Based Dashboard

This section provides:

• Interactive fire detection map
• Fire class filter
• Minimum persistence filter
• Date filter
• Total detection count
• Average FRP
• Maximum FRP
• Persistent detection count
• Interactive map markers
• Selected detection prediction
• Prediction probability
• CSV download

### 🤖 Random Forest ML Prediction

This section provides:

• Manual input fields
• Automatic feature calculation
• Random Forest prediction
• Prediction probability
• Model confidence
• CSV download

---

## 📈 Dashboard Flow

                    STREAMLIT APPLICATION
                            │
             ┌──────────────┴──────────────┐
             │                             │
             ▼                             ▼
      GIS Dashboard                 ML Prediction
             │                             │
             ▼                             ▼
        Apply Filters                 Enter Values
             │                             │
             ▼                             ▼
        Display Map                 Feature Creation
             │                             │
             ▼                             ▼
       Click Detection               Random Forest
             │                             │
             ▼                             ▼
      Select FIRMS Data               Prediction
             │                             │
             └──────────────┬──────────────┘
                            ▼
                   Probability / Confidence
                            │
                            ▼
                      CSV Download

---

## 🏗️ Project Structure

SIH26162_REAL/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── firms/
│       └── Readme.txt
│
├── notebooks/
│   └── 01_firms_analysis.ipynb
│
├── src/
│   └── test_firms.py
│
├── outputs/
│   ├── fire_model.pkl
│   ├── processed_firms_data.csv
│   └── manual_fire_prediction.csv
│
└── venv/

---

## 🛠️ Technologies Used

### Programming

• Python

### Data Processing

• Pandas
• NumPy

### Geospatial Processing

• GeoPandas
• Shapely
• PyProj

### Machine Learning

• Scikit-learn
• Random Forest

### GIS and Visualization

• Folium
• Streamlit
• Streamlit-Folium

### API and Environment

• Requests
• Python-dotenv
• NASA FIRMS API

### Development Tools

• Visual Studio Code
• Jupyter Notebook
• Git
• GitHub

---

## 📦 Installation

The project can be cloned from the GitHub repository and run locally using a Python virtual environment.

After cloning the project:

1. Open the project folder in Visual Studio Code.
2. Create a Python virtual environment.
3. Activate the virtual environment.
4. Install the packages listed in requirements.txt.
5. Run the Streamlit application.

The main dependencies are:

• pandas
• numpy
• geopandas
• shapely
• pyproj
• requests
• folium
• streamlit
• streamlit-folium
• python-dotenv
• scikit-learn

---

## ▶️ Running the Application

After installing the dependencies, start the Streamlit application.

The dashboard normally runs locally at:

http://localhost:8501

The application can then be opened in a web browser.

---

## 🔑 NASA FIRMS MAP_KEY

NASA FIRMS API access can use a MAP_KEY.

The API key must remain private.

The key is stored using an environment variable rather than directly inside the public source code.

The project uses a .env file for the key.

The .env file is excluded from GitHub using .gitignore.

The actual NASA FIRMS MAP_KEY should never be published in:

• GitHub
• README
• Source code
• Screenshots
• Public documentation

---

## 🌐 NASA FIRMS API

The project supports the use of the NASA FIRMS API for obtaining updated satellite thermal detection data.

The API request uses:

• NASA FIRMS MAP_KEY
• Satellite dataset
• Geographic bounding box
• Number of days

The general flow is:

NASA FIRMS MAP_KEY
        ↓
NASA FIRMS API
        ↓
VIIRS SNPP Data
        ↓
Thermal Detection Data
        ↓
Data Processing
        ↓
Dashboard / ML Pipeline

The API can be used to obtain newer FIRMS observations.

The current dashboard can also operate using the downloaded and processed FIRMS dataset.

---

## 📁 Raw NASA FIRMS Files

NASA FIRMS shapefile data consists of multiple related files.

Examples include:

• .shp
• .shx
• .dbf
• .prj
• .cpg

These files must remain together for the shapefile to work correctly.

The raw FIRMS files are intentionally not uploaded to GitHub because some files are very large.

The processed data and required project files are maintained separately.

---

## 🔒 Security

The following should not be uploaded to GitHub:

• .env
• venv/
• .venv/

API keys and private credentials should always be stored securely using environment variables.

---

## ☁️ Streamlit Cloud Deployment

The project is designed to be deployed using Streamlit Community Cloud.

Deployment flow:

GitHub Repository
        ↓
Streamlit Community Cloud
        ↓
Select Repository
        ↓
Select Main Branch
        ↓
Select app.py
        ↓
Read requirements.txt
        ↓
Install Dependencies
        ↓
Load fire_model.pkl
        ↓
Start Streamlit Application
        ↓
Online Dashboard

The requirements.txt file contains all required Python dependencies.

Scikit-learn is required because the application loads the trained Random Forest model.

---

## 📄 Requirements

The main project dependencies are:

• pandas
• numpy
• geopandas
• shapely
• pyproj
• requests
• folium
• streamlit
• streamlit-folium
• python-dotenv
• scikit-learn

---

## 📤 Output Files

The project uses the outputs directory for processed files and model files.

Main output files include:

• fire_model.pkl - trained Random Forest model
• processed_firms_data.csv - processed FIRMS data used by the dashboard
• manual_fire_prediction.csv - manual prediction result

---

## 🔄 Complete End-to-End Project Flow

START
  ↓
NASA FIRMS Satellite Data
  ↓
Load Data
  ↓
Data Validation
  ↓
Data Cleaning and Preparation
  ↓
Feature Engineering
  ↓
Temperature Difference
  ↓
FRP Log Transformation
  ↓
Day/Night Encoding
  ↓
Confidence Encoding
  ↓
Spatial Grid Creation
  ↓
Persistence Analysis
  ↓
Prototype Classification
  ↓
Machine Learning Dataset
  ↓
Random Forest Training
  ↓
Save fire_model.pkl
  ↓
Prepare Dashboard Data
  ↓
Streamlit Application
  ↓
 ┌─────────────────────┐
 │                     │
 ▼                     ▼
GIS Dashboard       ML Dashboard
 │                     │
 ▼                     ▼
Map Filters         Manual Inputs
 │                     │
 ▼                     ▼
Click Fire          Feature Creation
 │                     │
 ▼                     ▼
Prepare Features    Random Forest
 │                     │
 └──────────┬──────────┘
            ↓
      Fire Prediction
            ↓
   Confidence + Probability
            ↓
       CSV Download
            ↓
           END

---

## ⚠️ Important Limitations

This project is a prototype.

NASA FIRMS provides satellite-based thermal detections, but a thermal detection does not automatically provide a verified real-world fire type.

Therefore:

• Current fire classes are prototype classes.
• Classification labels are generated using available satellite features and rules.
• These labels should not be treated as verified ground truth.
• The Random Forest model is a prototype model.
• Model accuracy should not be interpreted as real-world fire-type accuracy without an independently verified labelled dataset.
• Industrial Fire classification requires reliable industrial reference data.
• Additional validation is required before operational use.

---

## 🚀 Future Scope

The project can be further improved by adding:

### Real-Time FIRMS Updates

Automatically fetch new satellite detections at regular intervals.

### OpenStreetMap Integration

Use OSM data to identify nearby industrial areas and facilities.

### Satellite Imagery

Integrate additional satellite imagery for better contextual analysis.

### Land Cover Information

Use land-cover information to distinguish between:

• Agricultural areas
• Forest areas
• Industrial areas
• Urban areas

### Verified Ground Truth

Create a reliable labelled dataset using verified fire events.

### Advanced Machine Learning

Future versions can experiment with:

• XGBoost
• LightGBM
• Neural Networks
• Deep Learning

### Real-Time Alerts

Generate alerts when high-confidence or persistent thermal sources are detected.

### Historical Analysis

Analyse thermal activity over longer time periods.

### Risk Analysis

Future versions can combine:

• FRP
• Persistence
• Location
• Land Cover
• Distance from Industrial Areas
• Historical Activity

to generate a more advanced risk score.

### Improved GIS

Future versions can include:

• Heatmaps
• Clustering
• Time slider
• Historical layers
• Industrial zones
• Forest boundaries
• Agricultural areas
• Risk zones

---

## ⭐ Key Features

✅ NASA FIRMS Satellite Data

✅ VIIRS SNPP Thermal Detection

✅ Thermal Anomaly Analysis

✅ Data Preprocessing

✅ Feature Engineering

✅ Spatial Persistence Analysis

✅ Prototype Fire Classification

✅ Random Forest Machine Learning

✅ Interactive GIS Map

✅ Streamlit Dashboard

✅ Map-Based Prediction

✅ Manual ML Prediction

✅ Prediction Probabilities

✅ CSV Download

✅ NASA FIRMS API Support

✅ GitHub Ready

✅ Streamlit Cloud Deployment Ready

---

## 👨‍💻 Development

This project was developed as a prototype for the Smart India Hackathon SIH26162 problem statement.

The main purpose is to demonstrate a complete workflow from satellite thermal detection data to data processing, spatial analysis, machine learning, GIS visualization and interactive web-based prediction.

The project provides a foundation that can be improved further using verified ground-truth data, reliable industrial reference information, additional satellite imagery, land-cover information and real-time processing.

---

## 📜 Disclaimer

This project is intended for educational, research and prototype purposes.

It is not intended to replace official fire detection systems, emergency response systems or government monitoring systems.

Satellite thermal detections and prototype ML predictions should be independently verified before being used for operational decisions.

---

## 🎯 Conclusion

SIH26162 REAL demonstrates how NASA FIRMS satellite thermal detection data can be combined with:

NASA FIRMS
+
Python
+
Data Processing
+
Feature Engineering
+
Spatial Analysis
+
Machine Learning
+
GIS
+
Streamlit

to create an interactive prototype for analysing and classifying thermal anomalies.

The system provides a complete workflow from satellite data ingestion to interactive visualization and machine learning prediction, while leaving scope for future improvement through better labelled datasets, verified industrial information, additional satellite data and real-time processing.
