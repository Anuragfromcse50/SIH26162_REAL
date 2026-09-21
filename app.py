# ============================================================
# SIH26162_REAL
# NASA FIRMS Thermal Source Detection Dashboard
# ============================================================

# -----------------------------
# IMPORT LIBRARIES
# -----------------------------
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import folium

from streamlit_folium import st_folium


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SIH26162 Thermal Source Detection",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

# Current app.py ke parent folder ko project folder maana jayega.
BASE_DIR = Path(__file__).resolve().parent

# Processed NASA FIRMS data.
DATA_PATH = BASE_DIR / "outputs" / "processed_firms_data.csv"

# Saved Random Forest model.
MODEL_PATH = BASE_DIR / "outputs" / "fire_model.pkl"


# ============================================================
# LOAD FIRMS DATA
# ============================================================

@st.cache_data
def load_fire_data():
    """
    Processed NASA FIRMS CSV ko load karta hai.
    """

    return pd.read_csv(DATA_PATH)


# ============================================================
# LOAD ML MODEL
# ============================================================

@st.cache_resource
def load_fire_model():
    """
    Saved Random Forest model ko pickle se load karta hai.
    """

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    return model


# ============================================================
# SAFE COLUMN HELPER
# ============================================================

def get_value(row, upper_name, lower_name=None, default=None):
    """
    FIRMS data me kabhi uppercase aur kabhi lowercase
    column names ho sakte hain.

    Ye function dono ko safely handle karta hai.
    """

    if upper_name in row.index:
        return row[upper_name]

    if lower_name is not None and lower_name in row.index:
        return row[lower_name]

    return default


# ============================================================
# LOAD DATA
# ============================================================

try:

    # NASA FIRMS processed data load karo.
    fires = load_fire_data()

except FileNotFoundError:

    st.error(
        "❌ processed_firms_data.csv nahi mila.\n\n"
        f"Expected location:\n{DATA_PATH}"
    )

    st.stop()

except Exception as e:

    st.error(
        f"❌ FIRMS data load karte time error aaya:\n{e}"
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

model_available = False
fire_model = None

if MODEL_PATH.exists():

    try:

        # Saved Random Forest model load karo.
        fire_model = load_fire_model()

        model_available = True

    except Exception as e:

        st.sidebar.warning(
            f"⚠️ ML model load nahi hua: {e}"
        )

else:

    st.sidebar.warning(
        "⚠️ outputs/fire_model.pkl nahi mila."
    )


# ============================================================
# DATA PREPARATION
# ============================================================

# Date column ko datetime format me convert karo.
if "ACQ_DATE" in fires.columns:

    fires["ACQ_DATE"] = pd.to_datetime(
        fires["ACQ_DATE"],
        errors="coerce"
    )

elif "acq_date" in fires.columns:

    fires["acq_date"] = pd.to_datetime(
        fires["acq_date"],
        errors="coerce"
    )


# ============================================================
# TITLE
# ============================================================

st.title("🔥 Thermal Source Detection Dashboard")

st.markdown(
    """
    **SIH26162 — NASA FIRMS + AI/ML + GIS based Thermal Anomaly Classification**

    This dashboard uses NASA FIRMS satellite thermal observations,
    persistence analysis and machine learning to classify detected
    thermal sources.
    """
)


# ============================================================
# SIDEBAR — CLASSIFICATION MODE
# ============================================================

st.sidebar.header("🤖 Classification Mode")

classification_mode = st.sidebar.radio(
    "Select Classification Method",
    [
        "🗺️ GIS / Rule-Based Dashboard",
        "🤖 Random Forest ML Prediction"
    ]
)


# ============================================================
# SIDEBAR — DATA FILTERS
# ============================================================

st.sidebar.header("🔎 Filters")


# ------------------------------------------------------------
# FIRE CLASS FILTER
# ------------------------------------------------------------

if "FIRE_CLASS" in fires.columns:

    available_classes = sorted(
        fires["FIRE_CLASS"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

else:

    available_classes = []


selected_classes = st.sidebar.multiselect(
    "🔥 Thermal Source Type",
    available_classes,
    default=available_classes
)


# ------------------------------------------------------------
# PERSISTENCE FILTER
# ------------------------------------------------------------

if "PERSISTENCE_DAYS" in fires.columns:

    max_persistence = int(
        pd.to_numeric(
            fires["PERSISTENCE_DAYS"],
            errors="coerce"
        )
        .fillna(0)
        .max()
    )

else:

    max_persistence = 0


min_persistence = st.sidebar.number_input(
    "Minimum Persistence Days",
    min_value=0,
    max_value=max_persistence if max_persistence > 0 else 1,
    value=0,
    step=1
)


# ------------------------------------------------------------
# DATE FILTER
# ------------------------------------------------------------

date_column = None

if "ACQ_DATE" in fires.columns:
    date_column = "ACQ_DATE"

elif "acq_date" in fires.columns:
    date_column = "acq_date"


if date_column is not None:

    valid_dates = fires[date_column].dropna()

    if len(valid_dates) > 0:

        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()

        selected_dates = st.sidebar.date_input(
            "📅 Acquisition Date",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

    else:

        selected_dates = None

else:

    selected_dates = None


# ============================================================
# FILTER DATA
# ============================================================

filtered = fires.copy()


# ------------------------------------------------------------
# FIRE CLASS FILTER
# ------------------------------------------------------------

if "FIRE_CLASS" in filtered.columns:

    if len(selected_classes) > 0:

        filtered = filtered[
            filtered["FIRE_CLASS"]
            .astype(str)
            .isin(selected_classes)
        ].copy()

    else:

        filtered = filtered.iloc[0:0].copy()


# ------------------------------------------------------------
# PERSISTENCE FILTER
# ------------------------------------------------------------

if "PERSISTENCE_DAYS" in filtered.columns:

    persistence_numeric = pd.to_numeric(
        filtered["PERSISTENCE_DAYS"],
        errors="coerce"
    ).fillna(0)

    filtered = filtered[
        persistence_numeric >= min_persistence
    ].copy()


# ------------------------------------------------------------
# DATE FILTER
# ------------------------------------------------------------

if (
    selected_dates is not None
    and date_column is not None
):

    # Agar user ne single date select ki ho.
    if isinstance(selected_dates, tuple):

        start_date = pd.to_datetime(
            selected_dates[0]
        )

        end_date = pd.to_datetime(
            selected_dates[-1]
        )

    else:

        start_date = pd.to_datetime(
            selected_dates
        )

        end_date = start_date

    filtered = filtered[
        (
            filtered[date_column] >= start_date
        )
        &
        (
            filtered[date_column] <= end_date
            + pd.Timedelta(days=1)
            - pd.Timedelta(seconds=1)
        )
    ].copy()


# ============================================================
# GIS / RULE-BASED DASHBOARD
# ============================================================

if classification_mode == "🗺️ GIS / Rule-Based Dashboard":

    st.header("📌 NASA FIRMS Thermal Source Classification")

    st.caption(
        "NASA FIRMS satellite detections with persistence "
        "analysis and prototype fire classification."
    )


    # ========================================================
    # METRICS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)


    # --------------------------------------------------------
    # TOTAL DETECTIONS
    # --------------------------------------------------------

    with col1:

        st.metric(
            "🔥 Total Detections",
            len(filtered)
        )


    # --------------------------------------------------------
    # INDUSTRIAL SOURCES
    # --------------------------------------------------------

    with col2:

        if "FIRE_CLASS" in filtered.columns:

            industrial_count = filtered[
                filtered["FIRE_CLASS"]
                .astype(str)
                .str.contains(
                    "Industrial",
                    case=False,
                    na=False
                )
            ].shape[0]

        else:

            industrial_count = 0

        st.metric(
            "🏭 Industrial Sources",
            industrial_count
        )


    # --------------------------------------------------------
    # PERSISTENT SOURCES
    # --------------------------------------------------------

    with col3:

        if "PERSISTENCE_DAYS" in filtered.columns:

            persistent_count = filtered[
                pd.to_numeric(
                    filtered["PERSISTENCE_DAYS"],
                    errors="coerce"
                ).fillna(0) >= 3
            ].shape[0]

        else:

            persistent_count = 0

        st.metric(
            "🔁 Persistent Sources",
            persistent_count
        )


    # --------------------------------------------------------
    # AVERAGE FRP
    # --------------------------------------------------------

    with col4:

        if "FRP" in filtered.columns:

            avg_frp = pd.to_numeric(
                filtered["FRP"],
                errors="coerce"
            ).mean()

        elif "frp" in filtered.columns:

            avg_frp = pd.to_numeric(
                filtered["frp"],
                errors="coerce"
            ).mean()

        else:

            avg_frp = 0

        if pd.isna(avg_frp):
            avg_frp = 0

        st.metric(
            "⚡ Average FRP",
            f"{avg_frp:.2f}"
        )


    # ========================================================
    # CLASSIFICATION COUNTS
    # ========================================================

    st.subheader("📊 Thermal Source Classification")


    if (
        len(filtered) > 0
        and "FIRE_CLASS" in filtered.columns
    ):

        classification_counts = (
            filtered["FIRE_CLASS"]
            .astype(str)
            .value_counts()
        )

        st.bar_chart(
            classification_counts
        )

    else:

        st.info(
            "No classification data available."
        )


    # ========================================================
    # GIS MAP
    # ========================================================

    st.subheader("🗺️ Interactive Thermal Anomaly Map")


    # --------------------------------------------------------
    # CHECK COORDINATES
    # --------------------------------------------------------

    latitude_column = None
    longitude_column = None


    if "LATITUDE" in filtered.columns:
        latitude_column = "LATITUDE"

    elif "latitude" in filtered.columns:
        latitude_column = "latitude"


    if "LONGITUDE" in filtered.columns:
        longitude_column = "LONGITUDE"

    elif "longitude" in filtered.columns:
        longitude_column = "longitude"


    if (
        len(filtered) > 0
        and latitude_column is not None
        and longitude_column is not None
    ):

        # Coordinates ko numeric banao.
        filtered[latitude_column] = pd.to_numeric(
            filtered[latitude_column],
            errors="coerce"
        )

        filtered[longitude_column] = pd.to_numeric(
            filtered[longitude_column],
            errors="coerce"
        )

        # Invalid coordinates remove karo.
        map_data = filtered.dropna(
            subset=[
                latitude_column,
                longitude_column
            ]
        ).copy()


        if len(map_data) > 0:

            # ------------------------------------------------
            # MAP CENTER
            # ------------------------------------------------

            center_lat = map_data[
                latitude_column
            ].mean()

            center_lon = map_data[
                longitude_column
            ].mean()


            # ------------------------------------------------
            # CREATE FOLIUM MAP
            # ------------------------------------------------

            m = folium.Map(
                location=[
                    center_lat,
                    center_lon
                ],
                zoom_start=5,
                tiles="OpenStreetMap",
                control_scale=True
            )


            # =================================================
            # MARKER COLORS
            # =================================================

            marker_colors = {

                "Industrial Fire":
                    "red",

                "Agricultural Fire":
                    "orange",

                "Forest / Wildlife Fire":
                    "green",

                "Other Thermal Anomaly":
                    "blue"

            }


            # =================================================
            # ADD FIRE MARKERS
            # =================================================

            for _, row in map_data.iterrows():

                # ---------------------------------------------
                # GET FIRE CLASS
                # ---------------------------------------------

                fire_class = str(
                    row.get(
                        "FIRE_CLASS",
                        "Other Thermal Anomaly"
                    )
                )


                # ---------------------------------------------
                # SELECT COLOR
                # ---------------------------------------------

                marker_color = marker_colors.get(
                    fire_class,
                    "blue"
                )


                # ---------------------------------------------
                # GET FIRMS VALUES
                # ---------------------------------------------

                latitude = row[
                    latitude_column
                ]

                longitude = row[
                    longitude_column
                ]

                brightness = get_value(
                    row,
                    "BRIGHTNESS",
                    "brightness",
                    "N/A"
                )

                frp = get_value(
                    row,
                    "FRP",
                    "frp",
                    "N/A"
                )

                confidence = get_value(
                    row,
                    "CONFIDENCE",
                    "confidence",
                    "N/A"
                )

                satellite = get_value(
                    row,
                    "SATELLITE",
                    "satellite",
                    "N/A"
                )

                daynight = get_value(
                    row,
                    "DAYNIGHT",
                    "daynight",
                    "N/A"
                )

                persistence = row.get(
                    "PERSISTENCE_DAYS",
                    "N/A"
                )

                temp_diff = row.get(
                    "TEMP_DIFF",
                    "N/A"
                )


                # ---------------------------------------------
                # FORMAT ACQUISITION DATE
                # ---------------------------------------------

                acq_date = row.get(
                    "ACQ_DATE",
                    "N/A"
                )

                if pd.notna(acq_date):

                    try:

                        acq_date = pd.to_datetime(
                            acq_date
                        ).strftime(
                            "%Y-%m-%d"
                        )

                    except Exception:

                        acq_date = str(
                            acq_date
                        )


                # ---------------------------------------------
                # POPUP
                # ---------------------------------------------

                popup_text = f"""
                <div style="
                    font-size:14px;
                    min-width:260px;
                ">

                    <h4 style="margin-bottom:8px;">
                        🔥 Thermal Source
                    </h4>

                    <b>Predicted / Classified Type:</b><br>
                    {fire_class}

                    <hr>

                    <b>Latitude:</b>
                    {latitude:.5f}<br>

                    <b>Longitude:</b>
                    {longitude:.5f}<br>

                    <b>Brightness:</b>
                    {brightness}<br>

                    <b>FRP:</b>
                    {frp}<br>

                    <b>Confidence:</b>
                    {confidence}<br>

                    <b>Persistence:</b>
                    {persistence} days<br>

                    <b>Temperature Difference:</b>
                    {temp_diff}<br>

                    <b>Satellite:</b>
                    {satellite}<br>

                    <b>Day / Night:</b>
                    {daynight}<br>

                    <b>Acquisition Date:</b>
                    {acq_date}

                </div>
                """


                # ---------------------------------------------
                # LARGE FIRE CIRCLE
                # ---------------------------------------------

                folium.CircleMarker(

                    location=[
                        latitude,
                        longitude
                    ],

                    radius=9,

                    color=marker_color,

                    fill=True,

                    fill_color=marker_color,

                    fill_opacity=0.85,

                    weight=3,

                    popup=folium.Popup(
                        popup_text,
                        max_width=400
                    ),

                    tooltip=folium.Tooltip(
                        f"🔥 {fire_class}",
                        sticky=True
                    )

                ).add_to(m)


                # ---------------------------------------------
                # OUTER RING
                # ---------------------------------------------

                folium.CircleMarker(

                    location=[
                        latitude,
                        longitude
                    ],

                    radius=15,

                    color=marker_color,

                    fill=False,

                    opacity=0.35,

                    weight=2

                ).add_to(m)


            # =================================================
            # MAP LEGEND
            # =================================================

            legend_html = """
            <div style="
                position: fixed;
                bottom: 30px;
                left: 30px;
                width: 260px;
                z-index: 9999;
                background-color: white;
                border: 2px solid grey;
                padding: 12px;
                font-size: 13px;
                border-radius: 8px;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
            ">

                <b>🔥 Thermal Source Legend</b>

                <br><br>

                <span style="color:red;">
                    ●
                </span>
                Industrial Fire

                <br>

                <span style="color:orange;">
                    ●
                </span>
                Agricultural Fire

                <br>

                <span style="color:green;">
                    ●
                </span>
                Forest / Wildlife Fire

                <br>

                <span style="color:blue;">
                    ●
                </span>
                Other Thermal Anomaly

            </div>
            """


            m.get_root().html.add_child(
                folium.Element(
                    legend_html
                )
            )


            # ------------------------------------------------
            # MAP DISPLAY
            # ------------------------------------------------

            map_result = st_folium(

                m,

                width=None,

                height=700,

                returned_objects=[
                    "last_object_clicked"
                ]

            )


            # =================================================
            # CLICKED MAP POINT PREDICTION
            # =================================================

            clicked = map_result.get(
                "last_object_clicked"
            )


            if clicked is not None:

                clicked_lat = clicked.get(
                    "lat"
                )

                clicked_lon = clicked.get(
                    "lng"
                )


                if (
                    clicked_lat is not None
                    and clicked_lon is not None
                ):

                    st.markdown("---")

                    st.subheader(
                        "🎯 Selected Thermal Detection"
                    )


                    # -----------------------------------------
                    # FIND NEAREST FIRMS RECORD
                    # -----------------------------------------

                    distance = (
                        (
                            map_data[latitude_column]
                            - clicked_lat
                        ) ** 2
                        +
                        (
                            map_data[longitude_column]
                            - clicked_lon
                        ) ** 2
                    )


                    nearest_index = distance.idxmin()

                    selected_fire = (
                        map_data
                        .loc[nearest_index]
                        .copy()
                    )


                    # -----------------------------------------
                    # DISPLAY SELECTED LOCATION
                    # -----------------------------------------

                    result_col1, result_col2, result_col3 = st.columns(3)


                    with result_col1:

                        st.metric(
                            "Latitude",
                            f"{float(selected_fire[latitude_column]):.5f}"
                        )


                    with result_col2:

                        st.metric(
                            "Longitude",
                            f"{float(selected_fire[longitude_column]):.5f}"
                        )


                    with result_col3:

                        selected_frp = get_value(
                            selected_fire,
                            "FRP",
                            "frp",
                            0
                        )

                        st.metric(
                            "FRP",
                            f"{float(selected_frp):.2f}"
                        )


                    # =================================================
                    # RUN ML MODEL ON CLICKED POINT
                    # =================================================

                    if model_available:

                        try:

                            # -----------------------------------------
                            # GET MODEL FEATURES
                            # -----------------------------------------

                            selected_brightness = float(
                                get_value(
                                    selected_fire,
                                    "BRIGHTNESS",
                                    "brightness",
                                    0
                                )
                            )


                            selected_background = float(
                                get_value(
                                    selected_fire,
                                    "BACKGROUND_TEMP",
                                    "background_temp",
                                    get_value(
                                        selected_fire,
                                        "BRIGHT_T31",
                                        "bright_t31",
                                        0
                                    )
                                )
                            )


                            selected_frp = float(
                                get_value(
                                    selected_fire,
                                    "FRP",
                                    "frp",
                                    0
                                )
                            )


                            selected_temp_diff = (
                                selected_brightness
                                - selected_background
                            )


                            selected_frp_log = np.log1p(
                                selected_frp
                            )


                            selected_persistence = float(
                                selected_fire.get(
                                    "PERSISTENCE_DAYS",
                                    0
                                )
                            )


                            selected_daynight = str(
                                get_value(
                                    selected_fire,
                                    "DAYNIGHT",
                                    "daynight",
                                    "D"
                                )
                            ).upper()


                            selected_is_night = (
                                1
                                if selected_daynight == "N"
                                else 0
                            )


                            selected_confidence = str(
                                get_value(
                                    selected_fire,
                                    "CONFIDENCE",
                                    "confidence",
                                    "n"
                                )
                            ).lower()


                            confidence_map = {
                                "l": 0,
                                "n": 1,
                                "h": 2
                            }


                            selected_confidence_num = (
                                confidence_map.get(
                                    selected_confidence,
                                    1
                                )
                            )


                            # -----------------------------------------
                            # CREATE MODEL INPUT
                            # -----------------------------------------

                            model_input = pd.DataFrame({

                                "BRIGHTNESS": [
                                    selected_brightness
                                ],

                                "BACKGROUND_TEMP": [
                                    selected_background
                                ],

                                "FRP": [
                                    selected_frp
                                ],

                                "TEMP_DIFF": [
                                    selected_temp_diff
                                ],

                                "FRP_LOG": [
                                    selected_frp_log
                                ],

                                "PERSISTENCE_DAYS": [
                                    selected_persistence
                                ],

                                "IS_NIGHT": [
                                    selected_is_night
                                ],

                                "CONFIDENCE_NUM": [
                                    selected_confidence_num
                                ]

                            })


                            # -----------------------------------------
                            # PREDICT FIRE CLASS
                            # -----------------------------------------

                            predicted_class = (
                                fire_model
                                .predict(
                                    model_input
                                )[0]
                            )


                            # -----------------------------------------
                            # PREDICTION PROBABILITY
                            # -----------------------------------------

                            probabilities = (
                                fire_model
                                .predict_proba(
                                    model_input
                                )[0]
                            )


                            model_classes = (
                                fire_model.classes_
                            )


                            probability_df = pd.DataFrame({

                                "Fire Class":
                                    model_classes,

                                "Probability (%)":
                                    probabilities * 100

                            })


                            probability_df[
                                "Probability (%)"
                            ] = probability_df[
                                "Probability (%)"
                            ].round(2)


                            prediction_confidence = (
                                float(
                                    probabilities.max()
                                ) * 100
                            )


                            # =============================================
                            # PREDICTION RESULT
                            # =============================================

                            st.subheader(
                                "🔥 ML Prediction"
                            )


                            prediction_col1, prediction_col2 = st.columns(2)


                            with prediction_col1:

                                st.success(
                                    f"🔥 **Predicted Fire Class:** "
                                    f"{predicted_class}"
                                )


                            with prediction_col2:

                                st.metric(
                                    "🎯 Model Confidence",
                                    f"{prediction_confidence:.2f}%"
                                )


                            # -----------------------------------------
                            # PROBABILITY TABLE
                            # -----------------------------------------

                            st.subheader(
                                "📊 Class Probabilities"
                            )


                            st.dataframe(
                                probability_df,
                                use_container_width=True,
                                hide_index=True
                            )


                            # -----------------------------------------
                            # SELECTED POINT DOWNLOAD
                            # -----------------------------------------

                            selected_output = pd.DataFrame([
                                {

                                    "LATITUDE":
                                        selected_fire[
                                            latitude_column
                                        ],

                                    "LONGITUDE":
                                        selected_fire[
                                            longitude_column
                                        ],

                                    "BRIGHTNESS":
                                        selected_brightness,

                                    "BACKGROUND_TEMP":
                                        selected_background,

                                    "FRP":
                                        selected_frp,

                                    "TEMP_DIFF":
                                        selected_temp_diff,

                                    "FRP_LOG":
                                        selected_frp_log,

                                    "PERSISTENCE_DAYS":
                                        selected_persistence,

                                    "DAYNIGHT":
                                        selected_daynight,

                                    "CONFIDENCE":
                                        selected_confidence,

                                    "PREDICTED_FIRE_CLASS":
                                        predicted_class,

                                    "MODEL_CONFIDENCE_PERCENT":
                                        round(
                                            prediction_confidence,
                                            2
                                        )

                                }
                            ])


                            st.download_button(

                                label="⬇️ Download Selected Fire Prediction",

                                data=selected_output.to_csv(
                                    index=False
                                ).encode("utf-8"),

                                file_name=(
                                    "selected_fire_prediction.csv"
                                ),

                                mime="text/csv"

                            )


                        except Exception as e:

                            st.error(
                                "❌ Selected point prediction error: "
                                f"{e}"
                            )


            # =================================================
            # MAP DATA INFO
            # =================================================

            st.caption(
                f"Showing {len(map_data):,} thermal detections "
                "on the map."
            )


        else:

            st.warning(
                "⚠️ Valid latitude/longitude data nahi mila."
            )

    else:

        st.warning(
            "⚠️ No thermal detections match the selected filters."
        )


    # ========================================================
    # DATA TABLE
    # ========================================================

    st.subheader("📋 Detection Details")


    preferred_columns = [

        "LATITUDE",
        "LONGITUDE",

        "BRIGHTNESS",
        "BRIGHT_T31",
        "BACKGROUND_TEMP",

        "FRP",
        "TEMP_DIFF",

        "ACQ_DATE",
        "ACQ_TIME",

        "SATELLITE",
        "CONFIDENCE",
        "DAYNIGHT",

        "PERSISTENCE_DAYS",

        "FIRE_CLASS"

    ]


    # Sirf wahi columns display karo jo available hain.
    available_columns = [

        column

        for column in preferred_columns

        if column in filtered.columns

    ]


    if len(filtered) > 0 and len(available_columns) > 0:

        st.dataframe(

            filtered[
                available_columns
            ],

            use_container_width=True,

            hide_index=True

        )

    else:

        st.info(
            "No detection details available."
        )


    # ========================================================
    # DOWNLOAD FILTERED DATA
    # ========================================================

    st.subheader(
        "⬇️ Download Classified FIRMS Data"
    )


    csv_data = filtered.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(

        label="⬇️ Download Filtered CSV",

        data=csv_data,

        file_name="classified_firms_data.csv",

        mime="text/csv"

    )


# ============================================================
# RANDOM FOREST ML DASHBOARD
# ============================================================

else:

    st.header(
        "🤖 Random Forest ML Classification"
    )


    st.markdown(
        """
        Enter thermal anomaly features below.

        The saved Random Forest model will classify
        the thermal source.
        """
    )


    # ========================================================
    # MODEL CHECK
    # ========================================================

    if not model_available:

        st.error(
            """
            ❌ Random Forest model available nahi hai.

            Expected file:

            outputs/fire_model.pkl
            """
        )

        st.stop()


    # ========================================================
    # MODEL STATUS
    # ========================================================

    st.success(
        "✅ Trained Random Forest model loaded successfully."
    )


    st.info(
        "Saved `.pkl` model directly use ho raha hai. "
        "Prediction ke time model dobara train nahi hota."
    )


    # ========================================================
    # MODEL CLASSES
    # ========================================================

    st.subheader(
        "🎯 Supported Thermal Sources"
    )


    for index, cls in enumerate(
        fire_model.classes_
    ):

        st.write(
            f"**{index} — {cls}**"
        )


    # ========================================================
    # INPUT SECTION
    # ========================================================

    st.subheader(
        "🔥 Thermal Event Input"
    )


    col1, col2 = st.columns(2)


    # ========================================================
    # LEFT COLUMN
    # ========================================================

    with col1:

        brightness = st.number_input(

            "Brightness",

            min_value=0.0,

            value=330.0,

            step=1.0,

            help=(
                "NASA FIRMS thermal brightness "
                "temperature."
            )

        )


        background_temp = st.number_input(

            "Background Temperature",

            min_value=0.0,

            value=290.0,

            step=1.0,

            help=(
                "Background brightness temperature "
                "from FIRMS BRIGHT_T31."
            )

        )


        frp = st.number_input(

            "FRP",

            min_value=0.0,

            value=8.0,

            step=0.1,

            help=(
                "Fire Radiative Power."
            )

        )


        persistence_days = st.number_input(

            "Persistence Days",

            min_value=0,

            value=2,

            step=1,

            help=(
                "Number of unique observation days "
                "at approximately the same location."
            )

        )


    # ========================================================
    # RIGHT COLUMN
    # ========================================================

    with col2:

        day_night = st.selectbox(

            "Day / Night",

            [
                "Day",
                "Night"
            ]

        )


        firms_confidence = st.selectbox(

            "FIRMS Confidence",

            [
                "Low",
                "Nominal",
                "High"
            ],

            index=1

        )


        # -----------------------------------------------
        # DERIVED FEATURES PREVIEW
        # -----------------------------------------------

        temp_diff_preview = (
            brightness
            - background_temp
        )


        frp_log_preview = np.log1p(
            frp
        )


        st.metric(
            "Temperature Difference",
            f"{temp_diff_preview:.2f}"
        )


        st.metric(
            "Log(FRP + 1)",
            f"{frp_log_preview:.4f}"
        )


    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📋 Input Summary"
    )


    input_summary = pd.DataFrame({

        "Feature": [

            "Brightness",

            "Background Temperature",

            "FRP",

            "Temperature Difference",

            "Log(FRP + 1)",

            "Persistence Days",

            "Day / Night",

            "FIRMS Confidence"

        ],

        "Value": [

            brightness,

            background_temp,

            frp,

            temp_diff_preview,

            frp_log_preview,

            persistence_days,

            day_night,

            firms_confidence

        ]

    })


    st.dataframe(

        input_summary,

        use_container_width=True,

        hide_index=True

    )


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    st.markdown("---")


    predict_button = st.button(

        "🔍 Predict Thermal Source",

        type="primary",

        use_container_width=True

    )


    # ========================================================
    # PREDICTION
    # ========================================================

    if predict_button:

        try:

            # ---------------------------------------------
            # ENCODE DAY / NIGHT
            # ---------------------------------------------

            is_night = (

                1

                if day_night == "Night"

                else 0

            )


            # ---------------------------------------------
            # ENCODE FIRMS CONFIDENCE
            # ---------------------------------------------

            confidence_map = {

                "Low": 0,

                "Nominal": 1,

                "High": 2

            }


            confidence_num = confidence_map[
                firms_confidence
            ]


            # ---------------------------------------------
            # DERIVED FEATURES
            # ---------------------------------------------

            temp_diff = (
                brightness
                - background_temp
            )


            frp_log = np.log1p(
                frp
            )


            # ---------------------------------------------
            # CREATE MODEL INPUT
            # ---------------------------------------------

            input_data = pd.DataFrame({

                "BRIGHTNESS": [
                    brightness
                ],

                "BACKGROUND_TEMP": [
                    background_temp
                ],

                "FRP": [
                    frp
                ],

                "TEMP_DIFF": [
                    temp_diff
                ],

                "FRP_LOG": [
                    frp_log
                ],

                "PERSISTENCE_DAYS": [
                    persistence_days
                ],

                "IS_NIGHT": [
                    is_night
                ],

                "CONFIDENCE_NUM": [
                    confidence_num
                ]

            })


            # ---------------------------------------------
            # PREDICT
            # ---------------------------------------------

            predicted_class = (
                fire_model
                .predict(
                    input_data
                )[0]
            )


            # ---------------------------------------------
            # PROBABILITIES
            # ---------------------------------------------

            probabilities = (
                fire_model
                .predict_proba(
                    input_data
                )[0]
            )


            model_classes = (
                fire_model.classes_
            )


            # ---------------------------------------------
            # CONFIDENCE
            # ---------------------------------------------

            prediction_confidence = (
                float(
                    probabilities.max()
                ) * 100
            )


            # =================================================
            # RESULT
            # =================================================

            st.markdown("---")

            st.subheader(
                "🎯 Prediction Result"
            )


            result_col1, result_col2 = st.columns(2)


            with result_col1:

                st.success(
                    f"🔥 Thermal Source Detected: "
                    f"**{predicted_class}**"
                )


            with result_col2:

                st.metric(
                    "Prediction Confidence",
                    f"{prediction_confidence:.2f}%"
                )


            # =================================================
            # CLASS PROBABILITIES
            # =================================================

            st.subheader(
                "📊 Thermal Source Probabilities"
            )


            probability_data = pd.DataFrame({

                "Thermal Source":
                    model_classes,

                "Probability (%)":
                    probabilities * 100

            })


            probability_data = (

                probability_data

                .sort_values(

                    "Probability (%)",

                    ascending=False

                )

                .reset_index(
                    drop=True
                )

            )


            probability_data[
                "Probability (%)"
            ] = (

                probability_data[
                    "Probability (%)"
                ]

                .round(2)

            )


            st.dataframe(

                probability_data,

                use_container_width=True,

                hide_index=True

            )


            # =================================================
            # PROBABILITY CHART
            # =================================================

            st.subheader(
                "📈 Prediction Probability"
            )


            st.bar_chart(

                probability_data.set_index(
                    "Thermal Source"
                )[
                    "Probability (%)"
                ]

            )


            # =================================================
            # FINAL RESULT TABLE
            # =================================================

            st.markdown("---")

            st.subheader(
                "📋 Final ML Classification"
            )


            final_result = pd.DataFrame({

                "Parameter": [

                    "Brightness",

                    "Background Temperature",

                    "FRP",

                    "Temperature Difference",

                    "Persistence Days",

                    "Day / Night",

                    "FIRMS Confidence",

                    "🔥 Thermal Source",

                    "🎯 Prediction Confidence"

                ],

                "Value": [

                    brightness,

                    background_temp,

                    frp,

                    temp_diff,

                    persistence_days,

                    day_night,

                    firms_confidence,

                    predicted_class,

                    f"{prediction_confidence:.2f}%"

                ]

            })


            st.dataframe(

                final_result,

                use_container_width=True,

                hide_index=True

            )


            # =================================================
            # DOWNLOAD MANUAL PREDICTION
            # =================================================

            prediction_output = pd.DataFrame([{

                "BRIGHTNESS":
                    brightness,

                "BACKGROUND_TEMP":
                    background_temp,

                "FRP":
                    frp,

                "TEMP_DIFF":
                    temp_diff,

                "FRP_LOG":
                    frp_log,

                "PERSISTENCE_DAYS":
                    persistence_days,

                "IS_NIGHT":
                    is_night,

                "CONFIDENCE_NUM":
                    confidence_num,

                "DAY_NIGHT":
                    day_night,

                "FIRMS_CONFIDENCE":
                    firms_confidence,

                "PREDICTED_FIRE_CLASS":
                    predicted_class,

                "MODEL_CONFIDENCE_PERCENT":
                    round(
                        prediction_confidence,
                        2
                    )

            }])


            st.download_button(

                label=(
                    "⬇️ Download ML Prediction CSV"
                ),

                data=prediction_output.to_csv(
                    index=False
                ).encode("utf-8"),

                file_name=(
                    "manual_ml_fire_prediction.csv"
                ),

                mime="text/csv"

            )


            # =================================================
            # MODEL INFORMATION
            # =================================================

            st.markdown("---")

            st.subheader(
                "ℹ️ Random Forest Classification"
            )


            st.write(
                """
                The Random Forest model uses the same eight
                features used during training:

                **Brightness, Background Temperature, FRP,
                Temperature Difference, Log(FRP + 1),
                Persistence Days, Day/Night and FIRMS Confidence.**

                The saved `.pkl` model is loaded directly and
                is not retrained during prediction.
                """
            )


        except Exception as e:

            st.error(
                f"❌ Prediction error: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "SIH26162 | NASA FIRMS + AI/ML + GIS Thermal "
    "Anomaly Detection Prototype"
)