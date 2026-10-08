import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DiaPredict",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# CUSTOM STYLE
# ============================================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background: #f4f7fb;
}

/* Main content width */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #172554;
}

section[data-testid="stSidebar"] * {
    color: #ffffff;
}

section[data-testid="stSidebar"] .stRadio label {
    padding: 8px 10px;
    border-radius: 8px;
}

/* Titles */
h1 {
    color: #172554;
    font-weight: 800;
}

h2 {
    color: #1e3a5f;
    font-weight: 700;
}

h3 {
    color: #294766;
    font-weight: 700;
}

/* Normal text */
p {
    color: #5f6f82;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e3eaf2;
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0 4px 15px rgba(30, 50, 80, 0.06);
}

/* Metric label */
div[data-testid="stMetricLabel"] {
    color: #718096;
}

/* Metric value */
div[data-testid="stMetricValue"] {
    color: #172554;
    font-weight: 800;
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    border: none;
    background: #0f766e !important;
    color: white !important;
    font-weight: 600;
    padding: 10px 20px;
}

.stButton > button p {
    color: white !important;
}

.stButton > button:hover {
    background: #115e59 !important;
    color: white !important;
}

.stButton > button:hover p {
    color: white !important;
}

/* Input boxes */
div[data-baseweb="input"] {
    border-radius: 8px;
}

div[data-baseweb="select"] {
    border-radius: 8px;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}

/* Info boxes */
div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Divider */
hr {
    border-color: #dce4ee;
}

</style>
""", unsafe_allow_html=True)



# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("diabetes.csv")


# ============================================================
# LOAD SCALER
# ============================================================

scaler = joblib.load("models/scaler.pkl")


# ============================================================
# MODEL FILES
# ============================================================

model_files = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "Random Forest": "models/random_forest.pkl",
    "KNN": "models/knn.pkl",
    "SVM": "models/svm.pkl",
    "Naive Bayes": "models/naive_bayes.pkl",
    "AdaBoost": "models/adaboost.pkl",
    "Gradient Boosting": "models/gradient_boosting.pkl",
    "XGBoost": "models/xgboost.pkl"
}


# ============================================================
# LOAD MODELS
# ============================================================

models = {}

for name, path in model_files.items():

    if os.path.exists(path):
        models[name] = joblib.load(path)


# ============================================================
# LOAD BEST MODEL
# ============================================================

with open("models/best_model.txt", "r") as file:
    best_model_name = file.read().strip()


# ============================================================
# LOAD MODEL RESULTS
# ============================================================

results_df = pd.read_csv(
    "models/model_results.csv"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🩺 DiaPredict")

st.sidebar.caption(
    "Diabetes Prediction & Risk Analysis Platform"
)

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Models",
        "Dataset",
        "Predict",
        "History",
        "About"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    f"{best_model_name} • ML Prediction"
)


# ============================================================
# HOME PAGE
# ============================================================

if page == "Home":

    st.title(
        "🩺 Diabetes Prediction and Risk Analysis"
    )

    st.subheader(
        "Machine Learning Based Diabetes Prediction System"
    )

    st.write(
        """
        This application uses machine learning models to predict
        the likelihood of diabetes based on health-related
        information.
        """
    )

    st.divider()

    # Metrics

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Dataset Rows",
            df.shape[0]
        )

    with col2:
        st.metric(
            "Features",
            8
        )

    with col3:
        st.metric(
            "ML Models",
            len(models)
        )

    with col4:
        st.metric(
            "Best Model",
            best_model_name
        )

    st.divider()

    st.subheader("Model Performance")

    best_row = results_df[
        results_df["Model"] == best_model_name
    ].iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            f"{best_row['Accuracy'] * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Precision",
            f"{best_row['Precision'] * 100:.2f}%"
        )

    with col3:
        st.metric(
            "Recall",
            f"{best_row['Recall'] * 100:.2f}%"
        )

    with col4:
        st.metric(
            "ROC-AUC",
            f"{best_row['ROC-AUC']:.2f}"
        )

    st.info(
        "Go to the Predict page to enter patient information "
        "and generate a prediction."
    )


# ============================================================
# MODELS PAGE
# ============================================================

elif page == "Models":

    st.title("🤖 Machine Learning Models")

    st.write(
        "Performance comparison of all trained classification models."
    )

    display_df = results_df.copy()

    display_df["Accuracy"] = (
        display_df["Accuracy"] * 100
    ).round(2)

    display_df["Precision"] = (
        display_df["Precision"] * 100
    ).round(2)

    display_df["Recall"] = (
        display_df["Recall"] * 100
    ).round(2)

    display_df["F1 Score"] = (
        display_df["F1 Score"] * 100
    ).round(2)

    display_df["ROC-AUC"] = (
        display_df["ROC-AUC"]
    ).round(3)

    st.dataframe(
        display_df,
        use_container_width=True
    )

    st.success(
        f"Best Model: {best_model_name}"
    )

    st.subheader("Accuracy Comparison")

    chart_data = results_df[
        ["Model", "Accuracy"]
    ].set_index("Model")

    st.bar_chart(chart_data)


# ============================================================
# DATASET PAGE
# ============================================================

elif page == "Dataset":

    st.title("📊 Dataset Information")

    st.write(
        """
        The dataset contains health-related information
        used for diabetes prediction.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Rows",
            df.shape[0]
        )

    with col2:
        st.metric(
            "Columns",
            df.shape[1]
        )

    with col3:
        st.metric(
            "Input Features",
            df.shape[1] - 1
        )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.subheader("Feature Description")

    feature_info = pd.DataFrame({
        "Feature": [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
            "Outcome"
        ],

        "Description": [
            "Number of pregnancies",
            "Plasma glucose concentration",
            "Diastolic blood pressure",
            "Skin fold thickness",
            "Serum insulin level",
            "Body Mass Index",
            "Diabetes pedigree function",
            "Age in years",
            "0 = No Diabetes, 1 = Diabetes"
        ]
    })

    st.table(feature_info)


# ============================================================
# PREDICT PAGE
# ============================================================

elif page == "Predict":

    st.title("🔍 Diabetes Prediction")

    st.write(
        "Enter the health information below."
    )

    st.divider()

    # Model Selection

    model_options = [
        "Best Model (Recommended)"
    ] + list(models.keys())

    selected_model = st.selectbox(
        "Select Machine Learning Model",
        model_options
    )

    if selected_model == "Best Model (Recommended)":

        selected_model_name = best_model_name

    else:

        selected_model_name = selected_model

    model = models[selected_model_name]

    st.write(
        f"Selected Model: **{selected_model_name}**"
    )

    st.divider()

    # Input Fields

    col1, col2 = st.columns(2)

    with col1:

        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=1
        )

        glucose = st.number_input(
            "Glucose",
            min_value=0.0,
            max_value=300.0,
            value=120.0
        )

        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )

        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=0.0,
            max_value=100.0,
            value=20.0
        )

    with col2:

        insulin = st.number_input(
            "Insulin",
            min_value=0.0,
            max_value=900.0,
            value=80.0
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=70.0,
            value=25.0
        )

        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.47
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30
        )

    st.divider()

    # Prediction Button

    if st.button(
        "🔍 Predict Diabetes Risk",
        use_container_width=True
    ):

        input_data = pd.DataFrame({

            "Pregnancies": [pregnancies],

            "Glucose": [glucose],

            "BloodPressure": [blood_pressure],

            "SkinThickness": [skin_thickness],

            "Insulin": [insulin],

            "BMI": [bmi],

            "DiabetesPedigreeFunction": [
                diabetes_pedigree
            ],

            "Age": [age]
        })

        # Scaling

        if selected_model_name in [
            "Logistic Regression",
            "KNN",
            "SVM",
            "Naive Bayes"
        ]:

            input_scaled = scaler.transform(
                input_data
            )

            prediction = model.predict(
                input_scaled
            )[0]

            probability = model.predict_proba(
                input_scaled
            )[0][1]

        else:

            prediction = model.predict(
                input_data
            )[0]

            probability = model.predict_proba(
                input_data
            )[0][1]

        # Prediction text

        if prediction == 1:

            prediction_text = (
                "Higher likelihood of diabetes"
            )

        else:

            prediction_text = (
                "Lower likelihood of diabetes"
            )

        probability_percent = (
            probability * 100
        )

        # Risk

        if probability < 0.30:

            risk = "LOW"

        elif probability < 0.70:

            risk = "MODERATE"

        else:

            risk = "HIGH"

        st.divider()

        st.subheader(
            "🔍 Prediction Result"
        )

        if prediction == 1:

            st.error(
                prediction_text
            )

        else:

            st.success(
                prediction_text
            )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Probability",
                f"{probability_percent:.2f}%"
            )

        with col2:

            st.metric(
                "Risk Level",
                risk
            )

        with col3:

            st.metric(
                "Model",
                selected_model_name
            )

        if risk == "LOW":

            st.info(
                "Continue healthy lifestyle habits "
                "and routine health checkups."
            )

        elif risk == "MODERATE":

            st.warning(
                "The model indicates an increased "
                "likelihood. Consider discussing "
                "the result with a qualified "
                "healthcare professional."
            )

        else:

            st.warning(
                "The model indicates a higher likelihood. "
                "Please consult a qualified healthcare "
                "professional for further evaluation."
            )

        # Save History

        history_file = "prediction_history.csv"

        new_record = pd.DataFrame({

            "Date & Time": [
                datetime.now().strftime(
                    "%d-%m-%Y %I:%M %p"
                )
            ],

            "Model": [
                selected_model_name
            ],

            "Pregnancies": [
                pregnancies
            ],

            "Glucose": [
                glucose
            ],

            "BloodPressure": [
                blood_pressure
            ],

            "BMI": [
                bmi
            ],

            "Prediction": [
                prediction_text
            ],

            "Risk Level": [
                risk
            ],

            "Probability": [
                f"{probability_percent:.2f}%"
            ]
        })

        if os.path.exists(history_file):

            old_history = pd.read_csv(
                history_file
            )

            history = pd.concat(
                [
                    old_history,
                    new_record
                ],
                ignore_index=True
            )

        else:

            history = new_record

        history.to_csv(
            history_file,
            index=False
        )

        st.success(
            "✅ Prediction saved to History."
        )


# ============================================================
# HISTORY PAGE
# ============================================================

elif page == "History":

    st.title("📜 Prediction History")

    history_file = "prediction_history.csv"

    if os.path.exists(history_file):

        history_df = pd.read_csv(
            history_file
        )

        if len(history_df) > 0:

            st.dataframe(
                history_df,
                use_container_width=True
            )

            st.write(
                f"**Total Predictions:** "
                f"{len(history_df)}"
            )

            st.divider()

            if st.button(
                "🗑️ Clear History"
            ):

                os.remove(
                    history_file
                )

                st.success(
                    "Prediction history cleared."
                )

                st.rerun()

        else:

            st.info(
                "No prediction history available."
            )

    else:

        st.info(
            "No prediction history available yet. "
            "Make a prediction first."
        )


# ============================================================
# ABOUT PAGE
# ============================================================

# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "About":

    st.title("ℹ️ About the Project")

    st.write(
        "Diabetes Prediction and Risk Analysis Using Machine Learning"
    )

    st.divider()

    # ========================================================
    # TOP TWO BOXES
    # ========================================================

    col1, col2 = st.columns(2)

    # -----------------------------
    # PROJECT OVERVIEW
    # -----------------------------

    with col1:

        st.markdown("""
        <div style="
            background: white;
            padding: 25px;
            border-radius: 14px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 5px 18px rgba(20,50,90,0.06);
            min-height: 220px;
        ">

        <div style="
            font-size: 24px;
            margin-bottom: 10px;
        ">
        📋
        </div>

        <h3 style="
            color: #17395f;
            margin-bottom: 12px;
        ">
        Project Overview
        </h3>

        <p style="
            color: #68788b;
            line-height: 1.7;
            font-size: 14px;
        ">
        This project uses machine learning techniques to
        analyze health-related information and predict the
        likelihood of diabetes.
        </p>

        <p style="
            color: #68788b;
            line-height: 1.7;
            font-size: 14px;
        ">
        Multiple classification algorithms are trained and
        evaluated to identify the best-performing model for
        diabetes prediction.
        </p>

        </div>
        """, unsafe_allow_html=True)


    # -----------------------------
    # TECHNOLOGY STACK
    # -----------------------------

    with col2:

        st.markdown("""
        <div style="
            background: white;
            padding: 25px;
            border-radius: 14px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 5px 18px rgba(20,50,90,0.06);
            min-height: 220px;
        ">

        <div style="
            font-size: 24px;
            margin-bottom: 10px;
        ">
        🛠️
        </div>

        <h3 style="
            color: #17395f;
            margin-bottom: 15px;
        ">
        Technology Stack
        </h3>

        <p style="
            color: #68788b;
            line-height: 1.8;
            font-size: 14px;
        ">
        <b>Programming:</b> Python
        <br>
        <b>Data Analysis:</b> Pandas, NumPy
        <br>
        <b>Machine Learning:</b> Scikit-learn, XGBoost
        <br>
        <b>Visualization:</b> Matplotlib
        <br>
        <b>Web Application:</b> Streamlit
        <br>
        <b>Model Saving:</b> Joblib
        </p>

        </div>
        """, unsafe_allow_html=True)


    # ========================================================
    # PROJECT WORKFLOW
    # ========================================================

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("🔄 Project Workflow")

    st.write(
        "The project follows a complete machine learning workflow "
        "from dataset preparation to prediction."
    )

    st.markdown("<br>", unsafe_allow_html=True)


    # -----------------------------
    # STEP 1
    # -----------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("""
        <div style="
            background:white;
            padding:20px;
            border-radius:12px;
            border:1px solid #e2e8f0;
            min-height:170px;
        ">

        <div style="
            font-size:22px;
            margin-bottom:8px;
        ">
        📂
        </div>

        <h4 style="color:#17395f;">
        01. Data Collection
        </h4>

        <p style="
            color:#718096;
            font-size:13px;
            line-height:1.6;
        ">
        Load the diabetes dataset and understand
        its structure, features and target variable.
        </p>

        </div>
        """, unsafe_allow_html=True)


    # -----------------------------
    # STEP 2
    # -----------------------------

    with col2:

        st.markdown("""
        <div style="
            background:white;
            padding:20px;
            border-radius:12px;
            border:1px solid #e2e8f0;
            min-height:170px;
        ">

        <div style="
            font-size:22px;
            margin-bottom:8px;
        ">
        🧹
        </div>

        <h4 style="color:#17395f;">
        02. Data Preprocessing
        </h4>

        <p style="
            color:#718096;
            font-size:13px;
            line-height:1.6;
        ">
        Prepare the dataset for machine learning
        and apply the required preprocessing steps.
        </p>

        </div>
        """, unsafe_allow_html=True)


    # -----------------------------
    # STEP 3
    # -----------------------------

    with col3:

        st.markdown("""
        <div style="
            background:white;
            padding:20px;
            border-radius:12px;
            border:1px solid #e2e8f0;
            min-height:170px;
        ">

        <div style="
            font-size:22px;
            margin-bottom:8px;
        ">
        🎯
        </div>

        <h4 style="color:#17395f;">
        03. Feature & Target
        </h4>

        <p style="
            color:#718096;
            font-size:13px;
            line-height:1.6;
        ">
        Separate the input features from the
        target variable used for prediction.
        </p>

        </div>
        """, unsafe_allow_html=True)


    st.markdown("<br>", unsafe_allow_html=True)


    # ========================================================
    # SECOND ROW
    # ========================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("""
        <div style="
            background:white;
            padding:20px;
            border-radius:12px;
            border:1px solid #e2e8f0;
            min-height:170px;
        ">

        <div style="font-size:22px;">📊</div>

        <h4 style="color:#17395f;">
        04. Data Splitting
        </h4>

        <p style="
            color:#718096;
            font-size:13px;
            line-height:1.6;
        ">
        Split the available data into training
        and testing sets for model development
        and evaluation.
        </p>

        </div>
        """, unsafe_allow_html=True)


    with col2:

        st.markdown("""
        <div style="
            background:white;
            padding:20px;
            border-radius:12px;
            border:1px solid #e2e8f0;
            min-height:170px;
        ">

        <div style="font-size:22px;">🤖</div>

        <h4 style="color:#17395f;">
        05. Model Training
        </h4>

        <p style="
            color:#718096;
            font-size:13px;
            line-height:1.6;
        ">
        Train multiple classification algorithms
        including Logistic Regression, Random Forest,
        SVM and XGBoost.
        </p>

        </div>
        """, unsafe_allow_html=True)


    with col3:

        st.markdown("""
        <div style="
            background:white;
            padding:20px;
            border-radius:12px;
            border:1px solid #e2e8f0;
            min-height:170px;
        ">

        <div style="font-size:22px;">📈</div>

        <h4 style="color:#17395f;">
        06. Model Evaluation
        </h4>

        <p style="
            color:#718096;
            font-size:13px;
            line-height:1.6;
        ">
        Evaluate and compare models using
        Accuracy, Precision, Recall, F1 Score
        and ROC-AUC.
        </p>

        </div>
        """, unsafe_allow_html=True)


    st.markdown("<br>", unsafe_allow_html=True)


    # ========================================================
    # THIRD ROW
    # ========================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("""
        <div style="
            background:white;
            padding:20px;
            border-radius:12px;
            border:1px solid #e2e8f0;
            min-height:170px;
        ">

        <div style="font-size:22px;">🏆</div>

        <h4 style="color:#17395f;">
        07. Best Model Selection
        </h4>

        <p style="
            color:#718096;
            font-size:13px;
            line-height:1.6;
        ">
        Identify the best-performing model based
        on the evaluation results.
        </p>

        </div>
        """, unsafe_allow_html=True)


    with col2:

        st.markdown("""
        <div style="
            background:white;
            padding:20px;
            border-radius:12px;
            border:1px solid #e2e8f0;
            min-height:170px;
        ">

        <div style="font-size:22px;">💾</div>

        <h4 style="color:#17395f;">
        08. Model Saving
        </h4>

        <p style="
            color:#718096;
            font-size:13px;
            line-height:1.6;
        ">
        Save the trained models and scaler so
        they can be reused for future predictions.
        </p>

        </div>
        """, unsafe_allow_html=True)


    with col3:

        st.markdown("""
        <div style="
            background:white;
            padding:20px;
            border-radius:12px;
            border:1px solid #e2e8f0;
            min-height:170px;
        ">

        <div style="font-size:22px;">🚀</div>

        <h4 style="color:#17395f;">
        09. Prediction & Deployment
        </h4>

        <p style="
            color:#718096;
            font-size:13px;
            line-height:1.6;
        ">
        Deploy the trained models through a
        Streamlit application for interactive prediction.
        </p>

        </div>
        """, unsafe_allow_html=True)


    # ========================================================
    # MODELS USED
    # ========================================================
    # ========================================================
    # MACHINE LEARNING MODELS
    #  ========================================================

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("🤖 Machine Learning Models")

    st.write(
    "The following classification algorithms are used and compared in this project.")

    st.markdown("""
    <div style="
    background: white;
    padding: 25px;
    border-radius: 14px;
    border: 1px solid #dbe5e7;
    box-shadow: 0 5px 18px rgba(20,50,90,0.06);
    ">

    <span style="display:inline-block; background:#e6f8f8; color:#07858a; padding:7px 13px; border-radius:15px; font-size:12px; font-weight:600; margin:4px;">
    Logistic Regression
    </span>

    <span style="display:inline-block; background:#e6f8f8; color:#07858a; padding:7px 13px; border-radius:15px; font-size:12px; font-weight:600; margin:4px;">
    Decision Tree
    </span>

    <span style="display:inline-block; background:#e6f8f8; color:#07858a; padding:7px 13px; border-radius:15px; font-size:12px; font-weight:600; margin:4px;">
    Random Forest
    </span>

    <span style="display:inline-block; background:#e6f8f8; color:#07858a; padding:7px 13px; border-radius:15px; font-size:12px; font-weight:600; margin:4px;">
    KNN
    </span>

    <span style="display:inline-block; background:#e6f8f8; color:#07858a; padding:7px 13px; border-radius:15px; font-size:12px; font-weight:600; margin:4px;">
    SVM
    </span>

    <span style="display:inline-block; background:#e6f8f8; color:#07858a; padding:7px 13px; border-radius:15px; font-size:12px; font-weight:600; margin:4px;">
    Naive Bayes
    </span>

    <span style="display:inline-block; background:#e6f8f8; color:#07858a; padding:7px 13px; border-radius:15px; font-size:12px; font-weight:600; margin:4px;">
    AdaBoost
    </span>

    <span style="display:inline-block; background:#e6f8f8; color:#07858a; padding:7px 13px; border-radius:15px; font-size:12px; font-weight:600; margin:4px;">
    Gradient Boosting
    </span>

    <span style="display:inline-block; background:#e6f8f8; color:#07858a; padding:7px 13px; border-radius:15px; font-size:12px; font-weight:600; margin:4px;">
    XGBoost
    </span>

    </div>
    """, unsafe_allow_html=True)

        


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.subheader("⚠️ Disclaimer")

    st.warning(
        """
        This application is intended for educational and
        informational purposes only. It does not provide a
        medical diagnosis. Predictions should not replace
        professional medical advice, diagnosis, or treatment.
        """
    )