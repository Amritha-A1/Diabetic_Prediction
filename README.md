# Diabetic_Prediction
Diabetes Prediction Using Machine Learning

## 🌐 Live Demo

Live Application: "Diabetes Prediction Dashboard" (https://amritha-a1-diabetic-prediction-app-wa4pd1.streamlit.app/)

## 📌 Project Overview

Diabetes Prediction is a Machine Learning project developed to predict diabetes outcomes using medical and health-related parameters.

The project includes data preprocessing, Exploratory Data Analysis (EDA), data visualization, model training, model comparison, and performance evaluation. An interactive Streamlit application provides a user-friendly interface for diabetes prediction.

## 🎯 Objectives

- Analyze diabetes-related medical data.
- Perform data cleaning and preprocessing.
- Identify missing and invalid values.
- Conduct Exploratory Data Analysis.
- Train multiple Machine Learning classification models.
- Compare model performance using evaluation metrics.
- Find a suitable K value for KNN.
- Apply pruning to the Decision Tree model.
- Include XGBoost for diabetes prediction.
- Develop an interactive Streamlit application.

## ✨ Features

- Data loading and inspection
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Data visualization and correlation heatmaps
- Train-test splitting
- Multiple Machine Learning models
- KNN best K value selection
- Decision Tree pruning
- XGBoost classification
- Model comparison
- Accuracy and ROC curve visualizations
- Confusion matrix analysis
- Saved model files
- Interactive Streamlit application

## 📊 Dataset

The project uses a diabetes-related dataset containing medical and health-related features.

### Dataset Information

- Dataset File: "diabetes.csv"
- Data Type: Structured data
- Problem Type: Binary classification
- Target Variable: Diabetes outcome

The dataset is used for data analysis, model training, evaluation, and prediction.

## 🔍 Exploratory Data Analysis (EDA)

Exploratory Data Analysis helps understand the dataset and identify patterns before model training.

### EDA Techniques

- Checking dataset shape and structure
- Inspecting column names and data types
- Identifying missing and invalid values
- Analyzing numerical and categorical features
- Visualizing feature distributions
- Examining relationships between variables
- Creating correlation heatmaps
- Analyzing the target variable distribution
- Identifying potential outliers

### 🧹 Data Preprocessing

Data preprocessing prepares the dataset for Machine Learning algorithms.

### Preprocessing Steps

1. Load and inspect the dataset.
2. Check missing and invalid values.
3. Handle data quality issues where required.
4. Separate input features and the target variable.
5. Split data into training and testing sets.
6. Apply the required preprocessing techniques.
7. Prepare data for model training.

## 🤖 Machine Learning Models Used

Multiple classification algorithms are used to predict diabetes outcomes and compare their performance.

### 1. Logistic Regression

Logistic Regression is a supervised classification algorithm that estimates the probability of an outcome based on input features.

**Purpose**: To establish a baseline model and predict diabetes outcomes.

### 2. K-Nearest Neighbors (KNN)

KNN classifies a data point based on the classes of its nearest neighbors using a distance measure.

**Purpose**: To predict diabetes outcomes based on similarities between observations.

**Optimization**: Different K values were evaluated to identify a suitable K value.

### 3. Decision Tree Classifier

Decision Tree is a supervised learning algorithm that makes predictions using a tree-like structure of decision rules.

**Purpose**: To classify diabetes outcomes using feature-based decisions.

**Optimization**: Pruning was applied to control tree complexity and help reduce overfitting.

### 4. Random Forest Classifier

Random Forest combines predictions from multiple decision trees to produce a classification result.

**Purpose**: To classify diabetes outcomes using an ensemble of decision trees.

### 5. Support Vector Machine (SVM)

SVM identifies a decision boundary that separates classes and can use different kernel functions to model relationships in the data.

**Purpose**: To distinguish between diabetes outcome classes.

### 6. Naive Bayes

Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem.

**Purpose**: To predict diabetes outcomes using probabilities calculated from input features.

### 7. AdaBoost Classifier

AdaBoost is an ensemble learning algorithm that combines multiple weak learners into a stronger classifier.

**Purpose**: To classify diabetes outcomes by giving greater attention to previously misclassified observations.

### 8. Gradient Boosting Classifier

Gradient Boosting builds decision trees sequentially, with each new tree helping improve the predictions of the existing ensemble.

**Purpose**: To predict diabetes outcomes by combining multiple weak learners.

### 9. XGBoost Classifier

XGBoost stands for Extreme Gradient Boosting. It is an ensemble learning algorithm that builds trees sequentially to improve predictive performance.

**Purpose**: To classify diabetes outcomes and compare its performance with other Machine Learning algorithms.

## ⚙️ Model Optimization

Selected optimization techniques were applied to specific models.

### KNN - Best K Value Selection

Different K values were evaluated to identify a suitable number of neighbors for the KNN classifier.

### Decision Tree - Pruning

Pruning was applied to control tree complexity and help reduce overfitting.

### Hyperparameter Tuning

Systematic hyperparameter tuning was not performed for all models. The optimization work described in this project is limited to K value selection for KNN and pruning for the Decision Tree.

## 📈 Model Evaluation

The trained models are evaluated using classification metrics to understand their performance.

### Accuracy

Measures the proportion of predictions that are correct.

### Precision

Measures how many predicted positive cases are actually positive.

### Recall

Measures how many actual positive cases are correctly identified.

### F1-Score

Combines precision and recall into a single metric.

### Confusion Matrix

Summarizes true positives, true negatives, false positives, and false negatives.

### ROC Curve

The Receiver Operating Characteristic curve illustrates the relationship between the true positive rate and the false positive rate at different classification thresholds.

## 📊 Model Comparison

The project compares the performance of the trained models using available evaluation results.

### Comparison Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve

Using multiple metrics provides a better understanding of model performance than relying on accuracy alone.

### 🖥️ Streamlit Application

The project includes a Streamlit application that provides an interactive interface for diabetes prediction.

### Live Application

"Open Diabetes Prediction Dashboard" (https://amritha-a1-diabetic-prediction-app-wa4pd1.streamlit.app/)

The application uses saved Machine Learning models and the required preprocessing components to generate predictions.

## 🛠️ Technologies Used

- Programming Language: Python
- Data Analysis: Pandas, NumPy
- Data Visualization: Matplotlib, Seaborn
- Machine Learning: Scikit-learn
- Gradient Boosting: XGBoost
- Web Application: Streamlit
- Development Environment: VS Code, Jupyter Notebook
- Version Control: Git and GitHub
- Deployment: Streamlit Community Cloud

## 📂 Project Structure

'''Diabetic_Prediction/
│
├── .vscode/
│   └── settings.json
│
├── models/
│   ├── adaboost.pkl
│   ├── best_model.txt
│   ├── decision_tree.pkl
│   ├── gradient_boosting.pkl
│   ├── knn.pkl
│   ├── logistic_regression.pkl
│   ├── model_results.csv
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── roc_comparison.png
│   ├── scaler.pkl
│   ├── svm.pkl
│   └── xgboost.pkl
│
├── app.py
├── diabetes.csv
├── logistic.ipynb
├── train_model.py
├── requirements.txt
└── README.md'''

### File Description

- app.py: Runs the Streamlit application.
- train_model.py: Contains the model training code.
- diabetes.csv: Dataset used for diabetes prediction.
- logistic.ipynb: Jupyter Notebook for analysis and experimentation.
- models/: Stores trained models and supporting files.
- best_model.txt: Stores information about the selected best model.
- model_results.csv: Contains model comparison results.
- scaler.pkl: Stores the fitted scaler, if used.
- roc_comparison.png: ROC curve comparison visualization.
- requirements.txt: Lists the Python dependencies.
- README.md: Documents the project.

## 📋 Requirements

The project uses Python and the libraries listed in "requirements.txt".

### Main Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- streamlit

## 🚀 Installation and Setup

### Step 1: Clone the Repository

git clone https://github.com/Amritha-A1/Diabetic_Prediction.git

### Step 2: Navigate to the Project Folder

cd Diabetic_Prediction

### Step 3: Create a Virtual Environment (Optional)

python -m venv venv

### Activate on Windows

venv\Scripts\activate

### Step 4: Install Dependencies

pip install -r requirements.txt

### Step 5: Train the Machine Learning Models

python train_model.py

This executes the training script. Ensure the dataset is available and the script completes successfully.

### Step 6: Run the Streamlit Application

streamlit run app.py

### Step 7: Open the Application

Open the local URL displayed in the terminal. The default address is usually:

"http://localhost:8501"

## 🔄 Project Workflow

1. Dataset loading
2. Data inspection
3. Data cleaning and preprocessing
4. Exploratory Data Analysis
5. Train-test split
6. Machine Learning model training
7. KNN best K value selection
8. Decision Tree pruning
9. Model comparison
10. Model evaluation
11. Saving trained models
12. Streamlit application
13. Diabetes prediction

## 🔮 Future Improvements

- Perform systematic hyperparameter tuning for additional models.
- Improve model validation and generalization.
- Add more interactive visualizations.
- Improve model explainability.
- Enhance the user interface.
- Explore additional feature selection techniques.

## ⚠️ Disclaimer

This project is intended for educational purposes and Machine Learning demonstration only.

The predictions generated by this project are not a medical diagnosis and should not replace professional medical advice, laboratory testing, or consultation with a qualified healthcare professional.

## 👩‍💻 Author

Amritha Nandakumar A

GitHub: "Amritha-A1" (https://github.com/Amritha-A1)

## 📄 License

No license has been specified for this repository. Add an appropriate open-source license if you intend to permit reuse and distribution of the project.
