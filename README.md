# Diabetes Classification 

This project aims to classify diabetes based on various medical features using machine learning algorithms. The dataset used contains diagnostic information, and the objective is to predict whether a patient has diabetes or not.

## Project Overview

The goal of this project is to build and evaluate machine learning models to predict the likelihood of a patient having diabetes based on input features like age, glucose level, blood pressure, insulin level, etc.

## Dataset

The dataset used in this project is the **Pima Indians Diabetes Database**, which contains the following columns:

- **Pregnancies**: Number of pregnancies
- **Glucose**: Plasma glucose concentration
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg / (height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function (a measure of family history)
- **Age**: Age in years
- **Outcome**: Whether the patient has diabetes (1) or not (0)

### Dataset Source

The dataset can be found at:  
[https://www.kaggle.com/uciml/pima-indians-diabetes-database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

## Technologies Used

- **Python**  
- **Google Colab** (for running the code and conducting analysis)
- **pandas** (for data manipulation and analysis)
- **scikit-learn** (for machine learning models and evaluation)
- **matplotlib** & **seaborn** (for data visualization)

## Steps Involved

1. **Data Import**: Import the dataset and load it into a pandas DataFrame.
2. **Data Preprocessing**: Handle missing values, convert categorical variables, and scale/normalize numerical features.
3. **Exploratory Data Analysis (EDA)**: Visualize data distributions and analyze relationships between features and the target variable.
4. **Modeling**: Implement various machine learning models like:
   - Logistic Regression
   - Decision Trees
   - Random Forest
   - Support Vector Machine (SVM)
   - k-Nearest Neighbors (KNN)
5. **Model Evaluation**: Evaluate models using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.
6. **Model Selection**: Choose the best performing model based on evaluation metrics.

## How to Run

1. Open the Google Colab notebook file in your browser.
2. Ensure that all required libraries are installed. If not, run the following:
   ```python
   !pip install pandas scikit-learn matplotlib seaborn
   ```
3. Run the notebook cells in sequence to load, preprocess, analyze the data, train models, and evaluate their performance.

## Model Evaluation Results

- **Accuracy**: 0.78 (for the best performing model)
- **Precision**: 0.75
- **Recall**: 0.82
- **F1-score**: 0.78
- **ROC-AUC**: 0.84

## Conclusion

The models trained on this dataset can predict the likelihood of diabetes with reasonable accuracy. The Random Forest classifier was found to be the most effective model for this problem. 

## Future Improvements

- Experiment with other algorithms like Neural Networks or XGBoost.
- Apply cross-validation for more robust model performance.
- Fine-tune model hyperparameters using grid search or random search.
