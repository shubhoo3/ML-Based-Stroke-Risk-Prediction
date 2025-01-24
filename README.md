# Heart Disease Prediction Project

## Overview
This project aims to develop a predictive model to determine the presence of heart disease in patients based on a set of medical attributes. By facilitating early diagnosis and potential intervention, this model seeks to improve patient outcomes and enhance healthcare efficiency.

## Dataset
The analysis is based on the **Heart Disease dataset** from the UCI Machine Learning Repository. The dataset contains 14 medical attributes for patients, including the target variable indicating the presence of heart disease.

### Data Attributes
| Attribute  | Description |
|------------|-------------|
| `age`      | The person's age in years |
| `sex`      | The person's sex (1 = male, 0 = female) |
| `cp`       | Type of chest pain experienced (1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic) |
| `trestbps` | Resting blood pressure (mm Hg on admission to the hospital) |
| `chol`     | Cholesterol measurement in mg/dl |
| `fbs`      | Fasting blood sugar (> 120 mg/dl, 1 = true, 0 = false) |
| `restecg`  | Resting electrocardiographic measurement (0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy) |
| `thalach`  | Maximum heart rate achieved |
| `exang`    | Exercise-induced angina (1 = yes, 0 = no) |
| `oldpeak`  | ST depression induced by exercise relative to rest |
| `slope`    | Slope of the peak exercise ST segment (1 = upsloping, 2 = flat, 3 = downsloping) |
| `ca`       | Number of major vessels (0-3) |
| `thal`     | Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect) |
| `target`   | Heart disease (0 = no, 1 = yes) |

## Project Objectives
- Analyze the dataset to identify patterns and relationships between attributes.
- Build a predictive model using **Logistic Regression** to classify patients based on the presence of heart disease.
- Evaluate the model's performance to ensure accuracy and reliability.

## Tools and Technologies
- **Programming Language**: Python
- **Development Environment**: Google Colab
- **Libraries**:
  - `numpy` and `pandas` for data manipulation
  - `sklearn.model_selection` for splitting the dataset into training and test sets
  - `sklearn.linear_model.LogisticRegression` for building the predictive model

## Implementation
1. **Data Preprocessing**:
   - Load and inspect the dataset.
   - Handle missing values (if any) and perform feature engineering if necessary.
2. **Exploratory Data Analysis (EDA)**:
   - Visualize distributions and relationships between attributes.
   - Identify potential correlations between features and the target variable.
3. **Model Development**:
   - Split the dataset into training and testing sets.
   - Train a Logistic Regression model.
   - Evaluate the model using metrics such as accuracy, precision, recall, and F1-score.
4. **Interpretation**:
   - Analyze model results to derive insights for healthcare applications.

## Getting Started
### Prerequisites
Ensure you have Python installed along with the following libraries:
- `numpy`
- `pandas`
- `scikit-learn`

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/shubhoo3/ML-Based-Stroke-Risk-Prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ML-Based-Stroke-Risk-Prediction
   ```
3. Open Google Colab and upload the notebook file `heart_disease_prediction.ipynb`.

### Usage
1. Open the notebook in Google Colab.
2. Run the cells sequentially to see the analysis and model development.

## Results
The Logistic Regression model achieves [insert performance metrics] on the test dataset, demonstrating its ability to accurately predict the presence of heart disease.

## Future Work
- Experiment with other machine learning algorithms (e.g., Decision Trees, Random Forest, SVM) to improve accuracy.
- Incorporate additional data sources to enhance predictive power.
- Deploy the model as a web application for real-world use.



---

Feel free to contribute or raise issues for improvements. Happy coding!
