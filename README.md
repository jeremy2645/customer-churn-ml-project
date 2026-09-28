# Customer Churn Prediction & Business Analysis

A beginner-friendly end-to-end machine learning and business analysis project focused on understanding customer churn, identifying high-risk customer segments, and generating actionable business insights.

## Project Overview

Customer churn is a major business problem for subscription-based companies. Losing customers can reduce recurring revenue and increase the cost of acquiring new customers.

This project uses customer data to:

* Clean and prepare real-world business data
* Perform exploratory data analysis (EDA)
* Identify patterns associated with customer churn
* Build and compare machine learning classification models
* Segment customers using a business-oriented risk scoring system
* Generate business insights and potential retention actions
* Store analysis results in reusable CSV files
* Create visualizations for business interpretation

The project combines **Python, data analysis, machine learning, and business analysis** concepts.

---

## Business Problem

A telecommunications company wants to understand why customers leave its service.

The business questions explored in this project include:

* What percentage of customers are churning?
* Which contract types have higher observed churn rates?
* Does customer tenure appear to be associated with churn?
* How does monthly charge relate to observed churn?
* Are churn rates different across internet service types?
* Which customers fall into higher business-risk segments?
* Can machine learning help identify customers who may be more likely to churn?

The analysis focuses on identifying **patterns and associations**, not proving that a particular factor directly causes churn.

---

## Dataset

The project uses the **Telco Customer Churn dataset**.

The original dataset contains customer information such as:

* Customer ID
* Gender
* Senior Citizen status
* Partner and Dependents
* Tenure
* Phone Service
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming Services
* Contract
* Paperless Billing
* Payment Method
* Monthly Charges
* Total Charges
* Churn

The original dataset contains **7,043 customer records and 21 columns**.

After cleaning invalid `TotalCharges` values, the working dataset contains **7,032 customer records**.

---

## Data Cleaning

The following data preparation steps were performed:

1. Loaded the raw CSV dataset using Pandas.
2. Checked the dataset structure and data types.
3. Identified blank values in `TotalCharges`.
4. Converted `TotalCharges` into a numeric column.
5. Removed 11 records where `TotalCharges` could not be converted.
6. Checked for duplicate customer IDs.
7. Created a `TenureGroup` feature.
8. Created a `MonthlyChargeGroup` feature.
9. Saved the cleaned dataset as:

```text
data/cleaned_customer_churn.csv
```

### Additional Features

#### TenureGroup

Customers were grouped into:

* 0–12 months
* 13–24 months
* 25–48 months
* 49–72 months

#### MonthlyChargeGroup

Customers were grouped into:

* $0–40
* $41–70
* $71–100
* $101–120

---

## Exploratory Data Analysis

The project analyzes churn across several business dimensions.

### Overall Churn

The cleaned dataset contains:

* **5,163 customers who did not churn**
* **1,869 customers who churned**
* Observed churn rate: approximately **26.58%**

### Churn by Contract

Observed churn rates:

| Contract       | Churn Rate |
| -------------- | ---------: |
| Month-to-month |     42.71% |
| One year       |     11.28% |
| Two year       |      2.85% |

The analysis shows a substantial difference in observed churn across contract types.

### Churn by Tenure

| Tenure Group | Churn Rate |
| ------------ | ---------: |
| 0–12 months  |     47.68% |
| 13–24 months |     28.71% |
| 25–48 months |     20.39% |
| 49–72 months |      9.51% |

Customers in the earliest tenure group show the highest observed churn rate in this dataset.

### Churn by Internet Service

| Internet Service    | Churn Rate |
| ------------------- | ---------: |
| DSL                 |     19.00% |
| Fiber optic         |     41.89% |
| No internet service |      7.43% |

These results represent associations in the dataset and should not be interpreted as proof that internet service type itself causes churn.

---

## Visualizations

The project generates visualizations for:

* Churn by contract
* Churn by tenure
* Churn by internet service
* Machine learning model comparison

The generated charts are stored inside:

```text
visual/
```

---

## Machine Learning

Three classification algorithms are used in the project:

### 1. Logistic Regression

Used as a baseline classification model.

### 2. Decision Tree

Used to model non-linear relationships and provide an interpretable tree-based approach.

### 3. Random Forest

Used as an ensemble model consisting of multiple decision trees.

### Preprocessing

The machine learning pipeline uses:

* Train/test split
* One-hot encoding for categorical variables
* Standard scaling for numerical variables
* `ColumnTransformer`
* Scikit-learn `Pipeline`

Using a preprocessing pipeline helps keep the transformation process organized and reduces the risk of applying preprocessing incorrectly across training and testing data.

### Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

The model comparison results are saved to:

```text
data/model_comparison.csv
```

The comparison visualization is saved to:

```text
visual/model_comparison.png
```

---

## Customer Risk Analysis

In addition to machine learning, the project includes a **business-rule-based customer risk scoring system**.

The risk score uses selected customer characteristics:

| Business Factor          | Score |
| ------------------------ | ----: |
| Month-to-month contract  |    +2 |
| Tenure ≤ 12 months       |    +2 |
| Monthly charges ≥ $70    |    +1 |
| Fiber optic internet     |    +1 |
| Electronic check payment |    +1 |

Customers are then categorized as:

* **Low Risk**
* **Medium Risk**
* **High Risk**

### Important

This risk score is **not an ML prediction**.

It is a transparent business segmentation method designed to demonstrate how business rules can be converted into a simple customer-risk framework.

Results are saved to:

```text
data/customer_risk_analysis.csv
data/risk_summary.csv
```

---

## Business Analysis

The project translates the analytical findings into potential business actions.

### Month-to-Month Customers

The dataset shows higher observed churn among month-to-month customers.

Potential business responses could include:

* Retention incentives
* Contract upgrade options
* Customer engagement campaigns
* Early identification of customers showing signs of disengagement

### New Customers

Customers with shorter tenure show higher observed churn.

Potential responses could include:

* Improved onboarding
* Early customer check-ins
* First-year retention campaigns
* Additional support during the early customer lifecycle

### Fiber Optic Customers

Fiber optic customers have a higher observed churn rate in this dataset.

A business team could investigate:

* Service quality
* Technical support
* Pricing and perceived value
* Customer service experience

The analysis itself does not establish which of these factors causes churn.

### Higher Monthly Charges

Higher monthly-charge groups show different observed churn patterns.

Possible areas for investigation include:

* Pricing perception
* Service value
* Competitor pricing
* Customer support
* Plan configuration

---

## Project Structure

```text
Customer Churn ML Project
│
├── data
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   ├── cleaned_customer_churn.csv
│   ├── model_comparison.csv
│   ├── business_insights.csv
│   ├── churn_by_contract.csv
│   ├── churn_by_tenure.csv
│   ├── churn_by_internet_service.csv
│   ├── churn_by_monthly_charge.csv
│   ├── customer_risk_analysis.csv
│   └── risk_summary.csv
│
├── src
│   ├── test_setup.py
│   ├── data_loading.py
│   ├── eda.py
│   ├── model.py
│   ├── business_analysis.py
│   └── risk_analysis.py
│
├── visual
│   ├── churn_by_contract.png
│   ├── churn_by_internet_service.png
│   ├── churn_by_tenure.png
│   └── model_comparison.png
│
├── notebooks
│
├── .gitignore
└── README.md
```

---

## Technologies Used

### Programming & Data Analysis

* Python
* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* Logistic Regression
* Decision Tree
* Random Forest
* One-Hot Encoding
* Standard Scaling
* Machine Learning Pipelines

### Development Tools

* Antigravity
* PowerShell
* Git
* GitHub

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/jeremy2645/customer-churn-ml-project.git
```

### 2. Open the project

```bash
cd customer-churn-ml-project
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment on Windows

```powershell
.venv\Scripts\activate
```

### 5. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 6. Run the data preparation script

```bash
python src/data_loading.py
```

### 7. Run exploratory data analysis

```bash
python src/eda.py
```

### 8. Run the machine learning models

```bash
python src/model.py
```

### 9. Run business analysis

```bash
python src/business_analysis.py
```

### 10. Run customer risk analysis

```bash
python src/risk_analysis.py
```

---

## Key Learning Outcomes

Through this project, I practiced:

* Working with real-world business datasets
* Data cleaning with Pandas
* Handling missing and invalid values
* Feature creation
* Exploratory data analysis
* Business-oriented data interpretation
* Customer segmentation
* Classification machine learning
* Model evaluation
* Data preprocessing pipelines
* Business-rule-based risk scoring
* Translating analytical findings into business recommendations
* Organizing a Python project
* Using Git and GitHub for version control

---

## Future Improvements

Possible future improvements include:

* Hyperparameter tuning
* Cross-validation
* Handling class imbalance
* Probability-based churn prediction
* SHAP-based model explainability
* SQL-based customer analysis
* Customer Lifetime Value (CLV) analysis
* Interactive dashboards
* Automated retention recommendations
* Deployment using Streamlit
* Integration of a customer retention workflow

---

## Author

**Francis Jeremy**

B.Tech Computer Engineering
Silver Oak University

Interested in:

* Business Analysis
* Data Analytics
* Artificial Intelligence
* Machine Learning
* Business Intelligence
* Software Development

---

## Project Purpose

This project was developed as a learning and portfolio project to combine **technical data skills with business analysis thinking**.

The goal is not only to build a machine learning model, but also to understand how analytical results can be communicated and translated into potential business actions.
