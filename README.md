# Automated ML Pipeline for Classification and Regression

This project provides a flexible and automated machine learning pipeline that can handle both classification and regression tasks. It includes preprocessing, handling imbalanced datasets, training multiple models, evaluation, and selection of the best-performing model.

## Features
- Automatically detects whether the task is **classification** or **regression** based on the target column
- Handles **numerical and categorical features**
- Imputes missing values:
  - Numerical: median
  - Categorical: most frequent
- Scales numerical features using **StandardScaler**
- Encodes categorical features using **OneHotEncoder**
- Uses **SMOTE** to balance classes in classification tasks
- Trains multiple models and selects the best based on performance:
  - **Classification models**: Logistic Regression, Random Forest, Decision Tree, SVM
  - **Regression models**: Linear Regression, Random Forest Regressor, Decision Tree Regressor, SVR

## Workflow
1. **Load Dataset**
```python
df = pd.read_csv("your_file.csv")
target = "target_column"
