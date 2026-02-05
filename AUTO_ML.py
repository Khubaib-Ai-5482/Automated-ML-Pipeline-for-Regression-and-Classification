import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor , DecisionTreeClassifier
from sklearn.svm import SVC, SVR

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

file_path = input("Enter CSV file path: ")
target = input("Enter target column name: ")

df = pd.read_csv(file_path)
print("dataset Loaded" , df.shape)

if df[target].dtype == "object" or df[target].nunique() <= 20:
    task = "classification"
else:
    task = "regression"

X = df.drop(columns=[target])
y = df[target]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

if task == "classification":
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "Decision Tree":DecisionTreeClassifier(),
        "SVM": SVC(probability=True)
    }
else:
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "Decision Tree":DecisionTreeRegressor,
        "SVR": SVR()
    }

results = {}
best_model = None
best_score = -999

for name, model in models.items():
    print(f"\nTraining {name}...")

    if task == "classification":
        pipeline = ImbPipeline(steps=[
            ("preprocess", preprocessor),
            ("smote", SMOTE()),
            ("model", model)
        ])
    else:
        pipeline = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", model)
        ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    if task == "classification":
        score = accuracy_score(y_test, preds)*100
        results[name] = score
        print("Accuracy:", score)
        if score > best_score:
            best_score = score
            best_model = pipeline
    else:
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        results[name] = rmse
        print("RMSE:", rmse)
        if best_score == -999 or rmse < best_score:
            best_score = rmse
            best_model = pipeline

print("Best Model Selected")
print(best_model)

final_preds = best_model.predict(X_test)

if task == "classification":
    print("Classification Report")
    print(classification_report(y_test, final_preds))
else:
    print("Regression Metrics")
    print("MAE:", mean_absolute_error(y_test, final_preds))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, final_preds)))
    print("R2:", r2_score(y_test, final_preds))*100
