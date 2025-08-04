import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def run_models(df):
    """
    Runs classification models on processed corporate litigation data.
    """

    # Removes NaNs (theres a ton in the data at the moment)
    X = pd.get_dummies(df[["court"]].fillna("Unknown"), dummy_na=False)
    y = df["outcome_code"]

    # Skips model training if outcome_code contains only one class (TEMP)
    if len(y.unique()) < 2:
        print("Only one class present in outcome_code: skipping model training.")
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Logistic Regression
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)
    log_acc = accuracy_score(y_test, log_model.predict(X_test))

    # Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test))

    print(f"Logistic Regression Accuracy (temp): {log_acc:.2f}")
    print(f"Random Forest Accuracy (temp): {rf_acc:.2f}")

    return log_model, rf_model