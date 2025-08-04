from sklearn.metrics import classification_report

def evaluate_model(model, X_test, y_test):
    """
    Prints classification report.
    """
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))