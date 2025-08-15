from sklearn.metrics import classification_report

def evaluate_model(model, X_test, y_test):
    """
    Check how well a trained model performs on test data.
    Prints a classification report with precision, recall, F1, and support.
    """

    # 1) Makes predictions using the model and test features
    y_pred = model.predict(X_test)

    # 2) Prints the results in a readable table
    print("Classification Report:")
    print(classification_report(y_test, y_pred))