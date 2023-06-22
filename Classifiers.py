def predict_with_classifiers(X_train, y_train, X_test, y_test):
    # Import necessary packages
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    # Import the necessary classifiers
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier

    # Import metrics
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score, f1_score

    # Instantiate the classifiers
    logistic_regression = LogisticRegression()
    decision_tree = DecisionTreeClassifier()
    random_forest = RandomForestClassifier()
    svm = SVC()
    knn = KNeighborsClassifier()
    naive_bayes = GaussianNB()
    gradient_boosting = GradientBoostingClassifier()
    neural_network = MLPClassifier()

    # Train the classifiers
    logistic_regression.fit(X_train, y_train)
    decision_tree.fit(X_train, y_train)
    random_forest.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    naive_bayes.fit(X_train, y_train)
    gradient_boosting.fit(X_train, y_train)
    neural_network.fit(X_train, y_train)

    # Make predictions with each classifier
    lr_predictions = logistic_regression.predict(X_test)
    dt_predictions = decision_tree.predict(X_test)
    rf_predictions = random_forest.predict(X_test)
    svm_predictions = svm.predict(X_test)
    knn_predictions = knn.predict(X_test)
    nb_predictions = naive_bayes.predict(X_test)
    gb_predictions = gradient_boosting.predict(X_test)
    nn_predictions = neural_network.predict(X_test)
    tree_predictions = decision_tree.predict(X_test)

    # Calculate confusion matrix for each classifier
    classifiers = {
        'Logistic Regression': lr_predictions,
        'Decision Tree': dt_predictions,
        'Random Forest': rf_predictions,
        'SVM': svm_predictions,
        'KNN': knn_predictions,
        'Naive Bayes': nb_predictions,
        'Gradient Boosting': gb_predictions,
        'Neural Network': nn_predictions,
        'Decision Tree': tree_predictions
    }

    metrics = {}
    for classifier_name, predictions in classifiers.items():
        cm = confusion_matrix(y_test, predictions)
        cm_df = pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_train))

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, cmap='Blues')
        plt.title(f'Confusion Matrix - {classifier_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        metrics[classifier_name] = {
            'Accuracy': accuracy,
            'F1 Score': f1,
        }

    # Print the metrics in the desired format
    for classifier_name, metric_values in metrics.items():
        print(f" {classifier_name}:")
        for metric_name, value in metric_values.items():
            print(f"  - {metric_name}: {value}")
