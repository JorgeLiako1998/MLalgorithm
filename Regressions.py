def predict_with_regressors(X_train, y_train, X_test, y_test):

    # import Regression Models
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor

    # import metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.metrics import explained_variance_score, mean_squared_log_error
    from sklearn.metrics import mean_absolute_percentage_error

    # Instantiate the regressors
    regressors = {
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'Elastic Net': ElasticNet(),
        'SVM': SVR(),
        'KNN': KNeighborsRegressor(),
        'GBR': GradientBoostingRegressor(),
        'MLP': MLPRegressor(),
        'Linear': LinearRegression(),
        'Tree': DecisionTreeRegressor(),
        'Forest': RandomForestRegressor()
    }

    # train and make predictions with each regressor
    regressions = {}
    for model_name, model in regressors.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {
            'Mean Squared Error': mean_squared_error(y_test, y_pred),
            'Mean Absolute Error': mean_absolute_error(y_test, y_pred),
            'R-squared': r2_score(y_test, y_pred),
            'Explained Variance Score': explained_variance_score(y_test, y_pred),
            'Mean Squared Logarithmic Error': mean_squared_log_error(y_test, y_pred),
            'Mean Absolute Percentage Error': mean_absolute_percentage_error(y_test, y_pred)
        }
        regressions[model_name] = metrics

    # Print the metrics in an organized format
    for model_name, metrics in regressions.items():
        print(f'{model_name}:')
        for metric_name, value in metrics.items():
            print(f'  - {metric_name}: {value}')

    return regressions

