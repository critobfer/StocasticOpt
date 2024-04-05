from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

def linear_regresion(X_train, X_test, y_train):
    result = {}
    transformations = ColumnTransformer([
        ('encoder', OneHotEncoder(), ['ZIPcod', 'Day Of Week', 'Holiday','Tomorrow Holiday'])
    ], remainder='passthrough')

    X_train_trans = transformations.fit_transform(X_train)
    X_test_trans = transformations.transform(X_test)
    # Inicializar y entrenar el modelo de regresión lineal
    model = LinearRegression()
    model.fit(X_train_trans, y_train )
    # Predecir en el conjunto de prueba
    demand_prediction = model.predict(X_test_trans)
    result['prediction'] = demand_prediction[0]

    # Extract the model info
    coefficients = model.coef_
    # Create a list with the information about feature coefficients
    coefficient_info_list = []
    for feature_name, coefficient in zip(transformations.get_feature_names_out(), coefficients):
        # Remove prefixes "remainder__" and "encoder__"
        feature_name = feature_name.replace("remainder__", "").replace("encoder__", "")
        coefficient_info_list.append(f"{feature_name}: {coefficient:.5f}")

    # Save the list to the result dictionary
    result['Coefficients'] = coefficient_info_list
    result['model_info'] = 'Coefficients'

    return result


def random_forest(X_train, X_test, y_train):
    result = {}
    transformations = ColumnTransformer([
        ('encoder', OneHotEncoder(), ['ZIPcod', 'Day Of Week', 'Holiday','Tomorrow Holiday'])
    ], remainder='passthrough')

    X_train_trans = transformations.fit_transform(X_train)
    X_test_trans = transformations.transform(X_test)
    # Inicializar y entrenar el modelo de regresión lineal
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_trans, y_train)
    # Predecir en el conjunto de prueba
    demand_prediction = model.predict(X_test_trans)
    result['prediction'] = demand_prediction[0]

    # Extract the model info
    features_importance = model.feature_importances_
    feature_importance_info_list = []
    for feature_name, importance in zip(transformations.get_feature_names_out(), features_importance):
        # Remove prefixes "remainder__" and "encoder__"
        feature_name = feature_name.replace("remainder__", "").replace("encoder__", "")
        feature_importance_info_list.append(f"{feature_name}: {importance:.5f}\n")

    # Save the string to the result dictionary
    result['Features Importance'] = feature_importance_info_list
    # result['Features Importance'] = features_importance
    result['model_info'] = 'Features Importance'

    return result


def svm(X_train, X_test, y_train):
    result = {}

    cathegorical = ['ZIPcod', 'Day Of Week', 'Holiday', 'Tomorrow Holiday']
    # List of all columns in the X_train DataFrame
    column_total = X_train.columns.tolist()
    # Filtering numerical columns
    numerical = [col for col in column_total if col not in cathegorical]

    transformations = ColumnTransformer([
        ('cathegorical', OneHotEncoder(), cathegorical),
        ('numerical', StandardScaler(), numerical)
                                            ])

    X_train_trans = transformations.fit_transform(X_train)
    X_test_trans = transformations.transform(X_test)
    # Adjust SVM model parameters using GridSearchCV
    param_grid = {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto']
    }
    grid_search = GridSearchCV(SVR(), param_grid, cv=5)
    grid_search.fit(X_train_trans, y_train)
    
    # Best model found by GridSearchCV
    model = grid_search.best_estimator_
    model.fit(X_train_trans, y_train)
    # Predecir en el conjunto de prueba
    demand_prediction = model.predict(X_test_trans)
    result['prediction'] = demand_prediction[0]

    # Extract the model info
    support_index = model.support_
    support_vector = X_train.iloc[support_index].reset_index(drop=True)
    result['Support Vector'] = support_index
    result['model_info'] = 'Support Vector'

    return result