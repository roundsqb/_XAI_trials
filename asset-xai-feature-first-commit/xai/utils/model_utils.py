from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBRegressor, XGBClassifier

# utlis
from time import time
from sklearn.utils import multiclass
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
from pandas import DataFrame


def get_trained_model_object(x, y, model_type):
    """
    The function
    (1) determines whether the problem is classification or regression (from data type of y);
    (2) instantiates an sklearn model of specified type (RF) using default settings;
    (3) trains the model with the data in x
    (4) returns a trained model object a binary flag of whether the trained model is_classification
    :param x: pandas dataframe of independent variables (in columns) to be used for model training
    :param y: array containing the dependent variable to be used as a target
    :param model_type: string describing the model type. Currently supported model types are
    ['xgboost', 'rf'].
    :return:
        model: trained sklearn model object
        is_classification: binary flag determining if the trained model is classification (True) or
        regression (False)
    """
    supported_model_types = ['rf', 'xgboost']
    assert (model_type in supported_model_types), \
        'currently supported model types are {}'.format(supported_model_types)

    supported_class_y_types = ['binary', 'multiclass']
    supported_reg_y_types = ['continuous']
    y_type = multiclass.type_of_target(y)
    assert (y_type in supported_class_y_types or y_type in supported_reg_y_types), \
        'currently supported target types are {} (classification) or {} (regression)' \
        .format(supported_class_y_types, supported_reg_y_types)
    # TODO: move default parameters to config
    try:
        if y_type in supported_class_y_types:
            is_classification = True
            if model_type == 'xgboost':
                model = XGBClassifier(max_depth=8,
                                      learning_rate=0.01,
                                      n_estimators=100,
                                      silent=True,
                                      objective='binary:logistic',
                                      booster='gbtree',
                                      n_jobs=1,
                                      nthread=None,
                                      gamma=0,
                                      min_child_weight=1,
                                      max_delta_step=0,
                                      subsample=1,
                                      colsample_bytree=1,
                                      colsample_bylevel=1,
                                      reg_alpha=0,
                                      reg_lambda=1,
                                      scale_pos_weight=1,
                                      base_score=0.5,
                                      random_state=42,
                                      seed=2018,
                                      missing=None
                                      )
            if model_type == 'rf':
                model = RandomForestClassifier(
                    n_estimators=100,
                    criterion='gini',
                    bootstrap=True,
                    class_weight=None,
                    max_depth=8,
                    max_features='auto',
                    max_leaf_nodes=None,
                    min_impurity_decrease=0.0,
                    min_impurity_split=None,
                    min_samples_leaf=1,
                    min_samples_split=2,
                    min_weight_fraction_leaf=0.0,
                    n_jobs=1,
                    oob_score=False,
                    random_state=42,
                    verbose=0,
                    warm_start=False
                )

        elif y_type in supported_reg_y_types:
            is_classification = False
            if model_type == 'xgboost':
                model = XGBRegressor(max_depth=6,
                                     learning_rate=0.01,
                                     n_estimators=100,
                                     silent=True,
                                     objective='reg:linear',
                                     booster='gbtree',
                                     n_jobs=1,
                                     nthread=None,
                                     gamma=0,
                                     min_child_weight=1,
                                     max_delta_step=0,
                                     subsample=1,
                                     colsample_bytree=1,
                                     colsample_bylevel=1,
                                     reg_alpha=0,
                                     reg_lambda=1,
                                     scale_pos_weight=1,
                                     base_score=0.5,
                                     random_state=42,
                                     seed=2018,
                                     missing=None
                                     )
            if model_type == 'rf':
                model = RandomForestRegressor(
                    n_estimators=100,
                    criterion='mse',
                    bootstrap=True,
                    max_depth=8,
                    max_features='auto',
                    max_leaf_nodes=None,
                    min_impurity_decrease=0.0,
                    min_impurity_split=None,
                    min_samples_leaf=1,
                    min_samples_split=2,
                    min_weight_fraction_leaf=0.0,
                    n_jobs=1,
                    oob_score=False,
                    random_state=42,
                    verbose=0,
                    warm_start=False
                )
    except ValueError:
        raise ValueError("Ambiguous y type: {}".format(y_type))
    return model.fit(x, y), is_classification


def score_model(model, x_test, y_test):
    """
    Computes and displays model performance on the holdout data provided
    :param model: sklearn model object. This must have .predict() method.
    :param x_test: pandas DataFrame
    :param y_test: numeric or binary array that must match the model type
    (classificaiton or regression)
    """
    assert hasattr(model, 'predict'), "Model must have .predict() method"
    if isinstance(y_test, DataFrame):
        assert x_test.shape[0] == y_test.shape[0], "X and Y provided must be of equal length."
    else:
        assert x_test.shape[0] == len(y_test), "X and Y provided must be of equal length."
    supported_class_y_types = ['binary', 'multiclass']
    supported_reg_y_types = ['continuous']
    try:
        y_type = multiclass.type_of_target(y_test)
    except ValueError:
        print("Unsupported y type")
        perf = 'None'
    else:
        assert (y_type in supported_class_y_types or y_type in supported_reg_y_types), \
            'currently supported target types are {} (classification) or {} (regression)'\
            .format(supported_class_y_types, supported_reg_y_types)
        y_score = model.predict(x_test)
        if y_type in supported_class_y_types:
            perf = roc_auc_score(y_test, y_score, sample_weight=None)
            print("Model test performance: \tAUC = {:.3f}, \tGini = {:.3f}"
                  .format(perf, 2 * perf - 1))
        if y_type in supported_reg_y_types:
            mse = mean_squared_error(y_test, y_score, sample_weight=None)
            perf = r2_score(y_test, y_score, sample_weight=None)
            print("Model test performance: \tMSE = {:.3f}, \tR^2 = {:.3f}".format(mse, perf))
    return perf


def generate_trained_model(model_type, x, x_test, y, y_test):
    """
    Function for reproducing modelling pipeline for the experiments with explain_model()
    Generates a model object, trains it, and scores it.
    :param model_type: string, presently supported values are 'xgboost' or 'rf'
    :param x:
    :param x_test:
    :param y:
    :param y_test:
    :return: model: pre-trained sklearn model object
    """
    ts = time()

    if isinstance(y, DataFrame):
        print("\nInput dataset shape: \nx {}, \ty {}".format(x.shape, y.shape))
    else:
        print("\nInput dataset shape: \nx {}, \ty {}".format(x.shape, len(y)))

    model, is_classification = get_trained_model_object(x, y, model_type=model_type)
    print("Model type: " + str(type(model)))
    score_model(model, x_test, y_test)
    te = time()
    print("Process took {:.2f} s.".format((te - ts)))
    return model, is_classification


def rf_feature_importances(model, x, n=None):
    """
    Identifies and returns top n features based on the Random Forest importance scores
    :param model: sklearn RandomForest object (must have .feature_importances_ method)
    :param x: pandas DataFrame containing features with which the model object was trained
    (must have .columns method)
    :param n: number of top features to return. n must be <= number of features
    :return:
        top_n_features (n-by-1) pandas DataFrame containing features in rows and their importance
        values in 0th column
    """
    assert hasattr(model, 'feature_importances_'), \
        'Model object must have feature_importances_ attribute'
    if n is None:
        n = x.shape[1]  # return all features
    elif n > x.shape[1]:
        n = x.shape[1]
    feature_importances = DataFrame(data=model.feature_importances_,
                                    index=x.columns, columns=['ft_imp'])
    top_n_features = feature_importances.sort_values(by='ft_imp', ascending=False).head(n)
    return top_n_features
