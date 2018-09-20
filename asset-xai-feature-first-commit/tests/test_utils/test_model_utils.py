import pytest
from pandas import DataFrame
from xai.utils.model_utils import (get_trained_model_object, generate_trained_model, score_model,
                                   rf_feature_importances)


@pytest.fixture(scope="session")
def data():
    dummy_data = [[33.0, 2, 13.0, 4], [36.0, 4, 11.0, 2], [58.0, 6, 5.0, 5], [21.0, 4, 11.0, 2],
                  [27.0, 4, 10.0, 2], [44.0, 4, 13.0, 2], [33.0, 4, 6.0, 4], [62.0, 6, 13.0, 2],
                  [20.0, 4, 9.0, 4], [33.0, 4, 9.0, 0]]

    data.x = DataFrame(dummy_data, columns=['Age', 'Workclass', 'Education-Num', 'Marital Status'])
    data.y = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
    data.y_reg = [0.1, 2.2, 0.34, -1.1, 12, 4.3, 0.5, 4, -2.3, -2]
    data.x_test = data.x
    data.y_test = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    return data


@pytest.fixture(scope="session")
def model_reg(data):
    model, is_classification = get_trained_model_object(data.x, data.y_reg, model_type='rf')
    return model


@pytest.fixture(scope="session")
def model_class(data):
    model, is_classification = get_trained_model_object(data.x, data.y, model_type='rf')
    return model


@pytest.fixture(scope="session")
def xgboost_reg(data):
    model, is_classification = get_trained_model_object(data.x, data.y_reg, model_type='xgboost')
    return model


@pytest.fixture(scope="session")
def xgboost_class(data):
    model, is_classification = get_trained_model_object(data.x, data.y, model_type='xgboost')
    return model


# Testing get trained model object for RF and XGBoost models


def test_get_trained_model_object_class_rf(data):
    model, is_classification = get_trained_model_object(data.x, data.y, model_type='rf')
    assert 'Classif' in str(type(model))
    assert is_classification is True
    assert hasattr(model, 'predict') is True
    assert hasattr(model, 'predict_proba') is True
    scores = model.predict_proba(data.x)
    assert scores.shape == (10, 2)
    assert (scores[0:2, 1] == [0.16, 0.58]).all()


def test_get_trained_model_object_reg_rf(data):
    model, is_classification = get_trained_model_object(data.x, data.y_reg, model_type='rf')
    assert 'Regress' in str(type(model))
    assert is_classification is False
    assert hasattr(model, 'predict') is True
    assert hasattr(model, 'predict_proba') is False
    scores = model.predict(data.x)
    assert scores.shape == (10,)
    assert round(float(scores[1]), 3) == 1.853


def test_get_trained_model_object_class_xgboost(data):
    model, is_classification = get_trained_model_object(data.x, data.y, model_type='xgboost')
    assert 'Classif' in str(type(model))
    assert is_classification is True
    assert hasattr(model, 'predict') is True
    assert hasattr(model, 'predict_proba') is True
    scores = model.predict_proba(data.x)
    assert scores.shape == (10, 2)
    assert round(float(scores[1, 1]), 2) == 0.54


def test_get_trained_model_object_reg_xgboost(data):
    model, is_classification = get_trained_model_object(data.x, data.y_reg, model_type='xgboost')
    assert 'Regress' in str(type(model))
    assert is_classification is False
    assert hasattr(model, 'predict') is True
    assert hasattr(model, 'predict_proba') is False
    assert model.predict(data.x).shape == (10,)


def test_assertions_in_get_trained_model_object(data):
    with pytest.raises(AssertionError, match='supported model types'):
        get_trained_model_object(data.x, data.y, model_type='shop')
    with pytest.raises(AssertionError, match='supported model types'):
        get_trained_model_object(data.x, data.y, model_type='neuralnet')
    with pytest.raises(AssertionError, match='supported target types'):
        get_trained_model_object(data.x, data.x, model_type='xgboost')


# Test score model for Classification and Regression XGBoost and sklearn RF models


def test_score_model_class_rf(model_class, data):
    perf = score_model(model_class, data.x_test, data.y_test)
    assert round(float(perf), 3) == 0.4


def test_score_model_reg_rf(model_reg, data):
    perf = score_model(model_reg, data.x_test, data.y_test)
    assert round(float(perf), 3) == 0.28


def test_score_model_class_xgboost(xgboost_class, data):
    perf = score_model(xgboost_class, data.x_test, data.y_test)
    assert round(float(perf), 3) == 0.2


def test_score_model_reg_xgboost(xgboost_reg, data):
    perf = score_model(xgboost_reg, data.x_test, data.y_test)
    assert round(float(perf), 3) == 0.46


def test_assertions_in_score_model(model_reg, data):
    with pytest.raises(AssertionError, match='predict()'):
        score_model(data, data.x, data.y_reg)

    with pytest.raises(AssertionError, match='equal length'):
        score_model(model_reg, data.x, data.y[0:5])

    with pytest.raises(ValueError, match='multiclass format'):
        score_model(model_reg, data.x, [10, 'temp', 45.8, -0.56, 'he', 'bromp', 'tot', -0.55, 1, 2])

    with pytest.raises(Exception) as excinfo:
        score_model(model_reg, data.x,
                    [[1.5, 2.0], [3.0, 1.6], [1.5], [2.0], [1.5], [2.0], [1.5], [2], [1.5], [2]])
        assert isinstance(excinfo.type, type(ValueError))


# Test Generate_trained_model


def test_generate_trained_model(data):
    model, is_classification = generate_trained_model('rf', data.x, data.x_test, data.y,
                                                      data.y_test)
    assert 'classif' in str(type(model)).lower()
    assert is_classification is True
    assert hasattr(model, 'predict_proba') is True

    model, is_classification = generate_trained_model('rf', data.x, data.x_test, data.y_reg,
                                                      data.y_test)
    assert 'regress' in str(type(model)).lower()
    assert is_classification is False
    assert hasattr(model, 'predict') is True

    model, is_classification = generate_trained_model('xgboost', data.x, data.x_test, data.y,
                                                      data.y_test)
    assert 'classif' in str(type(model)).lower()
    assert is_classification is True
    assert hasattr(model, 'predict') is True
    assert hasattr(model, 'predict_proba') is True

    model, is_classification = generate_trained_model('xgboost', data.x, data.x_test, data.y_reg,
                                                      data.y_test)
    assert 'regress' in str(type(model)).lower()
    assert is_classification is False
    assert hasattr(model, 'predict') is True


# Test RF reature importances


def test_rf_feature_importances_class(model_class, data):
    top_n_features = rf_feature_importances(model_class, data.x, n=None)
    assert len(top_n_features) == data.x.shape[1]
    assert isinstance(top_n_features, DataFrame)
    assert top_n_features.iloc[0, :].name == 'Age'
    assert top_n_features.iloc[0, 0].round(3) == 0.631

    top_n_features = rf_feature_importances(model_class, data.x, n=3)
    assert len(top_n_features) == 3
    assert top_n_features.iloc[0, :].name == 'Age'
    assert top_n_features.iloc[0, 0].round(3) == 0.631

    top_n_features = rf_feature_importances(model_class, data.x, n=None)
    assert len(top_n_features) == data.x.shape[1]
    assert top_n_features.iloc[0, :].name == 'Age'
    assert top_n_features.iloc[0, 0].round(3) == 0.631


def test_rf_feature_importances_reg(model_reg, data):
    top_n_features = rf_feature_importances(model_reg, data.x, n=None)
    assert len(top_n_features) == data.x.shape[1]
    assert isinstance(top_n_features, DataFrame)
    assert top_n_features.iloc[0, :].name == 'Age'
    assert top_n_features.iloc[0, 0].round(3) == 0.431

    top_n_features = rf_feature_importances(model_reg, data.x, n=1000)
    assert len(top_n_features) == data.x.shape[1]
    assert top_n_features.iloc[0, :].name == 'Age'
    assert top_n_features.iloc[0, 0].round(3) == 0.431


def test_assertion_in_rf_feature_importances(xgboost_class, data):
    with pytest.raises(AssertionError, match='feature_importances_ attribute'):
        rf_feature_importances(xgboost_reg, data.x, n=None)
