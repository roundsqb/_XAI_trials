import pytest
from xai.performance_benchmarking.robustness import LimeWrapper, ShapWrapper
from pandas import DataFrame
from xai.utils.model_utils import get_trained_model_object
from time import time


@pytest.fixture(scope="session")
def data():
    dummy_data = [[33.0, 2, 13.0, 4], [36.0, 4, 11.0, 2], [58.0, 6, 5.0, 5], [21.0, 4, 11.0, 2],
                  [27.0, 4, 10.0, 2], [44.0, 4, 13.0, 2], [33.0, 4, 6.0, 4], [62.0, 6, 13.0, 2],
                  [20.0, 4, 9.0, 4], [33.0, 4, 9.0, 0]]

    data.x = DataFrame(dummy_data, columns=['Age', 'Workclass', 'Education-Num', 'Marital Status'])
    data.y_bin = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
    data.y_cont = [0.2, 1.1, 1.3, 0.23, 0.24, 0.9, -0.12, 1.1, 0.8, -0.001]
    return data


@pytest.fixture(scope='session')
def model(data):
    model, model.is_classification = get_trained_model_object(data.x, data.y_bin, model_type='rf')
    return model


@pytest.fixture(scope='session')
def exp_wrapper_shap(data, model):
    mode = ['regression', 'classification'][1 * model.is_classification]
    exp = ShapWrapper(model,
                      mode=mode,
                      shap_type='tree',
                      link='',
                      multiclass=False,
                      train_data=data.x.as_matrix(),
                      class_names=['yes', 'no'],
                      feature_names=list(data.x.columns),
                      num_features=10,
                      verbose=False)
    return exp


@pytest.fixture(scope='session')
def exp_wrapper_lime(data, model):
    mode = ['regression', 'classification'][1 * model.is_classification]
    exp = LimeWrapper(model,
                      mode=mode,
                      lime_type='tabular',
                      multiclass=False,
                      train_data=data.x.as_matrix(),
                      class_names=['yes', 'no'],
                      feature_names=list(data.x.columns),
                      num_samples=10,
                      num_features=10,
                      verbose=False)
    return exp


def test_lime_wrapper(data, exp_wrapper_lime):
    att = exp_wrapper_lime(data.x.as_matrix()[:10], y=None, show_plot=True)
    assert str(type(att)) == "<class 'numpy.ndarray'>"
    assert att.shape == (10, 4)
    assert str(type(att[0, 0])) == "<class 'numpy.float64'>"
    assert -1 < att[0, 0] < 5
    assert att.mean(axis=1).mean().round(2) == 0.01


def test_shap_wrapper(data, exp_wrapper_shap):
    att = exp_wrapper_shap(data.x.as_matrix()[:10], y=None, show_plot=True)
    assert str(type(att)) == "<class 'numpy.ndarray'>"
    assert att.shape == (10, 4)
    assert str(type(att[0, 0])) == "<class 'numpy.float64'>"
    assert 0 < att[0, 0] < 5
    assert att.mean(axis=1).mean().round(2) == 0.01


def test_estimate_dataset_lipschitz_for_lime_with_gbrt(data, exp_wrapper_lime):
    ts = time()
    lips = exp_wrapper_lime.estimate_dataset_lipschitz(dataset=data.x.as_matrix(),
                                                       continuous=True,
                                                       eps=1,
                                                       maxpoints=None,
                                                       optim='gbrt',
                                                       bound_type='box',
                                                       n_jobs=1,
                                                       n_calls=10,
                                                       verbose=False)
    print("Time elapsed for LIME with Gradient Boosted Tree opt: {} s".format(time() - ts))
    assert str(type(lips)) == "<class 'numpy.ndarray'>"
    assert 0.01 < lips[0] < 1
    assert lips.size == 10
    mean_lip = lips.mean()
    print("Mean Lipschitz estimate for Tabular LIME = {:.3}".format(mean_lip))
    assert 0 < mean_lip < 0.2


def test_estimate_dataset_lipschitz_for_shap_with_gbrt(data, exp_wrapper_shap):
    ts = time()
    lips = exp_wrapper_shap.estimate_dataset_lipschitz(dataset=data.x.as_matrix(),
                                                       continuous=True,
                                                       eps=1,
                                                       maxpoints=None,
                                                       optim='gbrt', bound_type='box', n_jobs=1,
                                                       n_calls=10, verbose=False)
    print("Time elapsed for SHAP with Gradient Boosted Tree opt {} s".format(time() - ts))
    assert str(type(lips)) == "<class 'numpy.ndarray'>"
    assert 0.01 < lips[0] < 1
    assert lips.size == 10
    mean_lip = lips.mean()
    print("Mean Lipschitz estimate for TreeSHAP = {:.3}".format(mean_lip))
    assert 0 < mean_lip < 0.2


def test_estimate_dataset_lipschitz_for_lime_with_gp(data, exp_wrapper_lime):
    ts = time()
    lips = exp_wrapper_lime.estimate_dataset_lipschitz(dataset=data.x.as_matrix(), continuous=True,
                                                       eps=1, maxpoints=None,
                                                       optim='gp', bound_type='box', n_jobs=1,
                                                       n_calls=10, verbose=False)
    print("Time elapsed for LIME with Bayesian Opt: {} s".format(time() - ts))
    assert str(type(lips)) == "<class 'numpy.ndarray'>"
    assert 0.01 < lips[0] < 1
    assert lips.size == 10
    mean_lip = lips.mean()
    print("Mean Lipschitz estimate for Tabular LIME = {:.3}".format(mean_lip))
    assert 0 < mean_lip < 0.2


def test_estimate_dataset_lipschitz_for_shap_with_gp(data, exp_wrapper_shap):
    ts = time()
    lips = exp_wrapper_shap.estimate_dataset_lipschitz(dataset=data.x.as_matrix(), continuous=True,
                                                       eps=1, maxpoints=None,
                                                       optim='gp', bound_type='box', n_jobs=1,
                                                       n_calls=10, verbose=False)
    print("Time elapsed for SHAP with Bayesian Opt: {} s".format(time() - ts))
    assert str(type(lips)) == "<class 'numpy.ndarray'>"
    assert 0.01 < lips[0] < 1
    assert lips.size == 10
    mean_lip = lips.mean()
    print("Mean Lipschitz estimate for TreeSHAP = {:.3}".format(mean_lip))
    assert 0 < mean_lip < 0.2
