from xai.explainers.explainer_functions import *
from xai.utils.model_utils import get_trained_model_object
from numpy import round
import pytest
from pandas import DataFrame


@pytest.fixture(scope="session")
def data():
    dummy_data = [[33.0, 2, 13.0, 4], [36.0, 4, 11.0, 2], [58.0, 6, 5.0, 5], [21.0, 4, 11.0, 2],
                  [27.0, 4, 10.0, 2], [44.0, 4, 13.0, 2], [33.0, 4, 6.0, 4], [62.0, 6, 13.0, 2],
                  [20.0, 4, 9.0, 4], [33.0, 4, 9.0, 0]]

    data.x = DataFrame(dummy_data, columns=['Age', 'Workclass', 'Education-Num', 'Marital Status'])
    data.y_bin = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
    data.y_cont = [0.1, 2.2, 0.34, -1.1, 12, 4.3, 0.5, 4, -2.3, -2]
    return data


@pytest.fixture(scope="session")
def explainer(data):
    explainer = instantiate_lime_explainer(data.x, categorical_idx=[0, 1, 10],
                                           is_classification=True, random_state=42)
    return explainer


@pytest.fixture(scope="session")
def model_crf(data):
    model, is_classification = get_trained_model_object(data.x, data.y_bin, model_type='rf')
    return model


@pytest.fixture(scope='session')
def sample_exp(model_crf, explainer, data):
    sample_exp = lime_explain_instance(model_crf, explainer, sample=data.x.iloc[0, :],
                                       num_features=None, num_neighbours=None, print_flag=True)
    return sample_exp


def test_instantiate_lime_explainer(data):
    explainer = instantiate_lime_explainer(data.x, categorical_idx=[0, 1, 2],
                                           is_classification=True, random_state=42)
    assert str(type(explainer)) == "<class 'lime.lime_tabular.LimeTabularExplainer'>"
    assert (explainer.feature_names == data.x.columns).all()
    assert explainer.mode == 'classification'
    explainer = instantiate_lime_explainer(data.x, categorical_idx=[0, 1, 2],
                                           is_classification=False, random_state=42)
    assert explainer.mode == 'regression'


def test_lime_explain_instance(model_crf, explainer, data):
    sample_exp = lime_explain_instance(model_crf, explainer, sample=data.x.iloc[0, :])
    assert str(type(sample_exp)) == "<class 'lime.explanation.Explanation'>"
    assert len(sample_exp.local_exp[1]) == 4

    sample_exp = lime_explain_instance(model_crf, explainer, sample=data.x.iloc[0, :],
                                       num_features=100, num_neighbours=None, print_flag=True)
    assert len(sample_exp.local_exp[1]) == 4


def test_lime_sample_exp_to_array(sample_exp):
    sample_exp_array = lime_sample_exp_to_array(sample_exp)
    assert str(type(sample_exp_array)) == "<class 'numpy.ndarray'>"
    assert len(sample_exp_array) == 5


def test_lime_get_ft_idx(sample_exp):
    ft_idx = lime_get_ft_idx(sample_exp)
    assert str(type(ft_idx)) == "<class 'list'>"
    assert len(ft_idx) == 4
    assert ft_idx[3] == 3


def test_shap_explain_instance(model_crf, data):
    sample_exp_array = shap_explain_instance(model_crf, sample=data.x.iloc[0, :])
    assert str(type(sample_exp_array)) == "<class 'numpy.ndarray'>"
    assert len(sample_exp_array) == 5
    assert round(sample_exp_array[0], 5) == -0.28512
    assert round(sample_exp_array[-1], 5) == 0.52


def test_lime_explain_multiple_instances(model_crf, explainer, data):
    explanations, samples_explained = lime_explain_multiple_instances(
        model_crf, explainer, data.x,num_samples=2,num_features=None, num_neighbours=None)
    assert samples_explained.shape == (2, 4)
    assert str(type(explanations)) == "<class 'numpy.ndarray'>"
    assert explanations.shape == (2, 5)


def test_lime_explain_multiple_instances_with_num_features_above_max_available(
        model_crf, explainer, data):
    explanations, samples_explained = lime_explain_multiple_instances(
        model_crf, explainer, data.x, num_samples=2, num_features=1000, num_neighbours=None)
    assert samples_explained.shape == (2, 4)
    assert explanations.shape == (2, 5)


def test_lime_explain_multiple_instances_with_limited_num_features(model_crf, explainer, data):
    explanations, samples_explained = lime_explain_multiple_instances(
        model_crf, explainer, data.x, num_samples=2, num_features=2, num_neighbours=None)
    assert samples_explained.shape == (2, 4)
    assert explanations.shape == (2, 5)
    assert round(explanations[0, 2], 5) == -0.00000


def test_assertions_in_explain_multiple_instances(model_crf, explainer, data):
    with pytest.raises(AssertionError, match='predict'):
        lime_explain_multiple_instances(model=explainer, explainer=explainer, x=data.x)


def test_explain_model_shap(model_crf, data):
    explanations, samples_explained = explain_model(model_crf, data.x, categorical_idx=[0, 1, 2],
                                                    is_classification=True,
                                                    explainer_type='shap',
                                                    num_samples=2)
    assert samples_explained.shape == (2, 4)
    assert str(type(explanations)) == "<class 'numpy.ndarray'>"
    assert explanations.shape == (2, 5)
    assert round(explanations[0, 0], 5) == 0.0982

    explanations, samples_explained = explain_model(model_crf, data.x, categorical_idx=[0, 1, 2],
                                                    is_classification=True,
                                                    explainer_type='shap',
                                                    num_samples=None)
    assert samples_explained.shape == (10, 4)
    assert str(type(explanations)) == "<class 'numpy.ndarray'>"
    assert explanations.shape == (10, 5)
    assert round(explanations[0, 0], 5) == -0.28512

    explanations, samples_explained = explain_model(model_crf, data.x, categorical_idx=[0, 1, 2],
                                                    is_classification=True,
                                                    explainer_type='shap',
                                                    num_samples=2, num_features=3)
    # num_features should have no effect on SHAP explanations
    assert samples_explained.shape == (2, 4)
    assert explanations.shape == (2, 5)


def test_explain_model_lime(model_crf, data):
    explanations, samples_explained = explain_model(model_crf, data.x, categorical_idx=[0, 1, 2],
                                                    is_classification=True,
                                                    explainer_type='lime',
                                                    num_samples=2)
    assert samples_explained.shape == (2, 4)
    assert str(type(explanations)) == "<class 'numpy.ndarray'>"
    assert explanations.shape == (2, 5)

    explanations, samples_explained = explain_model(model_crf, data.x, categorical_idx=[0, 1, 2],
                                                    is_classification=True,
                                                    explainer_type='lime',
                                                    num_samples=None)
    assert samples_explained.shape == (10, 4)
    assert str(type(explanations)) == "<class 'numpy.ndarray'>"
    assert explanations.shape == (10, 5)

    explanations, samples_explained = explain_model(model_crf, data.x, categorical_idx=[0, 1, 2],
                                                    is_classification=True,
                                                    explainer_type='lime',
                                                    num_samples=2, num_features=2)
    # num_features should make LIME features 0
    assert samples_explained.shape == (2, 4)
    assert explanations.shape == (2, 5)
    assert round(explanations[0, 2], 5) == 0.00000


def test_assertions_in_explain_model(model_crf, data):
    with pytest.raises(AssertionError, match='predict'):
        lime_explain_multiple_instances(model=explainer, explainer=explainer, x=data.x)

    with pytest.raises(AssertionError, match='currently supported explanation models are'):
        explain_model(model_crf, data.x, categorical_idx=[0, 1, 2], explainer_type='shOp',
                      num_samples=2,
                      is_classification=True)


# Tests assessing funtionality for regression models #


@pytest.fixture(scope="session")
def explainer_reg(data):
    explainer = instantiate_lime_explainer(data.x, categorical_idx=[0, 1, 10],
                                           is_classification=False, random_state=42)
    return explainer


@pytest.fixture(scope="session")
def model_rrf(data):
    model, is_classification = get_trained_model_object(data.x, data.y_cont, model_type='rf')
    return model


@pytest.fixture(scope='session')
def sample_exp_reg(model_rrf, explainer_reg, data):
    sample_exp = lime_explain_instance(model_rrf, explainer_reg, sample=data.x.iloc[0, :],
                                       num_features=None, num_neighbours=None, print_flag=True)
    return sample_exp


def test_lime_sample_exp_to_array_reg(sample_exp_reg):
    sample_exp_array = lime_sample_exp_to_array(sample_exp_reg)
    assert str(type(sample_exp_array)) == "<class 'numpy.ndarray'>"
    assert len(sample_exp_array) == 5


def test_shap_explain_instance_regression(model_rrf, data):
    sample_exp_array = shap_explain_instance(model_rrf, sample=data.x.iloc[0, :],
                                             is_classification=False,
                                             print_flag=False)
    assert str(type(sample_exp_array)) == "<class 'numpy.ndarray'>"
    assert len(sample_exp_array) == 5
    assert round(sample_exp_array[0], 5) == -0.69588
    assert round(sample_exp_array[-1], 5) == 1.72874


def test_lime_explain_instance_regression(model_rrf, explainer_reg, data):
    assert str(type(model_rrf)) == "<class 'sklearn.ensemble.forest.RandomForestRegressor'>"
    sample_exp = lime_explain_instance(model_rrf, explainer_reg, sample=data.x.iloc[0, :],
                                       print_flag=True)
    assert str(type(sample_exp)) == "<class 'lime.explanation.Explanation'>"
    assert round(sample_exp.predicted_value, 2) == 1.07
    assert len(sample_exp.local_exp[1]) == 4


def test_explain_model_shap_regression(model_rrf, data):
    explanations, samples_explained = explain_model(model_rrf, data.x, categorical_idx=[0, 1, 2],
                                                    explainer_type='shap',
                                                    num_samples=2, is_classification=False)
    assert samples_explained.shape == (2, 4)
    assert str(type(explanations)) == "<class 'numpy.ndarray'>"
    assert explanations.shape == (2, 5)
    assert round(explanations[0, 0], 5) == -0.54646

    explanations, samples_explained = explain_model(model_rrf, data.x, categorical_idx=[0, 1, 2],
                                                    explainer_type='shap',
                                                    num_samples=None, is_classification=False)
    assert samples_explained.shape == (10, 4)
    assert str(type(explanations)) == "<class 'numpy.ndarray'>"
    assert explanations.shape == (10, 5)
    assert round(explanations[0, 0], 5) == -0.69588

    explanations, samples_explained = explain_model(model_rrf, data.x, categorical_idx=[0, 1, 2],
                                                    explainer_type='shap',
                                                    num_samples=2, num_features=3,
                                                    is_classification=False)
    # num_features should have no effect on SHAP explanations
    assert samples_explained.shape == (2, 4)
    assert explanations.shape == (2, 5)


def test_explain_model_lime_regression(model_rrf, data):
    explanations, samples_explained = explain_model(model_rrf, data.x, categorical_idx=[0, 1, 2],
                                                    explainer_type='lime',
                                                    num_samples=2, is_classification=False)
    assert samples_explained.shape == (2, 4)
    assert str(type(explanations)) == "<class 'numpy.ndarray'>"
    assert explanations.shape == (2, 5)

    explanations, samples_explained = explain_model(model_rrf, data.x, categorical_idx=[0, 1, 2],
                                                    explainer_type='lime',
                                                    num_samples=None, is_classification=False)
    assert samples_explained.shape == (10, 4)
    assert str(type(explanations)) == "<class 'numpy.ndarray'>"
    assert explanations.shape == (10, 5)

    explanations, samples_explained = explain_model(model_rrf, data.x, categorical_idx=[0, 1, 2],
                                                    explainer_type='lime',
                                                    num_samples=2, num_features=2,
                                                    is_classification=False)
    # num_features should make LIME features 0
    assert samples_explained.shape == (2, 4)
    assert explanations.shape == (2, 5)


# Testing XGBOOST model support

@pytest.fixture(scope="session")
def model_cxgb(data):
    model, is_classification = get_trained_model_object(data.x, data.y_bin, model_type='xgboost')
    return model


@pytest.fixture(scope="session")
def model_rxgb(data):
    model, is_classification = get_trained_model_object(data.x, data.y_cont, model_type='xgboost')
    return model


def test_explain_model_shap_regression_xgboost(model_rxgb, data):
    explanations, samples_explained = explain_model(model_rxgb, data.x, categorical_idx=[0, 1, 2],
                                                    explainer_type='shap',
                                                    num_samples=2, is_classification=False)
    assert samples_explained.shape == (2, 4)
    assert str(type(explanations)) == "<class 'numpy.ndarray'>"
    assert explanations.shape == (2, 5)
    assert round(float(explanations[0, 4]), 3) == 1.056

    explanations, samples_explained = explain_model(model_rxgb, data.x, categorical_idx=[0, 1, 2],
                                                    explainer_type='shap',
                                                    num_samples=None, is_classification=False)
    assert samples_explained.shape == (10, 4)
    assert str(type(explanations)) == "<class 'numpy.ndarray'>"
    assert explanations.shape == (10, 5)
    assert round(float(explanations[0, 4]), 3) == 1.056

    explanations, samples_explained = explain_model(model_rxgb, data.x, categorical_idx=[0, 1, 2],
                                                    explainer_type='shap',
                                                    num_samples=2, num_features=2,
                                                    is_classification=False)
    # num_features should have no effect on SHAP explanations
    assert samples_explained.shape == (2, 4)
    assert explanations.shape == (2, 5)
    assert round(float(explanations[0, 4]), 3) == 1.056


def test_explain_model_shap_class_xgboost(model_cxgb, data):
    explanations, samples_explained = explain_model(model_cxgb, data.x, categorical_idx=[0, 1, 2],
                                                    is_classification=True,
                                                    explainer_type='shap',
                                                    num_samples=2)
    assert samples_explained.shape == (2, 4)
    assert str(type(explanations)) == "<class 'numpy.ndarray'>"
    assert explanations.shape == (2, 5)
    assert round(float(explanations[0, 0]), 3) == -0.007

    explanations, samples_explained = explain_model(model_cxgb, data.x, categorical_idx=[0, 1, 2],
                                                    is_classification=True,
                                                    explainer_type='shap',
                                                    num_samples=None)
    assert samples_explained.shape == (10, 4)
    assert str(type(explanations)) == "<class 'numpy.ndarray'>"
    assert explanations.shape == (10, 5)
    assert round(float(explanations[0, 0]), 3) == -0.007

    explanations, samples_explained = explain_model(model_cxgb, data.x, categorical_idx=[0, 1, 2],
                                                    is_classification=True,
                                                    explainer_type='shap',
                                                    num_samples=2, num_features=2)
    # num_features should have no effect on SHAP explanations
    assert samples_explained.shape == (2, 4)
    assert explanations.shape == (2, 5)