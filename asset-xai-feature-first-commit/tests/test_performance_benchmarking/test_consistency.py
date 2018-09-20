import pytest
from xai.performance_benchmarking.consistency import evaluate_consistency_between_shap_and_lime
from xai.explainers.explainer_functions import lime_explain_instance, instantiate_lime_explainer, \
    shap_explain_instance
from pandas import DataFrame
from xai.utils.model_utils import get_trained_model_object
from numpy import isnan


@pytest.fixture(scope="session")
def data():
    dummy_data = [[33.0, 2, 13.0, 4], [36.0, 4, 11.0, 2], [58.0, 6, 5.0, 5], [21.0, 4, 11.0, 2],
                  [27.0, 4, 10.0, 2], [44.0, 4, 13.0, 2], [33.0, 4, 6.0, 4], [62.0, 6, 13.0, 2],
                  [20.0, 4, 9.0, 4], [33.0, 4, 9.0, 0]]

    data.x = DataFrame(dummy_data, columns=['Age', 'Workclass', 'Education-Num', 'Marital Status'])
    data.y_bin = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
    return data


@pytest.fixture(scope='session')
def model(data):
    model, is_classification = get_trained_model_object(data.x, data.y_bin, model_type='rf')
    return model


@pytest.fixture(scope='session')
def lime_sample_exp_obj(data, model):
    explainer = instantiate_lime_explainer(data.x, categorical_idx=[0, 1, 10],
                                           is_classification=True, random_state=42)
    lime_exp_obj = lime_explain_instance(model, explainer, sample=data.x.iloc[0, :],
                                         num_features=None, num_neighbours=None, print_flag=True)
    return lime_exp_obj


@pytest.fixture(scope='session')
def shap_values_array(data, model):
    sample_exp_array = shap_explain_instance(model, sample=data.x.iloc[0, :])
    return sample_exp_array


def test_evaluate_consistency_between_shap_and_lime(shap_values_array, lime_sample_exp_obj):
    tau = evaluate_consistency_between_shap_and_lime(shap_values_array, lime_sample_exp_obj,
                                                     num_top_features=10)
    assert tau == 1.00
    tau = evaluate_consistency_between_shap_and_lime(shap_values_array, lime_sample_exp_obj,
                                                     num_top_features=100)
    assert tau == 1.00

    different_shap_array = [-1 for x in shap_values_array]
    tau = evaluate_consistency_between_shap_and_lime(different_shap_array, lime_sample_exp_obj,
                                                     num_top_features=10)
    assert tau == 0.667

    tau = evaluate_consistency_between_shap_and_lime(different_shap_array, lime_sample_exp_obj,
                                                     num_top_features=1)
    assert isnan(tau)
