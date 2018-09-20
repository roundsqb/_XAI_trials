import pytest
from xai.explainers.explainer_functions import shap_explain_instance
from xai.utils.model_utils import get_trained_model_object
from xai.explainers.text_explanations import write_description
from pandas import DataFrame


@pytest.fixture(scope="session")
def data():
    dummy_data = [[33.0, 2, 13.0, 4], [36.0, 4, 11.0, 2], [58.0, 6, 5.0, 5], [21.0, 4, 11.0, 2],
                  [27.0, 4, 10.0, 2], [44.0, 4, 13.0, 2], [33.0, 4, 6.0, 4], [62.0, 6, 13.0, 2],
                  [20.0, 4, 9.0, 4], [33.0, 4, 9.0, 0]]

    data.x = DataFrame(dummy_data, columns=['Age', 'Workclass', 'Education-Num', 'Marital Status'])
    data.y_bin = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
    return data


@pytest.fixture(scope='session')
def sample_exp_array(data):
    model, is_classification = get_trained_model_object(data.x, data.y_bin, model_type='rf')
    return shap_explain_instance(model, sample=data.x.iloc[0, :])


@pytest.fixture(scope='session')
def lookup_table(data):
    lookup_table = DataFrame(columns=['alias', 'description', 'source name', 'unit', 'tolerance'],
                             index=data.x.columns)
    lookup_table.loc[:, 'alias'] = data.x.columns
    lookup_table.loc[:, 'description'] = data.x.columns
    lookup_table.loc['Education-Num', 'description'] = 'Time in education'

    lookup_table.loc[['Workclass', 'Education-Num'], 'source name'] = 'occupational data'
    lookup_table.loc[['Age', 'Marital Status', 'Education-Num'], 'source name'] \
        = 'self-reported protected attributes'

    # Leave unit as '', or ' ' to indicate nominal category.
    # If binary indicator, specify unit as 'flag'
    lookup_table.loc[:, 'unit'] = ''
    lookup_table.loc[['Education-Num', 'Age'], 'unit'] = 'years'

    lookup_table.loc[:, 'tolerance'] = 0
    lookup_table.loc[['Education-Num', 'Age'], 'tolerance'] = None
    return lookup_table


def test_write_description(sample_exp_array, data, lookup_table):
    sample = data.x.iloc[0, :]
    text = write_description(sample_exp_array, sample, lookup_table, pos_class_threshold=0.1,
                             top_n=10, model_name='RF classifier',
                             target_name='likelihood of earning over $50k')
    assert text.startswith("The dominant reason why Client #0 was predicted by the RF classifier "
                           "model as having a high likelihood of earning over $50k, was because "
                           "their Marital Status was 4")
    assert text.endswith("reason why Client #0 was not assigned even higher likelihood of earning "
                         "over $50k, was because their Age was 33 years.")

    text = write_description(sample_exp_array, sample, lookup_table, pos_class_threshold=0.9,
                             top_n=10, model_name='', target_name='')
    assert text.startswith("The dominant reason why Client #0 was predicted by the  "
                           "model as having a low")
