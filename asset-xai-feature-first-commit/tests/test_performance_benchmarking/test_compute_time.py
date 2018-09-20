import pytest
from xai.performance_benchmarking.compute_time import compute_time_vs_lime_params, \
    compute_time_vs_shap
from pandas import DataFrame


@pytest.fixture(scope='session')
def data_file(tmpdir_factory):
    dummy_data = [[33.0, 2, 13.0, 4], [36.0, 4, 11.0, 2], [58.0, 6, 5.0, 5], [21.0, 4, 11.0, 2],
                  [27.0, 4, 10.0, 2], [44.0, 4, 13.0, 2], [33.0, 4, 6.0, 4], [62.0, 6, 13.0, 2],
                  [20.0, 4, 9.0, 4], [33.0, 4, 9.0, 0], [24.0, 4, 13.0, 4], [36.0, 4, 15.0, 2],
                  [35.0, 4, 15.0, 2]]

    data_file.content = DataFrame(dummy_data,
                                  columns=['Age', 'Workclass', 'Education-Num', 'Marital Status'])
    data_file.content['target'] = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1]
    data_file.name = tmpdir_factory.mktemp("data").join("test_data.csv")
    data_file.content.to_csv(data_file.name, sep=',')
    data_file.target_name = 'target'
    return data_file


def test_compute_time_vs_lime_params(data_file):
    tdeltas = compute_time_vs_lime_params(data_file_name=data_file.name,
                                          target_name=data_file.target_name,
                                          num_neighbours_list=[10, 100],
                                          num_features_list=[2, 3],
                                          num_samples=2)
    assert all(tdeltas < 1)


def test_compute_time_vs_shap(data_file):
    tdeltas = compute_time_vs_shap(data_file_name=data_file.name,
                                   target_name=data_file.target_name,
                                   num_samples_list=[2, 10])
    assert all(tdeltas < 1)
