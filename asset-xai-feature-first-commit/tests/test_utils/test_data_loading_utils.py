from xai.utils.data_loading_utils import load_data_from_csv, load_data_census, load_data_boston
from pandas import DataFrame
import pytest


@pytest.fixture(scope='session')
def csv_file(tmpdir_factory):
    dummy_data = [[33.0, 2, 13.0, 4], [36.0, 4, 11.0, 2], [58.0, 6, 5.0, 5], [21.0, 4, 11.0, 2],
                  [27.0, 4, 10.0, 2], [44.0, 4, 13.0, 2], [33.0, 4, 6.0, 4], [62.0, 6, 13.0, 2],
                  [20.0, 4, 9.0, 4], [33.0, 4, 9.0, 0], [24.0, 4, 13.0, 4], [36.0, 4, 15.0, 2],
                  [35.0, 4, 15.0, 2]]

    csv_file.content = DataFrame(dummy_data,
                                 columns=['Age', 'Workclass', 'Education-Num', 'Marital Status'])
    csv_file.content['target'] = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1]
    csv_file.name = tmpdir_factory.mktemp("data").join("test_data.csv")
    csv_file.content.to_csv(csv_file.name, sep=',')
    csv_file.target_name = 'target'
    return csv_file


def test_load_data_from_csv(csv_file):
    x, x_test, y, y_test, categorical_idx = load_data_from_csv(csv_file.name, csv_file.target_name,
                                                               max_levels=100, test_size=0.2,
                                                               skiprows=False)
    assert x.shape[1] == 5, 'test failed: x must have 5 columns'
    assert x.shape[0] == len(y), 'test failed: x and y are not the same length'
    assert (
    x.columns == x_test.columns).all(), "test failed: x and x_test do not have the same columns"


def test_skiprows_in_load_data_from_csv(csv_file):
    x, x_test, y, y_test, categorical_idx = load_data_from_csv(csv_file.name, csv_file.target_name,
                                                               max_levels=100,
                                                               test_size=0.2, skiprows=True,
                                                               multiples_of_rows_to_skip=None)
    assert x.shape[0] == 0, 'test failed: number of samples must be 1'


def test_assertion_in_load_data_from_csv(csv_file):
    with pytest.raises(AssertionError, match='target_name'):
        load_data_from_csv(csv_file.name, csv_file.target_name.upper())
    with pytest.raises(FileNotFoundError):
        load_data_from_csv("HELLO.csv", csv_file.target_name)


def test_load_boston():
    x, x_test, y, y_test, categorical_idx = load_data_boston()
    assert x.shape[1] == 13, 'test failed: x must have 13 columns'
    assert x.shape[0] == len(y), 'test failed: x and y are not the same length'
    assert (
    x.columns == x_test.columns).all(), "test failed: x and x_test do not have the same columns"


def test_load_census():
    x, x_test, y, y_test, categorical_idx = load_data_census()
    assert x.shape[1] == 11, 'test failed: x must have 13 columns'
    assert x.shape[0] == len(y), 'test failed: x and y are not the same length'
    assert (
    x.columns == x_test.columns).all(), "test failed: x and x_test do not have the same columns"
