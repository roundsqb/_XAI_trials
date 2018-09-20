import pytest
from xai.utils.data_utils import find_categorical_idx, compute_vifs
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


def test_find_categorical(data):
    assert find_categorical_idx(data.x) == [1, 3]
    assert find_categorical_idx(data.x, max_levels=3) == [1]
    assert find_categorical_idx(data.x, max_levels=3, categorical_dtypes=['int16']) == []
    assert find_categorical_idx(data.x, max_levels=3, categorical_dtypes=None) == [1]


def test_compute_vifs(data):
    vif_df = compute_vifs(data.x)
    assert str(type(vif_df)) == "<class 'pandas.core.frame.DataFrame'>"
    assert round(vif_df.loc['Age', 'vif'], 2) == 2.52
