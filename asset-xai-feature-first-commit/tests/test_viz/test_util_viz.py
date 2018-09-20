import pytest
from xai.viz.util_viz import plot_corr_heatmap, plot_calibration_curve, bar_plot, \
    plot_rf_feature_importances
from xai.utils.model_utils import get_trained_model_object
from xai.performance_benchmarking.benchmarking_pipelines import timed_benchmarking_pipeline
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
def xgboost_class(data):
    model, is_classification = get_trained_model_object(data.x, data.y_bin, model_type='xgboost')
    return model


@pytest.fixture(scope="session")
def model_class(data):
    model, is_classification = get_trained_model_object(data.x, data.y_bin, model_type='rf')
    return model


@pytest.fixture(scope='session')
def shap_explanations():
    explanations, samples_explained, model, tdelta = timed_benchmarking_pipeline(
        model_type='rf', explainer_type='shap', data_file_name='census', target_name=None,
        num_samples=None, num_features=None, num_neighbours=None)
    return explanations, samples_explained


def test_plot_corr_heatmap(data):
    fig = plot_corr_heatmap(data.x)
    # fig.savefig("Test_Correlation_Heatmap_Plot.png")
    assert str(type(fig)) == "<class 'matplotlib.figure.Figure'>"
    assert len(fig.axes) == 2


def test_plot_calibration_curve(data, model_class, xgboost_class):
    fig = plot_calibration_curve(data.x, data.y_bin, models=[model_class, xgboost_class],
                                 model_names=["RF", "XGBOOST"])
    # fig.savefig("Test_Calibration_Plot.png")
    assert str(type(fig)) == "<class 'matplotlib.figure.Figure'>"
    assert len(fig.axes) == 1
    fig = plot_calibration_curve(data.x, data.y_bin, models=[model_class, xgboost_class],
                                 model_names=None)
    # fig.savefig("Test_Calibration_Plot_without_Model_Names.png")
    assert str(type(fig)) == "<class 'matplotlib.figure.Figure'>"
    assert len(fig.axes) == 1


def test_bar_plot(shap_explanations):
    explanations, samples_explained = shap_explanations
    explanation = explanations[0, :]
    sample = samples_explained.iloc[0, :]
    fig = bar_plot(explanation, sample, cmap_style=None)
    assert str(type(fig)) == "<class 'matplotlib.figure.Figure'>"
    assert len(fig.axes) == 1

    fig = bar_plot(explanation, sample, cmap_style='lime')
    assert str(type(fig)) == "<class 'matplotlib.figure.Figure'>"
    assert len(fig.axes) == 1

    fig = bar_plot(explanation, sample, cmap_style='shop')
    assert str(type(fig)) == "<class 'matplotlib.figure.Figure'>"
    assert len(fig.axes) == 1


def test_plot_rf_feature_importances(model_class, data):
    fig = plot_rf_feature_importances(model_class, data.x)
    assert str(type(fig)) == "<class 'matplotlib.figure.Figure'>"
    assert len(fig.axes) == 1
