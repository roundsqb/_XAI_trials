import pytest
from xai.performance_benchmarking.benchmarking_pipelines import timed_benchmarking_pipeline
from xai.viz.modified_shap_plots import js_force_plot, set_colormap, summary_plot, dependence_plot
import os


@pytest.fixture(scope='session')
def shap_explanations():

    explanations, samples_explained, model, tdelta = timed_benchmarking_pipeline(
        model_type='rf', explainer_type='shap',
        data_file_name=os.path.join('datasets', 'census_dataset.csv'), target_name='y',
        num_samples=None, num_features=None, num_neighbours=None)
    return explanations[0:100, :], samples_explained.iloc[0:100, :]


def test_js_force_plot(shap_explanations):
    explanations, samples_explained = shap_explanations
    js_force_plot(explanations, samples_explained, cmap_style='shap')
    assert True


def test_js_force_plot_single_expl(shap_explanations):
    explanations, samples_explained = shap_explanations
    js_force_plot(explanations[0], samples_explained.iloc[0, :], cmap_style='lime')


def test_summary_plot(shap_explanations):
    explanations, samples_explained = shap_explanations
    summary_plot(explanations, samples_explained, max_display=10, cmap_style='lime', show=False)
    assert True


def test_dependence_plot(shap_explanations):
    explanations, samples_explained = shap_explanations
    dependence_plot('Age', explanations, samples_explained, interaction_index='Sex', show=False)
    assert True


def test_set_colormap():
    cmap_str, cmap, cmap_solid, default_colors = set_colormap(cmap_style='lime')
    assert cmap_str =='PkYg'
    cmap_str, cmap, cmap_solid, default_colors = set_colormap()
    assert cmap_str == 'RdBu'

