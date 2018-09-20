from xai.performance_benchmarking.benchmarking_pipelines import timed_benchmarking_pipeline
import os


def test_shap_regression():
    explanations, samples_explained, model, tdelta = timed_benchmarking_pipeline(
        model_type='rf', explainer_type='shap',
        data_file_name=os.path.join('datasets', 'boston_dataset.csv'), target_name='y',
        num_samples=3)
    assert explanations.shape == (3, 14)
    assert 'Regress' in str(type(model))


def test_shap_classification():
    explanations, samples_explained, model, tdelta = timed_benchmarking_pipeline(
        model_type='rf', explainer_type='shap',
        data_file_name=os.path.join('datasets', 'census_dataset.csv'), target_name='y',
        num_samples=3)
    assert explanations.shape == (3, 13)
    assert 'Classif' in str(type(model))


def test_lime_regression():
    explanations, samples_explained, model, tdelta = timed_benchmarking_pipeline(
        model_type='rf', explainer_type='lime',
        data_file_name=os.path.join('datasets', 'boston_dataset.csv'), target_name='y',
        num_samples=3)
    assert explanations.shape == (3, 14)
    assert 'Regress' in str(type(model))


def test_lime_classification():
    explanations, samples_explained, model, tdelta = timed_benchmarking_pipeline(
        model_type='rf', explainer_type='lime',
        data_file_name=os.path.join('datasets', 'census_dataset.csv'), target_name='y',
        num_samples=3)
    assert explanations.shape == (3, 13)
    assert 'Classif' in str(type(model))

