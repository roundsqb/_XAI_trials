from xai.performance_benchmarking.benchmarking_pipelines import timed_benchmarking_pipeline
import os
prefix = '../'

# SHAP: Regression  & Classification
timed_benchmarking_pipeline(model_type='rf', explainer_type='shap', num_samples=10,
                            data_file_name=os.path.join(prefix, 'datasets', 'boston_dataset.csv'), target_name='y')
timed_benchmarking_pipeline(model_type='rf', explainer_type='shap', num_samples=10,
                            data_file_name=os.path.join(prefix, 'datasets', 'census_dataset.csv'), target_name='y')
timed_benchmarking_pipeline(model_type='xgboost', explainer_type='shap', num_samples=10,
                            data_file_name=os.path.join(prefix, 'datasets', 'boston_dataset.csv'), target_name='y')
timed_benchmarking_pipeline(model_type='xgboost', explainer_type='shap', num_samples=10,
                            data_file_name=os.path.join(prefix, 'datasets', 'census_dataset.csv'), target_name='y')


# LIME: Regression & Classification
timed_benchmarking_pipeline(model_type='rf', explainer_type='lime', num_samples=3,
                            data_file_name=os.path.join(prefix, 'datasets', 'boston_dataset.csv'), target_name='y')
timed_benchmarking_pipeline(model_type='rf', explainer_type='lime', num_samples=3,
                            data_file_name=os.path.join(prefix, 'datasets', 'census_dataset.csv'), target_name='y')


# LIME: with and without num_features and num_neighbours specified
timed_benchmarking_pipeline(model_type='rf', explainer_type='lime', num_samples=10, num_features=100,
                            num_neighbours=500,
                            data_file_name=os.path.join(prefix, 'datasets', 'census_dataset.csv'), target_name='y')

timed_benchmarking_pipeline(model_type='rf', explainer_type='lime', num_samples=10, num_features=None,
                            num_neighbours=None,
                            data_file_name=os.path.join(prefix, 'datasets', 'census_dataset.csv'), target_name='y')
