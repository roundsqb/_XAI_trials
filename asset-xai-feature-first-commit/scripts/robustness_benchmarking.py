from xai.performance_benchmarking.benchmarking_pipelines import robustness_benchmarking_pipeline
from xai.utils.data_loading_utils import load_data_from_csv
from xai.viz.util_viz import plot_robustness_comparisons
from xai.utils.model_utils import generate_trained_model
import os

prefix = '../'

# Classification - Census, SHAP & LIME
data = os.path.join(prefix, 'datasets', 'census_dataset.csv')
x, x_test, y, y_test, categorical_idx = load_data_from_csv(file_name=data, target_name='y',
                                                           max_levels=100, test_size=0.2, skiprows=False,
                                                           multiples_of_rows_to_skip=100)
model, is_classification = generate_trained_model('rf', x, x_test, y, y_test)

lips1, tdelta1 = robustness_benchmarking_pipeline(model, x_test, explainer_type='shap', num_samples=100,
                                                  lipshitz_optimiser='gbrt', lipshitz_bound_type='box', n_jobs=1,
                                                  random_seed=42)

lips2, tdelta2 = robustness_benchmarking_pipeline(model, x_test, explainer_type='lime', num_samples=100,
                                                  lipshitz_optimiser='gbrt', lipshitz_bound_type='box', n_jobs=1,
                                                  random_seed=42)

lips12, tdelta12 = robustness_benchmarking_pipeline(model, x_test, explainer_type='shap', num_samples=100,
                                                    lipshitz_optimiser='gp', lipshitz_bound_type='box', n_jobs=1,
                                                    random_seed=42)

lips22, tdelta22 = robustness_benchmarking_pipeline(model, x_test, explainer_type='lime', num_samples=100,
                                                    lipshitz_optimiser='gp', lipshitz_bound_type='box', n_jobs=1,
                                                    random_seed=42)

fig = plot_robustness_comparisons([lips1, lips12, lips2, lips22], dataset_name='Census',
                                  model_names=['SHAP gbrt', 'SHAP gp', 'LIME gbrt', 'LIME gp'])

fig.savefig(fname='Robustness_comparison_on_Census.png')

print([tdelta1, tdelta12, tdelta2, tdelta22])

# Regression -  Boston, SHAP & LIME

data = os.path.join(prefix, 'datasets', 'boston_dataset.csv')
lips1, tdelta1 = robustness_benchmarking_pipeline(model, x_test, explainer_type='shap', num_samples=100,
                                                  lipshitz_optimiser='gbrt', lipshitz_bound_type='box', n_jobs=1,
                                                  random_seed=42)

lips2, tdelta2 = robustness_benchmarking_pipeline(model, x_test, explainer_type='lime', num_samples=100,
                                                  lipshitz_optimiser='gbrt', lipshitz_bound_type='box', n_jobs=1,
                                                  random_seed=42)

lips12, tdelta12 = robustness_benchmarking_pipeline(model, x_test, explainer_type='shap', num_samples=100,
                                                    lipshitz_optimiser='gp', lipshitz_bound_type='box', n_jobs=1,
                                                    random_seed=42)

lips22, tdelta22 = robustness_benchmarking_pipeline(model, x_test, explainer_type='lime', num_samples=100,
                                                    lipshitz_optimiser='gp', lipshitz_bound_type='box', n_jobs=1,
                                                    random_seed=42)

fig = plot_robustness_comparisons([lips1, lips12, lips2, lips22], dataset_name='Boston',
                                  model_names=['SHAP gbrt', 'SHAP gp', 'LIME gbrt', 'LIME gp'])
fig.savefig(fname='Robustness_comparison_on_Boston.png')
print([tdelta1, tdelta12, tdelta2, tdelta22])
