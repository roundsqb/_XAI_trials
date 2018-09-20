from xai.performance_benchmarking.compute_time import compute_time_vs_lime_params, compute_time_vs_shap
import os
data_file_name = os.path.join('../', 'datasets', 'census_dataset.csv')
target_name = 'y'


tdeltas = compute_time_vs_shap(data_file_name, target_name, num_samples_list=[10, 100, 1000, 10000])
print(tdeltas)


tdeltas = compute_time_vs_lime_params(data_file_name, target_name,
                                      num_neighbours_list=[10, 100, 500, 1000],
                                      num_features_list=[20],
                                      num_samples=10)
print(tdeltas)
