from pandas import DataFrame
from time import time

from ..explainers.explainer_functions import explain_model
from ..utils.data_loading_utils import load_data_from_csv
from ..utils.model_utils import generate_trained_model


def compute_time_vs_lime_params(data_file_name, target_name,
                                num_neighbours_list,
                                num_features_list,
                                num_samples):
    """
    Computes how long it takes to run n=num_samples LIME explanations for varying num_neighbours and
     num_features
    :param data_file_name:
    :param target_name:
    :param num_neighbours_list: list of integers specifying the num_neighbours to be used in LIME
    explainer
    :param num_features_list: list of integers specifying the num_features to be used in LIME
    explainer
    :param num_samples: integer specifying the number of samples to be explained
    :return:
           pandas DataFrame containing time it took to generate the explanations
    """
    x, x_test, y, y_test, categorical_idx = load_data_from_csv(
        file_name=data_file_name, target_name=target_name,
        max_levels=100, test_size=0.2, skiprows=False,
        multiples_of_rows_to_skip=100)
    model, is_classification = generate_trained_model('rf', x, x_test, y, y_test)
    tdeltas = DataFrame(index=num_features_list, columns=num_neighbours_list)

    for i, num_features in enumerate(num_features_list):
        for j, num_neighbours in enumerate(num_neighbours_list):
            ts = time()
            _, _ = explain_model(model, x, categorical_idx,
                                 is_classification=is_classification,
                                 explainer_type='lime',
                                 num_samples=num_samples,
                                 num_features=num_features,
                                 num_neighbours=num_neighbours)
            tdeltas.iloc[i, j] = time() - ts
            print("\nExperiment took {:.2f} s.".format(tdeltas.iloc[i, j]))
    return tdeltas


def compute_time_vs_shap(data_file_name, target_name, num_samples_list):
    """
    Computes how long it takes to run n=num_samples SHAP explanations for varying num_samples
    :param data_file_name:
    :param target_name:
    :param num_samples_list: integers specifying the number of samples to be explained
    :return:
           pandas DataFrame containing time it took to generate the explanations
    """
    x, x_test, y, y_test, categorical_idx = load_data_from_csv(file_name=data_file_name,
                                                               target_name=target_name,
                                                               max_levels=100, test_size=0.2,
                                                               skiprows=False,
                                                               multiples_of_rows_to_skip=100)
    model, is_classification = generate_trained_model('rf', x, x_test, y, y_test)
    tdeltas = DataFrame(columns=num_samples_list)

    for i, num_samples in enumerate(num_samples_list):
        ts = time()
        _, _ = explain_model(model, x, categorical_idx, explainer_type='shap',
                             num_samples=num_samples, is_classification=is_classification)
        tdeltas.loc[0, num_samples] = time() - ts
        print("\nExperiment took {:.2f} s.".format(tdeltas.loc[0, num_samples]))
    return tdeltas
