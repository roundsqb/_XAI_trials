from time import time
import os
from ..explainers.explainer_functions import explain_model
from ..utils.model_utils import generate_trained_model
from ..utils.data_loading_utils import load_data_from_csv
from .robustness import ShapWrapper, LimeWrapper
from numpy import median

path_prefix = "../../"


def timed_benchmarking_pipeline(model_type, explainer_type, data_file_name, target_name='y',
                                num_samples=None, num_features=None, num_neighbours=None):
    """
    Function for calling timed experiments with SHAP and LIME on sklearn RF and XGBOOST.
    Loads the dummy dataset specified in 'data_file_name' (boston dataset for regression, census for
    classification). Automatically determines the type of the model (classification or regression)
    from the input dataset type. Generates a model object, and runs the post-hoc explainer model on
    top to output the explanations for every sample.
    :param model_type: string, presently supported values are 'xgboost' or 'rf'
    :param data_file_name: string specifying the full path, file name and extension
    (must be .csv with a header)
    :param target_name:string specifying the name of the dependent variable
    (must be one of the columns in the header)
    :param explainer_type: string, presently supported values are 'lime' or 'shap'
    :param num_features: optional integer  - the number of features to be used in the explanations
    :param num_samples: integer specifying the number of samples to be explained. Default:
    :param num_neighbours: optional integer - the maximum number of samples to be explained
    :return: tuple comprising:
        explanations: array of arrays of floats
        x: input dataframe
        y: array of true values
        model: pre-trained sklearn model object
        tdelta: float time
    """
    if data_file_name == 'boston':
        data_file_name = os.path.join(path_prefix, 'datasets', 'boston_dataset.csv')
        target_name = 'y'

    if data_file_name == 'census':
        data_file_name = os.path.join(path_prefix, 'datasets', 'census_dataset.csv')
        target_name = 'y'

    x, x_test, y, y_test, categorical_idx = load_data_from_csv(
        file_name=data_file_name, target_name=target_name,
        max_levels=100, test_size=0.2, skiprows=False,
        multiples_of_rows_to_skip=100)
    model, is_classification = generate_trained_model(model_type, x, x_test, y, y_test)

    ts = time()
    explanations, samples_explained = explain_model(model, x_test, categorical_idx,
                                                    explainer_type=explainer_type,
                                                    num_samples=num_samples,
                                                    num_features=num_features,
                                                    num_neighbours=num_neighbours,
                                                    is_classification=is_classification)
    tdelta = time() - ts
    print("Experiments took {:.2f} s.".format(tdelta))
    return explanations, samples_explained, model, tdelta


def robustness_benchmarking_pipeline(model, x, explainer_type, is_classification=True,
                                     random_seed=42,
                                     lipshitz_optimiser='gbrt', lipshitz_bound_type='box', n_jobs=1,
                                     n_calls_optimiser=10, num_samples=None, num_features=None,
                                     num_neighbours=500):
    """
    Function for calling timed experiments for evaluating robustness of TreeSHAP and LIME on
    sklearn RF and XGBOOST. Accepts a trained model object and generates the post-hoc explainer
    model to output the explanations for x.
    Args:
        model:
        x:
        explainer_type: string, presently supported values are 'lime' or 'shap'
        is_classification:
        random_seed:
        lipshitz_optimiser:
        lipshitz_bound_type:
        n_jobs:
        n_calls_optimiser:
        num_samples: optional integer specifying the maximum number of samples to be explained.
        efault: num of rows in x.
        num_features: optional integer specifying the number of top features to be used in the
        explanations Default: number of columns in x.
        num_neighbours: optional integer specifying the neighbourhood size in LIME. Default: 500.

    Returns:
        tuple comprising:
           lips: array of Lipshitz estimates
           tdelta: float time

    """
    assert explainer_type in ['shap','lime'], \
        "Presently supported explainer types are 'lime' or 'shap."
    assert lipshitz_optimiser in ['gbrt','gp'], \
        "Lipshitz estmate is computed with either Gradient-Boosted Tree ('gbrt'), or " \
        "Bayesian Optimisation ('gp')"
    assert lipshitz_bound_type in ['box', 'box_sdt', 'box_norm'], \
        "Select Lipshitz bound type as either box, box_norm, or box_std"
    mode = ['regression', 'classification'][1 * is_classification]
    if explainer_type == 'shap':
        explainer = ShapWrapper(model,
                                mode=mode,
                                shap_type='tree',
                                link='',
                                multiclass=False,
                                train_data=x.as_matrix(),
                                class_names=['yes', 'no'],
                                feature_names=list(x.columns),
                                num_features=10,
                                verbose=False)

    elif explainer_type == 'lime':
        explainer = LimeWrapper(model,
                                mode=mode,
                                lime_type='tabular',
                                multiclass=False,
                                train_data=x.as_matrix(),
                                class_names=['yes', 'no'],
                                feature_names=list(x.columns),
                                num_samples=num_neighbours,
                                num_features=num_features,
                                verbose=False)

    ts = time()

    data = x.sample(num_samples, replace=False, random_state=random_seed).as_matrix()
    lips = explainer.estimate_dataset_lipschitz(dataset=data, continuous=True, eps=1,
                                                maxpoints=None,
                                                optim=lipshitz_optimiser,
                                                bound_type=lipshitz_bound_type,
                                                n_jobs=n_jobs, n_calls=n_calls_optimiser,
                                                verbose=False)

    tdelta = time() - ts
    print(
        "Time elapsed for {} with Gradient Boosted Tree opt {:.2} s".format(explainer_type, tdelta))
    print("Median Lipschitz estimate for {} = {:.3}".format(explainer_type, median(lips)))
    return lips, tdelta
