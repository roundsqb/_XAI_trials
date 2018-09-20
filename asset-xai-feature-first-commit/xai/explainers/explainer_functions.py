from shap import TreeExplainer
from lime.lime_tabular import LimeTabularExplainer
from time import time
from math import sqrt
from numpy import array, asarray, zeros, random


# Functions for post-hoc explainability

def instantiate_lime_explainer(x, categorical_idx, is_classification, random_state=42):
    """
    Instantiates LimeTabularExplainer with the data in x
    :param x: input dataset
    :param categorical_idx: array of integers corresponding to the indices of the categorical
    columns in x
    :param is_classification: binary flag indicating the LimeTabularExplainer mode.
    Set True for 'classification', False for 'regression'
    :param random_state: integer for the random seed. Default: 42
    :return: LimeTabularExplainer object
    """
    print("\tInstantiating LIME explainer ...")
    ts = time()
    mode = ['regression', 'classification'][1 * is_classification]
    explainer = LimeTabularExplainer(
        training_data=array(x),
        feature_names=list(x.columns),
        mode=mode,
        categorical_features=categorical_idx,
        kernel_width=sqrt(len(x.columns)) * 0.75,
        verbose=False,
        class_names=None,
        feature_selection='auto',
        discretize_continuous=True,
        discretizer='decile',
        sample_around_instance=True,
        random_state=random_state)
    te = time()
    print("\tProcess took {:.2f} s.".format((te - ts)))
    return explainer


def lime_sample_exp_to_array(sample_exp):
    """
    Function to convert native LIME output object (sample_exp) into an array of feature attributions
     and an intercept
    :param sample_exp: native LIME explanation object
    :return: 1 x (p+1) array containing p feature attributions and 1 local intercept
    """
    return asarray([x[1] for x in sorted(sample_exp.local_exp[1], key=lambda x: x[0])]
                   + [sample_exp.intercept[1]])


def lime_get_ft_idx(sample_exp):
    """
    Function determines indices of features which feature in the LIME's explanation object
    :param sample_exp: native LIME explanation object
    :return: array of indices for the features that have been explained
    """
    return sorted([x[0] for x in sample_exp.local_exp[1]])


def lime_explain_instance(model, explainer, sample, num_features=None, num_neighbours=None,
                          print_flag=True):
    """
    This auxiliary function is a wrapper that calls on the native LimeTabularExplainer
    explain_instance() method
    :param num_features: int specifying maximum number of features present in explanation
    :param num_neighbours: in specifying the size of the neighborhood to learn the linear mode
    :param print_flag: boolean, set True to print global and local predictions
    :param model: pre-trained sklearn model object (must have .predict_proba or predict attributes)
    :param explainer: LimeTabularExplainer object instantiated for the data in x
    :param sample: an single instance in x
    :return:
            sample_exp: lime Explanation object with built-in visualisation methods

    """
    # Ensure num_features does not exceed available number of samples
    if num_features is None:
        num_features = sample.shape[0]
    elif num_features > sample.shape[0]:
        num_features = sample.shape[0]

    # Default is 500
    if num_neighbours is None:
        num_neighbours = 500

    # Setting random seed for reproducibility
    random.seed(42)
    if (explainer.mode == 'classification') & hasattr(model, 'predict_proba'):
        sample_exp = explainer.explain_instance(data_row=sample, num_features=num_features,
                                                num_samples=num_neighbours,
                                                distance_metric='euclidean',
                                                model_regressor=None,
                                                # Defaults to sklearn Ridge regression in LimeBase
                                                top_labels=None, labels=(1,),
                                                predict_fn=model.predict_proba)
        if print_flag:
            print('Global prediction = %.2f' % sample_exp.predict_proba[1])
            print('Local prediction = %.2f' % sample_exp.local_pred)
    elif explainer.mode == 'classification':
        raise AttributeError(
            "LIME does not currently support classifier models without probability scores. "
            "If this conflicts with your use case, refer to this issue: "
            "https://github.com/datascienceinc/lime/issues/16")
    elif (explainer.mode == 'regression') & hasattr(model, 'predict'):
        sample_exp = explainer.explain_instance(data_row=sample, num_features=num_features,
                                                num_samples=num_neighbours,
                                                distance_metric='euclidean',
                                                model_regressor=None,
                                                predict_fn=model.predict)
        if print_flag:
            print('Global prediction = %.2f' % sample_exp.predicted_value)
            print('Local prediction = %.2f' % sample_exp.local_pred[0])
    else:
        raise AttributeError("Model must have predict() or predict_proba() attributes")
    return sample_exp


def shap_explain_instance(model, sample, is_classification=True, print_flag=True):
    """
    This auxiliary function is a wrapper that calls on the native TreeExplainer shap_values() method
    :param model: pre-trained sklearn model object (must have .predict_proba or .predict attributes)
    :param sample: an single instance in x
    :param is_classification: binary flag, set True for 'classification', False for 'regression'
    :param print_flag: boolean, set True to print global and local predictions
    :return:
            sample_exp: array of arrays of shapley values


    """
    # TODO: add subsetting on top num_features
    if is_classification:
        class_of_interest = 1
        sample_exp_array = TreeExplainer(model).shap_values(sample)[class_of_interest]
    else:
        sample_exp_array = TreeExplainer(model).shap_values(sample)
    if print_flag:
        print('Local prediction = Global prediction = %.2f' % sample_exp_array.sum())
    return sample_exp_array


def lime_explain_multiple_instances(model, explainer, x, num_samples=100, num_features=None,
                                    num_neighbours=None):
    """
    The function combines the individual explanations for num_samples instances into a numpy array
    :param model: pre-trained sklearn model object (must have .predict_proba or .predict attributes)
    :param explainer: LimeTabularExplainer object instantiated for the data in x
    :param x: input dataset x for which explanations to be generated
    :param num_samples: maximum number of instances to be sampled from x.
    :param num_features: int specifying maximum number of features present in explanation.
    Default: len(x.columns)
    :param num_neighbours: in specifying the size of the neighborhood to learn the linear model.
    Default 500
    :return: tuple comprising:
        explanations: numpy array of arrays of float LIME feature attributions and local intercepts
        for each sample
        samples_explained: array containing the slice of x addressed by the explanations

    """
    assert hasattr(model, 'predict_proba') | hasattr(model, 'predict'), \
        'To be supported by XAI, the model must have  predict_proba() or predict() attributes.'

    # Default to all features if num_features not specified
    # num_features = num_features if num_features else x.shape[1]
    if num_features is None:
        num_features = x.shape[1]
    elif num_features > x.shape[1]:
        num_features = x.shape[1]

    if num_samples >= x.shape[0]:
        num_samples = x.shape[0]
        samples_explained = x
    else:
        samples_explained = x.sample(n=num_samples, replace=False, random_state=42, axis=None)

    ts = time()

    explanations = zeros(shape=(num_samples, x.shape[1] + 1), dtype=float)
    counter = 0
    # TODO: parralelise the for loop
    for sample_n, sample in samples_explained.iterrows():
        sample_exp = lime_explain_instance(model, explainer, sample, num_features=num_features,
                                           num_neighbours=num_neighbours, print_flag=False)
        fts = [x[0] for x in sample_exp.local_exp[1]] + [x.shape[1]]
        imp = [x[1] for x in sample_exp.local_exp[1]] + [sample_exp.intercept[1]]
        explanations[counter, fts] = imp
        counter += 1

    print('LIME values identified for {} samples.'.format(explanations.shape[0]))
    te = time()
    print("Process took {:.2f} s.".format((te - ts)))
    return explanations, samples_explained


def explain_model(model, x, categorical_idx, explainer_type, is_classification,
                  num_samples=None, num_features=None, num_neighbours=None):
    """
    The function takes in the sklearn pre-trained model object and input data x and runs SHAP's
    TreeExplainer or LIME's LimeTabularExplainer to output the array of arrays, where each float is
    the explanation attributed to a particular feature of a particular sample. The output is native
    to SHAP; for LIME, the function calls lime_explain_instance() which is currently slow.
    :param model: pre-trained sklearn model object (must have .predict_proba or .predict)
    :param is_classification: binary flag, set True for 'classification', False for 'regression'
    :param x: input dataset
    :param categorical_idx: array of integers specifying the indices of the categical columns in x
    :param num_samples: optional integer specifying the number of samples to be explained (randomly
    sub-sampled from x)
    :param explainer_type: string, currently supported values: 'shap' or 'lime'
    :param num_features: (not required for shap) int specifying maximum number of features present
    in LIME explanation.
    Default: len(x.columns)
    :param num_neighbours: (not required for shap) int specifying the neighborhood size in LIME's
    linear model.
    Default 500
    :return: tuple comprising
        explanations: numpy array of arrays of float feature attributions for each sample
        samples_explained: pandas dataframe input x (or subset of x if num_samples were specified)
    """
    assert hasattr(model, 'predict_proba') | hasattr(model, 'predict'), \
        'To be supported by XAI, the model must have  predict_proba() or predict() attributes.'

    supported_explainer_types = ['shap', 'lime']
    assert explainer_type in supported_explainer_types, \
        'currently supported explanation models are {}'.format(supported_explainer_types)

    if num_samples is None:
        num_samples = x.shape[0]

    print("Generating explanations ...")
    if explainer_type == 'shap':
        ts = time()
        if num_samples >= x.shape[0]:
            samples_explained = x
        else:
            samples_explained = x.sample(n=num_samples, replace=False, random_state=42, axis=None)

        explanations = TreeExplainer(model).shap_values(samples_explained)
        if is_classification & (str(type(explanations)) == "<class 'list'>"):
            class_of_interest = 1
            explanations = explanations[class_of_interest]
        print('SHAP values identified for {} samples.'.format(explanations.shape[0]))
        te = time()
        print("Process took {:.2f} s.".format((te - ts)))
    if explainer_type == 'lime':
        explainer = instantiate_lime_explainer(x, categorical_idx,
                                               is_classification=is_classification,
                                               random_state=42)
        explanations, samples_explained = lime_explain_multiple_instances(
            model, explainer, x, num_samples=num_samples, num_features=num_features,
            num_neighbours=num_neighbours)
    print("Output explanations shape: {}".format(explanations.shape))
    return explanations, samples_explained
