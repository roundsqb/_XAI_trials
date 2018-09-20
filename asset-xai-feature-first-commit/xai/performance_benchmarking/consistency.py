from scipy.stats import kendalltau
from numpy import round


def compute_kendaltau(rank1, rank2):
    """
    Calculates Kendall’s Tau - a measure of correspondence between two rankings - using
    scipy.stats implementation. Values close to 1 indicate strong agreement, values close to -1
    indicate strong disagreement. This is the 1945 “tau-b” version of Kendall’s tau, which can
    account for ties. :param rank1: list of integers, corresponding to the indices of the
    features in the sample explained, where the order corresponds to the magnitude of the feature
    attribution in explanation 1 :param rank2: list of integers, ... [as above]... feature
    attrubution in explanation 2 :return: tuple of floats contraining the correlation and p-value
    """
    assert len(rank1) == len(rank2), "All inputs to `kendalltau` must be of the same size, " \
                                     "found rank1 %s and rank2 %s" % (len(rank1), len(rank2))
    tau, p_value = kendalltau(rank1, rank2, nan_policy='propagate')
    return tau, p_value


def lime_generate_feature_importance_rank(lime_sample_exp_obj, num_top_features=100):
    """
    Converts LIME explanation object into a list of features' indices ordered by the magnitude of
    the feature's coefficient in the lime explanation :param lime_sample_exp_obj: LIME
    explanation object :param num_top_features: integer specifying the number of top features to
    be included in the ranking :return: a list of integer indices
    """
    num_top_features = min(num_top_features, len(lime_sample_exp_obj.local_exp[1]))
    return [x[0] for x in lime_sample_exp_obj.local_exp[1][:num_top_features]]


def shap_generate_feature_importance_rank(shap_values_array, num_top_features=100):
    """
    Converts feature attributions from the array outputted by SHAP.shap_values() into a list of
    features' indices ordered by the magnitude of the feature's attribution :param
    shap_values_array: array of floats outputted by SHAP.shap_values() :param num_top_features:
    num_top_features: integer number of top features to be included in the ranking :return: a
    list of integer indices
    """
    if num_top_features < len(shap_values_array)-1:
        feature_values = shap_values_array[0:num_top_features]
    else:
        # dropping the last element of the shap array, which is the base value
        feature_values = shap_values_array[0:-1]
    feature_indices = range(len(feature_values))
    return sorted(feature_indices, key=lambda v: abs(feature_values[v]), reverse=True)


def evaluate_consistency_between_shap_and_lime(shap_values_array, lime_sample_exp_obj,
                                               num_top_features=100):
    """
    Computes consistency between the SHAP and LIME feature ranks using Kendall's Tau-B. :param
    shap_values_array: array of floats outputted by SHAP.shap_values() :param
    lime_sample_exp_obj: LIME explanation object :param num_top_features: integer number of top
    features to be compared :return: tuple of floats contraining the correlation and p-value:
    tau: float between -1 and 1, where 1 indicates strong agreement, and -1 - strong disagreement
    in the ranks. p_value: float between 0 and 1
    """
    # TODO: this could be extended so that features with negligible (< epsilon) differences in
    # attributions are tied
    # TODO: another extension: to separate +ve and negative drivers
    shap_ranks = shap_generate_feature_importance_rank(shap_values_array, num_top_features)
    lime_ranks = lime_generate_feature_importance_rank(lime_sample_exp_obj, num_top_features)
    tau, _ = compute_kendaltau(shap_ranks, lime_ranks)
    if tau > 0.1:
        verdict = "The explanations agree with Kendall's Tau of {}".format(round(tau, 2))
    elif tau < -0.1:
        verdict = "The explanations disagree (Kendall's Tau = {})".format(round(tau, 2))
    else:
        verdict = "The agreement of explanations is inconclusive (Kendall's Tau = {})"\
            .format(round(tau, 2))
    print(verdict)
    return round(tau, 3)
