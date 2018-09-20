from numpy import zeros_like, triu_indices_from, arange, median
from matplotlib import pyplot as plt
from seaborn import heatmap, diverging_palette
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from pandas import DataFrame
from typing import List

from .modified_shap_plots import set_colormap
from ..utils.model_utils import rf_feature_importances


def bar_plot(explanation, sample, cmap_style=None):
    """
    Bar plot of individualised explanations
    :param explanation: numpy array of explanation values where the last value is the intercept/base value
    :param sample: slice of pandas dataframe with the feature names and values for a single sample
    :param cmap_style: string setting the style of the colourmap. Currently either 'lime' or default SHAP value
    :return:
    """
    cmap_str, cmap, cmap_solid, default_colors = set_colormap(cmap_style=cmap_style)
    df = DataFrame(explanation[:-1], index=sample.index, columns=['ft_attr'])
    df['abs'] = abs(df['ft_attr'])
    df = df.sort_values(by='abs', ascending=True)
    position = arange(len(df['ft_attr'])) + .5
    colors = [cmap(50) if x < 0 else cmap(cmap.N) for x in df['ft_attr']]
    fig = plt.figure()
    plt.barh(position, df['ft_attr'], align='center', color=colors)
    plt.yticks(position, df.index)
    plt.xlabel("Feature attribution value")
    plt.close()
    return fig


def plot_corr_heatmap(df):
    """
    Plots a heatmap of correlations with the pairwise correlation values
    Args:
        df: pandas dataframe

    Returns: matplotlib figure

    """
    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = zeros_like(corr, dtype=bool)
    mask[triu_indices_from(mask)] = True

    fig, ax = plt.subplots(figsize=(20, 20))
    cmap = diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1,
                square=True, linecolor="lightgray", linewidths=.5, cbar_kws={"shrink": .5},
                ax=ax)
    # Add labels
    for i in range(len(corr)):
        ax.text(-0.5, len(corr) - (i + 0.5), corr.columns[i],
                ha="right", va="center", rotation=0, fontsize=8)
        ax.text(i + 0.5, -0.5, corr.columns[i],
                ha="center", va="top", rotation=90, fontsize=8)
        for j in range(i + 1, len(corr)):
            s = "{:.2f}".format(corr.values[i, j])
            ax.text(j + 0.5, len(corr) - (i + 0.5), s,
                    ha="center", va="center", fontsize=4)
    plt.axis('off')
    plt.close()
    return fig


def plot_calibration_curve(x, y, models, model_names=None):
    """
    Produces matplotlib plot if the calibration curves for the models specified in models
    :param x: independent variables in the test data to verify calibration
    :param y: dependent variable in the test data to verify calibration
    :param models: a collection of pre-trained sklearn model objects (each object must have predict_proba attribute)
    :param model_names: an optional list of strings to be used for labelling the models
    :return:
        matplotlib figure handle
    """
    fig, ax = plt.subplots()

    for i, model in enumerate(models):
        if model is not None:
            assert hasattr(model, 'predict_proba') | hasattr(model, 'predict'), \
                'Currently  XAI only supports models with either predict_proba or predict attribute'
            try:
                model_name = model_names[i]
            except (IndexError, TypeError):
                model_name = 'model' + str(i)

            if hasattr(model, 'predict_proba'):
                prob_pos = model.predict_proba(x)[:, 1]
            elif hasattr(model, 'predict'):
                prob_pos = model.predict(x)

            fraction_of_positives, mean_predicted_value = calibration_curve(y, prob_pos, n_bins=10)
            if hasattr(y, 'max'):
                clf_score = brier_score_loss(y, prob_pos, pos_label=y.max())
            else:
                clf_score = brier_score_loss(y, prob_pos, pos_label=max(y))
            ax.plot(mean_predicted_value, fraction_of_positives, "s-",
                    label="%s (%1.3f)" % (model_name, clf_score))
        ax.set_ylabel("Fraction of positives")
        ax.set_xlabel("Predicted probability")
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([-0.05, 1.05])
        ax.legend(loc="upper left")
        ax.set_title('Calibration plots  (reliability curve)')
    plt.close()
    return fig


def plot_rf_feature_importances(model, data):
    """
    Horizontal bar plot that depicts the native feature importance values from the RF and XGBoost models.
    :param model: model object that has feature_importance_ attribute
    :param data: pandas dataframe on which the feature importances should be outputed
    :return:
        matplotlib figure handle
    """
    rfs = rf_feature_importances(model, data)
    fig = plt.figure()
    position = arange(len(rfs['ft_imp'])) + .5
    fig = plt.figure()
    plt.barh(position, rfs['ft_imp'], align='center')
    plt.yticks(position, rfs.index)
    plt.xlabel("Native feature importance values")
    plt.close()
    return fig


def plot_robustness_comparisons(values_list, model_names=None, dataset_name=''):
    fig, ax = plt.subplots()

    if (model_names is None) or (len(model_names) < len(values_list)):
        model_names = ['model' + str(i) for i in range(len(values_list))]

    ax.boxplot(values_list, labels=model_names)
    # ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['A', 'B'], loc='upper right')
    # "%s (%0.3f)" % (model_name, median(values))
    ax.set_ylabel("Lipshitz estimate")
    ax.set_xlabel("Method")
    # ax.set_ylim([-0.05, 1.05])
    # ax.set_xlim([-0.05, 1.05])
    ax.legend(loc="upper left")
    ax.set_title('Robustness comparisons for {} dataset with {} samples'.format(dataset_name, len(values_list[0])))
    plt.grid()
    plt.close()
    return fig

