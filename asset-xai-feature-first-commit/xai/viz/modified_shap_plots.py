import numpy as np
import warnings
from scipy.stats import gaussian_kde


from shap import force_plot, initjs
from shap.plots import shorten_text
import matplotlib.pyplot as pl
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.cm import get_cmap

""" Plots for feature attributions - adapted from SHAP to include custom color maps."""
""" The body of functions written by 2018 Scott Lundberg. Modified under MIT license."""

# To be moved to config_viz
labels = {
    'MAIN_EFFECT': "SHAP main effect value for\n%s",
    'INTERACTION_VALUE': "SHAP interaction value",
    'INTERACTION_EFFECT': "SHAP interaction value for\n%s and %s",
    'VALUE': "Feature attribution value (impact on model output)",
    'GLOBAL_VALUE': "mean(|Feature attribution value|) (average impact on model output magnitude)",
    'VALUE_FOR': "Feature attribution value for\n%s",
    'PLOT_FOR': "Feature attribution plot for %s",
    'FEATURE': "Feature %s",
    'FEATURE_VALUE': "Feature value",
    'FEATURE_VALUE_LOW': "Low",
    'FEATURE_VALUE_HIGH': "High",
    'JOINT_VALUE': "Joint Feature attribution value"
}


def set_colormap(cmap_style=None):
    """
    Allows for alternative ('lime') colormap style to be set
    :param cmap_style: string describing the style. Currently either 'lime' or default
    :return:
        tuple containing colourmap name as a string, 2x matplotlib colourmap objects, and an array of default colours
    """
    if cmap_style == 'lime':
        cmap_str = "PkYg" # PkYg: ["#DD4477", "#66AA00"]
        #  Supported colormaps: https://github.com/interpretable-ml/iml/blob/master/javascript/color-set.js
        cmap = get_cmap('RdYlGn').reversed()
        cmap_solid = get_cmap('RdYlGn').reversed()
    else:
        cmap_str = "RdBu"
        cdict1 = {
            'red': ((0.0, 0.11764705882352941, 0.11764705882352941),
                    (1.0, 0.9607843137254902, 0.9607843137254902)),

            'green': ((0.0, 0.5333333333333333, 0.5333333333333333),
                      (1.0, 0.15294117647058825, 0.15294117647058825)),

            'blue': ((0.0, 0.8980392156862745, 0.8980392156862745),
                     (1.0, 0.3411764705882353, 0.3411764705882353)),

            'alpha': ((0.0, 1, 1),
                      (0.5, 0.3, 0.3),
                      (1.0, 1, 1))
        }
        cmap = LinearSegmentedColormap('RedBlue', cdict1)

        cdict2 = {
            'red': ((0.0, 0.11764705882352941, 0.11764705882352941),
                    (1.0, 0.9607843137254902, 0.9607843137254902)),

            'green': ((0.0, 0.5333333333333333, 0.5333333333333333),
                      (1.0, 0.15294117647058825, 0.15294117647058825)),

            'blue': ((0.0, 0.8980392156862745, 0.8980392156862745),
                     (1.0, 0.3411764705882353, 0.3411764705882353)),

            'alpha': ((0.0, 1, 1),
                      (0.5, 1, 1),
                      (1.0, 1, 1))
        }
        cmap_solid = LinearSegmentedColormap('RedBlue', cdict2)

    # default_colors = ["#1E88E5", "#ff0052", "#13B755", "#7C52FF", "#FFC000", "#00AEEF"]
    blue_rgba = np.array([0.11764705882352941, 0.5333333333333333, 0.8980392156862745, 1.0])
    default_colors = []
    tmp = blue_rgba.copy()
    for i in range(10):
        default_colors.append(tmp.copy())
        if tmp[-1] > 0.1:
            tmp[-1] *= 0.7
    return cmap_str, cmap, cmap_solid, default_colors


def dependence_plot(ind, explanations, x, feature_names=None, display_features=None,
                    interaction_index="auto", cmap_style=None, axis_color="#333333",
                    dot_size=16, alpha=1, title=None, show=True):
    """
    Create a dependence plot for a feature specified in ind, colored by an interaction feature specified by interaction_index.
    Parameters
    ----------
    ind : int
        Index of the feature to plot.
    explanations : numpy.array
        Matrix of SHAP values (# samples x # features)
    x : numpy.array or pandas.DataFrame
        Matrix of feature values (# samples x # features)
    feature_names : list
        Names of the features (length # features)
    display_features : numpy.array or pandas.DataFrame
        Matrix of feature values for visual display (such as strings instead of coded values)
    interaction_index : "auto", None, or int
        The index of the feature used to color the plot.
    cmap_style: string setting the style of the colourmap. Currently either 'lime' or default SHAP value
    """
    cmap_str, cmap, cmap_solid, default_colors = set_colormap(cmap_style=cmap_style)
    # convert from DataFrames if we got any
    if str(type(x)).endswith("'pandas.core.frame.DataFrame'>"):
        if feature_names is None:
            feature_names = x.columns
        x = x.values
    if str(type(display_features)).endswith("'pandas.core.frame.DataFrame'>"):
        if feature_names is None:
            feature_names = display_features.columns
        display_features = display_features.values
    elif display_features is None:
        display_features = x

    if feature_names is None:
        feature_names = [labels['FEATURE'] % str(i) for i in range(explanations.shape[1] - 1)]

    # allow vectors to be passed
    if len(explanations.shape) == 1:
        explanations = np.reshape(explanations, len(explanations), 1)
    if len(x.shape) == 1:
        x = np.reshape(x, len(x), 1)

    def convert_name(ind):
        if type(ind) == str:
            nzinds = np.where(feature_names == ind)[0]
            if len(nzinds) == 0:
                print("Could not find feature named: " + ind)
                return None
            else:
                return nzinds[0]
        else:
            return ind

    ind = convert_name(ind)

    # plotting SHAP interaction values
    if len(explanations.shape) == 3 and len(ind) == 2:
        ind1 = convert_name(ind[0])
        ind2 = convert_name(ind[1])
        if ind1 == ind2:
            proj_shap_values = explanations[:, ind2, :]
        else:
            proj_shap_values = explanations[:, ind2, :] * 2  # off-diag values are split in half

        # TODO: remove recursion; generally the functions should be shorter for more maintainable code
        dependence_plot(
            ind1, proj_shap_values, x, feature_names=feature_names,
            interaction_index=ind2, display_features=display_features, show=False
        )
        if ind1 == ind2:
            pl.ylabel(labels['MAIN_EFFECT'] % feature_names[ind1])
        else:
            pl.ylabel(labels['INTERACTION_EFFECT'] % (feature_names[ind1], feature_names[ind2]))

        if show:
            pl.show()
            pl.close()
        return

    assert explanations.shape[0] == x.shape[
        0], "'shap_values' and 'features' values must have the same number of rows!"
    assert explanations.shape[1] == x.shape[1] + 1, "'shap_values' must have one more column than 'features'!"

    # get both the raw and display feature values
    xv = x[:, ind]
    xd = display_features[:, ind]
    s = explanations[:, ind]
    if type(xd[0]) == str:
        name_map = {}
        for i in range(len(xv)):
            name_map[xd[i]] = xv[i]
        xnames = list(name_map.keys())

    # allow a single feature name to be passed alone
    if type(feature_names) == str:
        feature_names = [feature_names]
    name = feature_names[ind]

    # guess what other feature as the stongest interaction with the plotted feature
    if interaction_index == "auto":
        interaction_index = approx_interactions(ind, explanations, x)[0]
    interaction_index = convert_name(interaction_index)
    categorical_interaction = False

    # get both the raw and display color values
    if interaction_index is not None:
        cv = x[:, interaction_index]
        cd = display_features[:, interaction_index]
        clow = np.nanpercentile(x[:, interaction_index].astype(np.float), 5)
        chigh = np.nanpercentile(x[:, interaction_index].astype(np.float), 95)
        if type(cd[0]) == str:
            cname_map = {}
            for i in range(len(cv)):
                cname_map[cd[i]] = cv[i]
            cnames = list(cname_map.keys())
            categorical_interaction = True
        elif clow % 1 == 0 and chigh % 1 == 0 and len(set(x[:, interaction_index])) < 50:
            categorical_interaction = True

    # discritize colors for categorical features
    color_norm = None
    if categorical_interaction and clow != chigh:
        bounds = np.linspace(clow, chigh, chigh - clow + 2)
        color_norm = BoundaryNorm(bounds, cmap.N)

    # the actual scatter plot, TODO: adapt the dot_size to the number of data points?
    if interaction_index is not None:
        pl.scatter(xv, s, s=dot_size, linewidth=0, c=x[:, interaction_index], cmap=cmap,
                   alpha=alpha, vmin=clow, vmax=chigh, norm=color_norm, rasterized=len(xv) > 500)
    else:
        pl.scatter(xv, s, s=dot_size, linewidth=0, color="#1E88E5",
                   alpha=alpha, rasterized=len(xv) > 500)

    if interaction_index != ind and interaction_index is not None:
        # draw the color bar
        if type(cd[0]) == str:
            tick_positions = [cname_map[n] for n in cnames]
            if len(tick_positions) == 2:
                tick_positions[0] -= 0.25
                tick_positions[1] += 0.25
            cb = pl.colorbar(ticks=tick_positions)
            cb.set_ticklabels(cnames)
        else:
            cb = pl.colorbar()

        cb.set_label(feature_names[interaction_index], size=13)
        cb.ax.tick_params(labelsize=11)
        if categorical_interaction:
            cb.ax.tick_params(length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)
        bbox = cb.ax.get_window_extent().transformed(pl.gcf().dpi_scale_trans.inverted())
        cb.ax.set_aspect((bbox.height - 0.7) * 20)

    # make the plot more readable
    if interaction_index != ind:
        pl.gcf().set_size_inches(7.5, 5)
    else:
        pl.gcf().set_size_inches(6, 5)
    pl.xlabel(name, color=axis_color, fontsize=13)
    pl.ylabel(labels['VALUE_FOR'] % name, color=axis_color, fontsize=13)
    if title is not None:
        pl.title(title, color=axis_color, fontsize=13)
    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('left')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    pl.gca().tick_params(color=axis_color, labelcolor=axis_color, labelsize=11)
    for spine in pl.gca().spines.values():
        spine.set_edgecolor(axis_color)
    if type(xd[0]) == str:
        pl.xticks([name_map[n] for n in xnames], xnames, rotation='vertical', fontsize=11)
    if show:
        pl.show()
        pl.close()


def approx_interactions(index, shap_values, x):
    """ Order other features by how much interaction they seem to have with the feature at the given index.
    This just bins the SHAP values for a feature along that feature's value. For true Shapley interaction
    index values for SHAP see the interaction_contribs option implemented in XGBoost.
    """

    if x.shape[0] > 10000:
        a = np.arange(x.shape[0])
        np.random.shuffle(a)
        inds = a[:10000]
    else:
        inds = np.arange(x.shape[0])

    x = x[inds, index]
    srt = np.argsort(x)
    shap_ref = shap_values[inds, index]
    shap_ref = shap_ref[srt]
    inc = max(min(int(len(x) / 10.0), 50), 1)
    interactions = []
    for i in range(x.shape[1]):
        val_other = x[inds, i][srt].astype(np.float)
        v = 0.0
        if not (i == index or np.sum(np.abs(val_other)) < 1e-8):
            for j in range(0, len(x), inc):
                if np.std(val_other[j:j + inc]) > 0 and np.std(shap_ref[j:j + inc]) > 0:
                    v += abs(np.corrcoef(shap_ref[j:j + inc], val_other[j:j + inc])[0, 1])
        interactions.append(v)

    return np.argsort(-np.abs(interactions))


def summary_plot(explanations, samples_explained=None, cmap_style=None, max_display=None, plot_type="dot",
                 feature_names=None, show=True, sort=True, color_bar=True, auto_size_plot=True):
    """Create a SHAP summary plot, colored by feature values when they are provided.
    Parameters
    ----------
    explanations : numpy.array
        Matrix of explanations values (# samples x # features)
    samples_explained : numpy.array or pandas.DataFrame or list
        Matrix of feature values (# samples x # features) or a feature_names list as shorthand
    cmap_style: string setting the style of the colourmap. Currently either 'lime' or default SHAP value
    feature_names : list
        Names of the features (length # features)
    max_display : int
        How many top features to include in the plot (default is 20, or 7 for interaction plots)
    plot_type : "dot" (default) or "violin"
        What type of summary plot to produce
    """
    color = None
    axis_color = "#333333"
    alpha = 1
    layered_violin_max_num_bins = 20
    class_names = None

    cmap_str, cmap, cmap_solid, default_colors = set_colormap(cmap_style=cmap_style)

    multi_class = False
    if str(type(explanations)).endswith("list'>"):
        multi_class = True
        plot_type = "bar"  # only type supported for now
    else:
        assert len(explanations.shape) != 1, "Summary plots need a matrix of shap_values, not a vector."

    # default color:
    if color is None:
        color = "coolwarm" if plot_type == 'layered_violin' else "#1E88E5"  # "#ff0052"

    # convert from a DataFrame or other types
    if str(type(samples_explained)) == "<class 'pandas.core.frame.DataFrame'>":
        if feature_names is None:
            feature_names = samples_explained.columns
        samples_explained = samples_explained.values
    elif str(type(samples_explained)) == "<class 'list'>":
        if feature_names is None:
            feature_names = samples_explained
        samples_explained = None
    elif (samples_explained is not None) and len(samples_explained.shape) == 1 and feature_names is None:
        feature_names = samples_explained
        samples_explained = None

    num_features = (explanations[0].shape[1] if multi_class else explanations.shape[1]) - 1

    if feature_names is None:
        feature_names = [labels['FEATURE'] % str(i) for i in range(num_features)]

    # plotting SHAP interaction values
    if not multi_class and len(explanations.shape) == 3:
        if max_display is None:
            max_display = 7
        else:
            max_display = min(len(feature_names), max_display)

        sort_inds = np.argsort(-np.abs(explanations[:, :-1, :-1].sum(1)).sum(0))

        # get plotting limits
        delta = 1.0 / (explanations.shape[1] ** 2)
        slow = np.nanpercentile(explanations, delta)
        shigh = np.nanpercentile(explanations, 100 - delta)
        v = max(abs(slow), abs(shigh))
        slow = -v
        shigh = v

        pl.figure(figsize=(1.5 * max_display + 1, 0.8 * max_display + 1))
        pl.subplot(1, max_display, 1)
        proj_shap_values = explanations[:, sort_inds[0], np.hstack((sort_inds, len(sort_inds)))]
        proj_shap_values[:, 1:] *= 2  # because off diag effects are split in half
        summary_plot(
            proj_shap_values, samples_explained[:, sort_inds],
            feature_names=feature_names[sort_inds],
            sort=False, show=False, color_bar=False,
            auto_size_plot=False,
            max_display=max_display
        )
        pl.xlim((slow, shigh))
        pl.xlabel("")
        title_length_limit = 11
        pl.title(shorten_text(feature_names[sort_inds[0]], title_length_limit))
        for i in range(1, min(len(sort_inds), max_display)):
            ind = sort_inds[i]
            pl.subplot(1, max_display, i + 1)
            proj_shap_values = explanations[:, ind, np.hstack((sort_inds, len(sort_inds)))]
            proj_shap_values *= 2
            proj_shap_values[:, i] /= 2  # because only off diag effects are split in half
            summary_plot(
                proj_shap_values, samples_explained[:, sort_inds],
                sort=False,
                feature_names=["" for i in range(samples_explained.shape[1])],
                show=False,
                color_bar=False,
                auto_size_plot=False,
                max_display=max_display
            )
            pl.xlim((slow, shigh))
            pl.xlabel("")
            if i == max_display // 2:
                pl.xlabel(labels['INTERACTION_VALUE'])
            pl.title(shorten_text(feature_names[ind], title_length_limit))
        pl.tight_layout(pad=0, w_pad=0, h_pad=0.0)
        pl.subplots_adjust(hspace=0, wspace=0.1)
        if show:
            pl.show()
            pl.close()

    if max_display is None:
        max_display = 20

    if sort:
        # order features by the sum of their effect magnitudes
        if multi_class:
            feature_order = np.argsort(np.sum(np.mean(np.abs(explanations), axis=0), axis=0)[:-1])
        else:
            feature_order = np.argsort(np.sum(np.abs(explanations), axis=0)[:-1])
        feature_order = feature_order[-min(max_display, len(feature_order)):]
    else:
        feature_order = np.flip(np.arange(min(max_display, num_features)), 0)

    row_height = 0.4
    if auto_size_plot:
        pl.gcf().set_size_inches(8, len(feature_order) * row_height + 1.5)
    pl.axvline(x=0, color="#999999", zorder=-1)

    if plot_type == "dot":
        for pos, i in enumerate(feature_order):
            pl.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
            shaps = explanations[:, i]
            values = None if samples_explained is None else samples_explained[:, i]
            inds = np.arange(len(shaps))
            np.random.shuffle(inds)
            if values is not None:
                values = values[inds]
            shaps = shaps[inds]
            colored_feature = True
            try:
                values = np.array(values, dtype=np.float64)  # make sure this can be numeric
            except:
                colored_feature = False
            N = len(shaps)
            # hspacing = (np.max(shaps) - np.min(shaps)) / 200
            # curr_bin = []
            nbins = 100
            quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
            inds = np.argsort(quant + np.random.randn(N) * 1e-6)
            layer = 0
            last_bin = -1
            ys = np.zeros(N)
            for ind in inds:
                if quant[ind] != last_bin:
                    layer = 0
                ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
                layer += 1
                last_bin = quant[ind]
            ys *= 0.9 * (row_height / np.max(ys + 1))

            if samples_explained is not None and colored_feature:
                # trim the color range, but prevent the color range from collapsing
                vmin = np.nanpercentile(values, 5)
                vmax = np.nanpercentile(values, 95)
                if vmin == vmax:
                    vmin = np.nanpercentile(values, 1)
                    vmax = np.nanpercentile(values, 99)
                    if vmin == vmax:
                        vmin = np.min(values)
                        vmax = np.max(values)

                assert samples_explained.shape[0] == len(shaps), "Feature and SHAP matrices must have the same number of rows!"
                nan_mask = np.isnan(values)
                pl.scatter(shaps[nan_mask], pos + ys[nan_mask], color="#777777", vmin=vmin,
                           vmax=vmax, s=16, alpha=alpha, linewidth=0,
                           zorder=3, rasterized=len(shaps) > 500)
                pl.scatter(shaps[np.invert(nan_mask)], pos + ys[np.invert(nan_mask)],
                           cmap=cmap, vmin=vmin, vmax=vmax, s=16,
                           c=values[np.invert(nan_mask)], alpha=alpha, linewidth=0,
                           zorder=3, rasterized=len(shaps) > 500)
            else:

                pl.scatter(shaps, pos + ys, s=16, alpha=alpha, linewidth=0, zorder=3,
                           color=color if colored_feature else "#777777", rasterized=len(shaps) > 500)

    elif plot_type == "violin":
        for pos, i in enumerate(feature_order):
            pl.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)

        if samples_explained is not None:
            global_low = np.nanpercentile(explanations[:, :len(feature_names)].flatten(), 1)
            global_high = np.nanpercentile(explanations[:, :len(feature_names)].flatten(), 99)
            for pos, i in enumerate(feature_order):
                shaps = explanations[:, i]
                shap_min, shap_max = np.min(shaps), np.max(shaps)
                rng = shap_max - shap_min
                xs = np.linspace(np.min(shaps) - rng * 0.2, np.max(shaps) + rng * 0.2, 100)
                if np.std(shaps) < (global_high - global_low) / 100:
                    ds = gaussian_kde(shaps + np.random.randn(len(shaps)) * (global_high - global_low) / 100)(xs)
                else:
                    ds = gaussian_kde(shaps)(xs)
                ds /= np.max(ds) * 3

                values = samples_explained[:, i]
                window_size = max(10, len(values) // 20)
                smooth_values = np.zeros(len(xs) - 1)
                sort_inds = np.argsort(shaps)
                trailing_pos = 0
                leading_pos = 0
                running_sum = 0
                back_fill = 0
                for j in range(len(xs) - 1):

                    while leading_pos < len(shaps) and xs[j] >= shaps[sort_inds[leading_pos]]:
                        running_sum += values[sort_inds[leading_pos]]
                        leading_pos += 1
                        if leading_pos - trailing_pos > 20:
                            running_sum -= values[sort_inds[trailing_pos]]
                            trailing_pos += 1
                    if leading_pos - trailing_pos > 0:
                        smooth_values[j] = running_sum / (leading_pos - trailing_pos)
                        for k in range(back_fill):
                            smooth_values[j - k - 1] = smooth_values[j]
                    else:
                        back_fill += 1

                vmin = np.nanpercentile(values, 5)
                vmax = np.nanpercentile(values, 95)
                if vmin == vmax:
                    vmin = np.nanpercentile(values, 1)
                    vmax = np.nanpercentile(values, 99)
                    if vmin == vmax:
                        vmin = np.min(values)
                        vmax = np.max(values)
                pl.scatter(shaps, np.ones(explanations.shape[0]) * pos, s=9, cmap=cmap_solid, vmin=vmin, vmax=vmax,
                           c=values, alpha=alpha, linewidth=0, zorder=1)
                # smooth_values -= nxp.nanpercentile(smooth_values, 5)
                # smooth_values /= np.nanpercentile(smooth_values, 95)
                smooth_values -= vmin
                if vmax - vmin > 0:
                    smooth_values /= vmax - vmin
                for i in range(len(xs) - 1):
                    if ds[i] > 0.05 or ds[i + 1] > 0.05:
                        pl.fill_between([xs[i], xs[i + 1]], [pos + ds[i], pos + ds[i + 1]],
                                        [pos - ds[i], pos - ds[i + 1]], color=cmap_solid(smooth_values[i]),
                                        zorder=2)

        else:
            parts = pl.violinplot(explanations[:, feature_order], range(len(feature_order)), points=200, vert=False,
                                  widths=0.7,
                                  showmeans=False, showextrema=False, showmedians=False)

            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_edgecolor('none')
                pc.set_alpha(alpha)

    elif plot_type == "layered_violin":  # courtesy of @kodonnell
        num_x_points = 200
        bins = np.linspace(0, samples_explained.shape[0], layered_violin_max_num_bins + 1).round(0).astype(
            'int')  # the indices of the feature data corresponding to each bin
        shap_min, shap_max = np.min(explanations[:, :-1]), np.max(explanations[:, :-1])
        x_points = np.linspace(shap_min, shap_max, num_x_points)

        # loop through each feature and plot:
        for pos, ind in enumerate(feature_order):
            # decide how to handle: if #unique < layered_violin_max_num_bins then split by unique value, otherwise use bins/percentiles.
            # to keep simpler code, in the case of uniques, we just adjust the bins to align with the unique counts.
            feature = samples_explained[:, ind]
            unique, counts = np.unique(feature, return_counts=True)
            if unique.shape[0] <= layered_violin_max_num_bins:
                order = np.argsort(unique)
                thesebins = np.cumsum(counts[order])
                thesebins = np.insert(thesebins, 0, 0)
            else:
                thesebins = bins
            nbins = thesebins.shape[0] - 1
            # order the feature data so we can apply percentiling
            order = np.argsort(feature)
            # x axis is located at y0 = pos, with pos being there for offset
            y0 = np.ones(num_x_points) * pos
            # calculate kdes:
            ys = np.zeros((nbins, num_x_points))
            for i in range(nbins):
                # get shap values in this bin:
                shaps = explanations[order[thesebins[i]:thesebins[i + 1]], ind]
                # if there's only one element, then we can't
                if shaps.shape[0] == 1:
                    warnings.warn(
                        "not enough data in bin #%d for feature %s, so it'll be ignored. Try increasing the number of records to plot."
                        % (i, feature_names[ind]))
                    # to ignore it, just set it to the previous y-values (so the area between them will be zero). Not ys is already 0, so there's
                    # nothing to do if i == 0
                    if i > 0:
                        ys[i, :] = ys[i - 1, :]
                    continue
                # save kde of them: note that we add a tiny bit of gaussian noise to avoid singular matrix errors
                ys[i, :] = gaussian_kde(shaps + np.random.normal(loc=0, scale=0.001, size=shaps.shape[0]))(x_points)
                # scale it up so that the 'size' of each y represents the size of the bin. For continuous data this will
                # do nothing, but when we've gone with the unqique option, this will matter - e.g. if 99% are male and 1%
                # female, we want the 1% to appear a lot smaller.
                size = thesebins[i + 1] - thesebins[i]
                bin_size_if_even = samples_explained.shape[0] / nbins
                relative_bin_size = size / bin_size_if_even
                ys[i, :] *= relative_bin_size
            # now plot 'em. We don't plot the individual strips, as this can leave whitespace between them.
            # instead, we plot the full kde, then remove outer strip and plot over it, etc., to ensure no
            # whitespace
            ys = np.cumsum(ys, axis=0)
            width = 0.8
            scale = ys.max() * 2 / width  # 2 is here as we plot both sides of x axis
            for i in range(nbins - 1, -1, -1):
                y = ys[i, :] / scale
                c = pl.get_cmap(color)(i / (
                    nbins - 1)) if color in pl.cm.datad else color  # if color is a cmap, use it, otherwise use a color
                pl.fill_between(x_points, pos - y, pos + y, facecolor=c)
        pl.xlim(shap_min, shap_max)

    elif not multi_class and plot_type == "bar":
        feature_inds = feature_order[:max_display]
        y_pos = np.arange(len(feature_inds))
        global_shap_values = np.abs(explanations).mean(0)
        pl.barh(y_pos, global_shap_values[feature_inds], 0.7, align='center', color=color)
        pl.yticks(y_pos, fontsize=13)
        pl.gca().set_yticklabels([feature_names[i] for i in feature_inds])

    elif multi_class and plot_type == "bar":
        if class_names is None:
            class_names = ["Class " + str(i) for i in range(len(explanations))]
        # print("feature_order", feature_order)
        feature_inds = feature_order[:max_display]
        y_pos = np.arange(len(feature_inds))
        left_pos = np.zeros(len(feature_inds))
        # print("feature_inds", feature_inds)

        class_inds = np.argsort([-np.abs(explanations[i]).mean() for i in range(len(explanations))])
        for i, ind in enumerate(class_inds):
            global_shap_values = np.abs(explanations[ind]).mean(0)
            # print("global_shap_values", global_shap_values)

            # print("default_colors", default_colors)
            # print("np.min(i, len(default_colors)-1)", min(i, len(default_colors)-1))
            pl.barh(
                y_pos, global_shap_values[feature_inds], 0.7, left=left_pos, align='center',
                color=default_colors[min(i, len(default_colors) - 1)], label=class_names[ind]
            )
            left_pos += global_shap_values[feature_inds]
        pl.yticks(y_pos, fontsize=13)
        pl.gca().set_yticklabels([feature_names[i] for i in feature_inds])
        pl.legend(frameon=False, fontsize=12)

    # draw the color bar
    if color_bar and samples_explained is not None and plot_type != "bar" and \
            (plot_type != "layered_violin" or color in pl.cm.datad):
        import matplotlib.cm as cm
        m = cm.ScalarMappable(cmap=cmap_solid if plot_type != "layered_violin" else pl.get_cmap(color))
        m.set_array([0, 1])
        cb = pl.colorbar(m, ticks=[0, 1], aspect=1000)
        cb.set_ticklabels([labels['FEATURE_VALUE_LOW'], labels['FEATURE_VALUE_HIGH']])
        cb.set_label(labels['FEATURE_VALUE'], size=12, labelpad=0)
        cb.ax.tick_params(labelsize=11, length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)
        bbox = cb.ax.get_window_extent().transformed(pl.gcf().dpi_scale_trans.inverted())
        cb.ax.set_aspect((bbox.height - 0.9) * 20)
        # cb.draw_all()

    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('none')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    pl.gca().spines['left'].set_visible(False)
    pl.gca().tick_params(color=axis_color, labelcolor=axis_color)
    pl.yticks(range(len(feature_order)), [feature_names[i] for i in feature_order], fontsize=13)
    if plot_type != "bar":
        pl.gca().tick_params('y', length=20, width=0.5, which='major')
    pl.gca().tick_params('x', labelsize=11)
    pl.ylim(-1, len(feature_order))
    if plot_type == "bar":
        pl.xlabel(labels['GLOBAL_VALUE'], fontsize=13)
    else:
        pl.xlabel(labels['VALUE'], fontsize=13)
    if show:
        pl.show()
        pl.close()


def js_force_plot(explanations, x, cmap_style=None):
    """
    :param explanations: Matrix of additive attribution values (# samples x # features)
    :param x: numpy.array or pandas.DataFrame Matrix of feature values (# samples x # features)
    :param cmap_style: string setting the style of the colourmap. Currently either 'lime' or default SHAP value
    :return:
    """
    cmap_str, cmap, cmap_solid, default_colors = set_colormap(cmap_style=cmap_style)
    initjs()
    return force_plot(explanations, x, plot_cmap=cmap_str)
