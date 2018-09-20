from pandas import DataFrame


def write_description(sample_exp_array, sample, lookup_table, pos_class_threshold=0.1, top_n=5,
                      model_name='', target_name=''):
    """
    Generates free text description of the sample's LIME or SHAP explanations array. Works best with
     continuous, ordinal categorical, and dummy indicator variables.
    :param sample_exp_array: numpy array describing the feature attributions from the XAI module
    :param sample: pandas Series describing the feature names and values present in a single sample
    :param lookup_table: pandas dataframe with the table containing the 'alias', 'description',
    'source name',
    'unit', 'tolerance' for every feature in the sample.
    :param pos_class_threshold: float specifying what probability should be classed as "high".
    Particularly relevant for regression models, and classification models with with high class
    imbalance/low calibration.
    :param top_n: integer specifying how many top n features contributing towards the predicted
    score to display
    :param model_name: string describing model name, for example "XYZ propensity score model"
    :param target_name: string describing the classification target name, for example "likelihood of
     recovery"
    :return: string containing free-text
    """
    assert {'alias', 'description', 'source name', 'unit', 'tolerance'}.issubset(
        lookup_table.columns), "Header of the lookup_table file must contain columns " \
                               "'alias', 'description', 'source name', 'unit', and 'tolerance'."
    pred_prob = sample_exp_array.sum()

    df = DataFrame(sample_exp_array[:-1], index=list(sample.index), columns=['ft_attr'])
    df['value'] = sample.values
    df = df.join(lookup_table.set_index('alias'), how='left')

    if pred_prob >= pos_class_threshold:
        prob_qualifier = 'high'
        ascending = False
    else:
        prob_qualifier = 'low'
        ascending = True

    # Generating text for the main reason
    top_reason = df.sort_values('ft_attr', ascending=ascending).iloc[0]
    text1 = (("The dominant reason why Client #{0} was predicted by the " + model_name +
              " model as having a " + prob_qualifier + ' ' + target_name +
              ", was because their {1} was {2:.0f} {3}.")
             .format(sample.name, top_reason.description, top_reason.value, top_reason.unit))

    # Genereating text for the next top n reasons that will be grouped by source name
    top_n_reasons = (df
                     .sort_values('ft_attr', ascending=ascending)
                     .head(top_n)
                     .groupby('source name')
                     .agg(lambda x: list(x)))

    conjunctions = {0: 'Additionally', 1: 'Moreover', 2: 'Furthermore', 3: 'Further', 4: 'Also'}
    text2 = '\n'
    for j, reason_group in enumerate(top_n_reasons.index):
        text2 = text2 + "\n" + conjunctions[j] + " the client's {} indicate " \
            .format(top_n_reasons.iloc[j].name)
        for i, reason in enumerate(top_n_reasons.iloc[j].description):
            if i < len(top_n_reasons.iloc[j].description) - 1:
                punctuation = ' and '
            else:
                punctuation = '.'
            text2 = text2 + (
                value_qualifier(top_n_reasons.iloc[j].value[i], top_n_reasons.iloc[j].unit[i],
                                top_n_reasons.iloc[j].tolerance[i])
                + ' ' + reason + punctuation)

    # Generating text for one reason against the prediction
    top_against = df.sort_values('ft_attr', ascending=not ascending).iloc[0]
    text3 = (("\n\nThe reason why Client #{0} was not assigned even " +
              prob_qualifier + "er " + target_name + ", was because their {1} was {2:.0f} {3}.")
             .format(sample.name, top_against.description, top_against.value, top_against.unit))

    return text1 + text2 + text3


def value_qualifier(value, unit, tolerance):
    """
    Auxiliary function for write_description() that describes the numeric value of a categorical or
    ordinal feature using a string qualifier.
    Args:
        value: numeral specifying the value of a feature
        unit: string specifying the unit; value will be treated as unitless if unit == '' or ' '.
        tolerance: tolerance: +/- value that defines what should be classed as 'almost no'

    Returns: string qualifier
    """
    if tolerance is None:
        tolerance = 0.1

    tolerance_multiplier = 5

    if (unit == '') | (unit == ' '):
        # Return the value for unitless features
        return str(value)
    else:
        if value == 0:
            return 'no'
        elif -tolerance < value < tolerance:
            return 'almost no'
        elif value > 0:
            if value < tolerance * tolerance_multiplier:
                return 'small positive'
            else:
                return 'large positive'
        elif value < 0:
            if value < tolerance * tolerance_multiplier:
                return 'small negative'
            else:
                return 'large negative'
