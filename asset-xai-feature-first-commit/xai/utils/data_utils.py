from pandas import DataFrame
from statsmodels.api import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor


def find_categorical_idx(x, max_levels=100,
                         categorical_dtypes=['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16',
                                             'uint32', 'uint64', 'bool', 'str']):
    """
    Function to determine which columns in x are categorical (nominal or ordinal) based on specific
    data types and the number of unique values the x has
    :param x: pandas dataframe with features in columns
    :param max_levels: integer specifying the maximum number of unique values in the column for it
    to be categorical
    :param categorical_dtypes: list of strings that describe datatypes to be classed as categorical
    :return: list of integer indices in x that constitute to categorical features
    """
    if categorical_dtypes is not None:
        categorical_idx = [x.columns.get_loc(col) for col in x.columns if
                           (x[col].nunique() <= max_levels and x[col].dtypes in categorical_dtypes)]
    else:
        categorical_idx = [x.columns.get_loc(col) for col in x.columns if
                           x[col].nunique() <= max_levels]
    print("Identified {} out of {} features as categorical.".format(len(categorical_idx),
                                                                    len(x.columns)))
    # TODO: user message that XYZ features will be treated as categorical
    return categorical_idx


def compute_vifs(x):
    """
    Computes Variance Inflation Factors for the model variables specified in x
    Args:
        x: dataframe containing independent variables

    Returns:
        vif: array containing VIF values for every variable in x, plus intercept
    """
    assert isinstance(x, DataFrame), "X must be a pandas DataFrame object"
    temp = add_constant(x, has_constant='add')
    vif = [variance_inflation_factor(temp.values, temp.columns.get_loc(i)) for i in temp.columns]
    vif_dataframe = DataFrame(vif, index=['const'] + list(x.columns), columns=['vif'])
    return vif_dataframe
