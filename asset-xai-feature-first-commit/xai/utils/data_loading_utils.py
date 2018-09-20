from csv import reader
from pandas import read_csv
import os
from sklearn.model_selection import train_test_split
from .data_utils import find_categorical_idx


def load_data_boston():
    """
    # CRIM - per capita crime rate by town
    # ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
    # INDUS - proportion of non-retail business acres per town.
    # CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
    # NOX - nitric oxides concentration (parts per 10 million)
    # RM - average number of rooms per dwelling
    # AGE - proportion of owner-occupied units built prior to 1940
    # DIS - weighted distances to five Boston employment centres
    # RAD - index of accessibility to radial highways
    # TAX - full-value property-tax rate per $10,000
    # PTRATIO - pupil-teacher ratio by town
    # B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    # LSTAT - % lower status of the population
    # MEDV - Median value of owner-occupied homes in $1000's

    :return
    x: dataframe with 14 attributes across 80% of all samples
    x_test: dataframe with 14 attributes corresponding to 20% of samples
    y: array with the house price corresponding to the samples in x
    y_test: array with the house price corresponding to the samples in x_test
    categorical_idx: array of integers containing indices of the categorical columns in x

    """
    boston = read_csv(os.path.join('datasets', 'boston_dataset.csv'), dtype='float64')
    y = boston.pop('y')
    x = boston
    x, x_test, y, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    categorical_idx = find_categorical_idx(x, max_levels=10)
    return x, x_test, y, y_test, categorical_idx


def load_data_census():
    """
    age: continuous.
    workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov,
            Without-pay, Never-worked.
    fnlwgt: continuous.
    education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th,
            7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    education-num: continuous.
    marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed,
            Married-spouse-absent, Married-AF-spouse.
    occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty,
            Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving,
            Priv-house-serv, Protective-serv, Armed-Forces.
    relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    sex: Female, Male.
    capital-gain: continuous.
    capital-loss: continuous.
    hours-per-week: continuous.
    native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany,
                    Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran,
                    Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal,
                    Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia,
                    Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador,
                    Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

    :return
    x: dataframe with 14 attributes across 80% of all samples
    x_test: dataframe with 14 attributes corresponding to 20% of samples
    y: binary array for whether or not the income is >50K corresponding to the samples in x
    y_test: binary array corresponding to the samples in x_test. Values are true if income is >50K
    categorical_idx: array of integers containing indices of the categorical columns in x

    """
    census = read_csv(os.path.join('datasets', 'census_dataset.csv'), dtype='float64')
    y = census.pop('y')
    x = census.drop("Relationship", axis=1)
    x, x_test, y, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    categorical_idx = [1, 3, 4, 5, 6, 7, 11]
    return x, x_test, y, y_test, categorical_idx


def load_data_from_csv(file_name, target_name, max_levels=100, test_size=0.2,
                       skiprows=False, multiples_of_rows_to_skip=100):
    """
    Loads a dataframe from a local file specified and outputs it in a form of x, x_test, y, y_test,
    categorical_idx. It identifies the dependent variable using the target name specified.
    The categorical columns in the input are determined by having less than  max_levels.
    The test subset is generated by random sampling without replacing and is of test_size specified.
    :param file_name: string specifying the full path, file name and extension
    (must be .csv with a header)
    :param target_name: string specifying the name of the dependent variable
    (must be one of the columns in the header)
    :param test_size: float specifying the proportion of samples to be set aside for test
    :param max_levels: integer specifying  the maximum number of levels a variable can have to be
    considered categorical
    :param skiprows: boolean, True if rows should be skipped when reading the file
    :param multiples_of_rows_to_skip: integer, if skiprows == True, then the reader will read every
    Nth row
    :return: tuple
        x: dataframe containing 80% of all samples
        x_test: dataframe corresponding to the remaining 20% of samples
        y: array of dependent variable corresponding to x
        y_test: array of dependent variable corresponding to x_test
        categorical_idx: array of integers containing indices of the categorical columns in x
    """
    try:
        with open(file_name, newline='') as f:
            header = next(reader(f))
    except FileNotFoundError:
        raise FileNotFoundError("File {} not found. Check the directory specified "
                                "and the file_name format (must be .csv)".format(file_name))
    else:
        assert target_name in header, "Column name specified in target_name does not exists."
        if multiples_of_rows_to_skip is None:
            multiples_of_rows_to_skip = 100
        if skiprows:
            num_lines = sum(1 for _ in open(file_name, newline=''))
            skip_idx = [x for x in range(1, num_lines) if x % multiples_of_rows_to_skip != 0]
            df = read_csv(file_name, sep=",", low_memory=False, na_values='?', skiprows=skip_idx,
                          dtype='float64')
        else:
            df = read_csv(file_name, sep=",", low_memory=False, na_values='?', dtype='float64')
        y = df.pop(target_name)
        x = df
        categorical_idx = find_categorical_idx(x, max_levels=max_levels)
        x, x_test, y, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
        return x, x_test, y, y_test, categorical_idx