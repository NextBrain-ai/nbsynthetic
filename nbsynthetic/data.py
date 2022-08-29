"""
Created on 01/09/2022
@author: Javier Marin 
Load Data.
"""

import pandas as pd


def input_data(filename, decimal='.'):
    """
    Args:
        file name(str)
            name of the csv file with data

    Returns:
        pandas dataframe(pd.DataFrame) 
    """
    df = pd.read_csv(
        filename + str('.csv'),
        sep=',',
        decimal=decimal
    )
    if isinstance(df, pd.DataFrame) is False:
        raise ValueError(
            "Input data is a {}, not a pandas dataframe".format(type(df))
        )

    return df


def manage_datetime_columns(df, column_name) -> pd.DataFrame:
    """Convert datetime columns into its isocalendar
    values creating three new columns.
    Args:
        df (pd.DataFrame):
            dataframe with a datetime column
        columns_name(str):
            name of the datetime column

    Returns:
        pandas dataframe(pd.DataFrame)
        dataframe with three new categorical columns
        (year/week/day) instead of date time column.
        """

    for col in df.columns:
        if df[col].dtype == '<M8[ns]':
            df['year'] = df[column_name].dt.isocalendar().iloc[:, 0]
            df['week'] = df[column_name].dt.isocalendar().iloc[:, 1]
            df['day'] = df[column_name].dt.isocalendar().iloc[:, 2]
            df = df.drop(columns=[col])

    return df.astype(float)
