# Copyright 2022 Softpoint Consultores SL. All Rights Reserved.
#
# Licensed under MIT License (the "License");
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

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
