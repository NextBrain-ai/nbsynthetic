# -*- coding: utf-8 -*-

"""
Created on 25/03/2017
@author: Javier Marin 
Dataset preparation following some heuristis.
"""

import pandas as pd
import numpy as np
import scipy
import sklearn
import sklearn.model_selection

kl_div = scipy.special.kl_div
stats = scipy.stats
train_test_split = sklearn.model_selection.train_test_split
OrdinalEncoder = sklearn.preprocessing.OrdinalEncoder


class SmartBrain(object):
    """
      Functions to prepare datasets focus in three areas:
        1- Handling missing data.
        2- Augment dataset if is necessary
        3- Set the correct dtypes

      Parameters:
      -----------
      df: data

      Returns:
      --------
      Utils for smart management of NaN values, dtypes and 
      data augmentation

      Call:
      -----
      s = Smart(df)
      df = s.smart_preparation(df) """

    def __init__(self):

        self

    def nbTypes(self, df):

        objects, datetime = [], []
        [objects.append(c) for c in df.columns if df[c].dtype == object]
        for o in objects:
            if (df[o].dropna().reset_index(drop=True).loc[1].find('-') >= 0 and
                (df[o].dropna().reset_index(drop=True).loc[1].find('20') >= 0 or
                 df[o].dropna().reset_index(drop=True).loc[1].find('19') >= 0)):
                datetime.append(o)
        for d in datetime:
            df[d] = pd.to_datetime(df[d])

        return df

    def nbFillNan(self, df):
        """
          Returns a dataframe with nan values replaced by columns mean or median
          values following a minimum lost of information criteria.

          Parameters:
          -----------
          df: dataframe with continuous variables with nan values

          Returns:
          --------
          Dataframe """

        df_original = df.copy()

        objects = [c for c in df.columns if df[c].dtype == object]
        datetime = [o for o in objects if (
            df[o].dropna().reset_index(drop=True).loc[1].find('-') >= 0 and
            (df[o].dropna().reset_index(drop=True).loc[1].find('20') >= 0 or
             df[o].dropna().reset_index(drop=True).loc[1].find('19') >= 0)
        )]

        for d in datetime:
            df[d] = pd.to_datetime(df[d])

        numbers = [c for c in self.nbTypes(df).columns if
                   df[c].dtype in [int, float]]
        ids = [col for col in df.columns if
               len(df[col].unique())/len(df) > 0.95 and
               (df[col].nunique()/df[col].count() > 0.99999)]

        bin_num = set(ids).intersection(numbers)

        for c in bin_num:
            if (df[c].dtype not in ['datetime64[ns]', 'object'] and
                    df[c].skew() > 0.2):
                ids.remove(c)

        #--------------------------------------------------------------------------#

        # loop: if there are empty cells
        if sum(df.isnull().sum()) != 0:

            if len(ids) > 0:
                df = df.drop(columns=[c for c in ids])
                print(f'''We have removed the columns {ids} because are likely id columns
        or will have a very poor predictive performance.''')

            high_nan = [c for c in df.columns if
                        df[c].isnull().sum()/len(df[c]) > 0.35]
            df = df.drop(columns=[c for c in high_nan])

            if len(high_nan) == 0:
                print('''We have not found variables with a
                large number of empty points''')
            else:
                print(f'''We have recommoved the following columns because has
        too much empty points: {high_nan}''')

            continuous_columns = [c for c in df.columns if
                                  (df[c].dtype not in ['object', 'datetime64[ns]'] and
                                   ((len(df[c].value_counts()))/len(df))*100 >= 20)
                                  ]
            empty_col = [co for co in continuous_columns if
                         df[co].isnull().cumsum().sum() > 0]

            for col in empty_col:
                kl_mean = np.nan_to_num(
                    kl_div(df[col].fillna(
                        df[col].mean()).iloc[:len(df[col].dropna())],
                        df[col].dropna().values
                    ),
                    neginf=0, posinf=0.0).sum()
                kl_median = np.nan_to_num(
                    kl_div(df[col].fillna(
                        df[col].median()).iloc[:len(df[col].dropna())],
                        df[col].dropna().values),
                    neginf=0, posinf=0.0).sum()

                if kl_mean < kl_median:

                    if kl_mean < 500:
                        df[col] = df[col].fillna(df[col].mean())

                    if kl_median < 500:
                        df[col] = df[col].fillna(df[col].median())

            # after smart NaN management, we drop the rest of NaN
            df = df.dropna()

            if len(df_original) != len(df.dropna()):
                print(f'''After filling some NaN and drop the rest, we have reduce your
        dataset from {len(df_original)} instances to {len(df.dropna())}.''')

            for col in df.columns:

                if df[col].dtype == 'object' and col not in ids:
                    df[col] = df[col].astype('category')

        #----------------------------------------------------------------------#

        # loop: when there are empty cells
        else:

            if len(ids) > 0:
                print(f'''We have not found empty cells in the dataset, but we have
        removed the columns {ids} because are likely id columns or will have
        a very poor predictive performance.''')

            df = df.dropna()
            df = df.drop(columns=[c for c in ids])

            for col in df.columns:
                if df[col].dtype == 'object' and col not in ids:
                    df[col] = df[col].astype('category')

        return df

    def nbPreparation(self, df):
        """
          Returns a prepared dataset with encoded features

          Params:
                df(pd.DataFrame)
                inpur data

          Returns:

            A pandas dataframe wiht NaN management, dtypes management
            and encoded categorical features. There is no other
            data transformation """

        orlen = len(df)  # original data length and
        df = self.nbTypes(df)  # data with datatype transformation
        df = self.nbFillNan(df)  # data with nan management
        cleanlen = len(df)  # data lenght after applying smart NaN management

        # Identify continuous columns
        cont_col = [c for c in df.columns if
                    (df[c].dtype in [int, float] and
                     df[c].nunique() >= 10)
                    ]

        # Identify all binary columns
        # All binary columns and useful binary columns for data augmentation
        bin_total = [col for col in df.columns if
                     len(df[col].value_counts()) == 2]
        binary = [c for c in bin_total if
                  (len(df[c].value_counts().unique()) > 1 and
                   bool(.7 <= (df[c].value_counts().unique()[0] /
                               df[c].value_counts().unique()[1]) >= 1.3) == True)
                  ]

        [binary.remove(col) for col in df.columns if
         df[col].dtype == 'datetime64[ns]' and col in binary
         ]
        [cont_col.remove(col) for col in df.columns if
         df[col].dtype == 'datetime64[ns]' and col in cont_col
         ]

        # Identify multiclass columns
        multi_class = list(set(set(df.columns.tolist())
                               - set(cont_col)
                               - set(bin_total))
                           )
        [multi_class.remove(col) for col in df.columns if
         df[col].dtype == 'datetime64[ns]' and col in multi_class]

        # Encode categorical variables
        encode = binary + multi_class
        ordinal = OrdinalEncoder()
        df[encode] = ordinal.fit_transform(df[encode])
        df[encode] = df[encode].astype('category')

        # Identify datetime and float columns and change dtype to float
        datetime_col = [c for c in df.columns if
                        df[c].dtype == 'datetime64[ns]']

        float_col = list(set(set(cont_col) - set(datetime_col)))

        for c in float_col:

            if df[c].dtype != 'category':
                df[c] = df[c].astype(float)

        # Correctly set the categories for categorical numbers
        for c in bin_total:         # All binary are categorical
            df[c] = df[c].astype('category')

        for c_ in multi_class:  # Multi-class are categorical
            df[c_] = df[c_].astype('category')

        for col in df.columns:
            if df[col].dtype == 'object':  # and col not in id:
                df[col] = df[col].astype('category')

        # Correctly set the categories for categorical numbers
        for c in bin_total:         # All binary are categorical
            df[c] = df[c].astype('category')

        for c_ in multi_class:  # Multi-class are categorical
            df[c_] = df[c_].astype('category')

        print(f'''Original dataset has {orlen} instances and prepared dataset has
                 {len(df)} instances.''')

        return df
