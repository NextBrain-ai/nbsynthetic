# -*- coding: utf-8 -*-

"""
Created on 25/03/2017
@author: Javier Marin Dataset preparation following some heuristis.
"""

import random 
import pandas as pd 
from pandas.core.arrays.categorical import CategoricalDtype
import numpy as np 
import scipy
import sklearn
import sklearn.model_selection
from sklearn import preprocessing

kl_div =  scipy.special.kl_div 
stats =  scipy.stats 
train_test_split = sklearn.model_selection.train_test_split 

class SmartBrain(object):
  """
    Functions to prepare datasets focus in three areas:
      1- Handling missing data.
      2- Augment dataset if is necessary
      3- Set the correct dtypes
      4- Encode categorical features

    Parameters:
    -----------
    df: data

    Returns:
    --------
    Utils for smart management of NaN values, dtypes ,
    data augmentation and encoding-
  
   """


  def __init__(self):

    self


  def nbTypes(self, df):

    objects, datetime = [],[]
    [objects.append(c) for c in df.columns if df[c].dtype == object]
    for o in objects:
      if (df[o].dropna().reset_index(drop=True).loc[1].find('-') >= 0 and
          (df[o].dropna().reset_index(drop=True).loc[1].find('20') >= 0 or
           df[o].dropna().reset_index(drop=True).loc[1].find('19') >= 0)) :
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

    objects= [c for c in df.columns if df[c].dtype == object]
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
                        (df[col].nunique()/df[col].count() > 0.99999) and
                        col not in datetime]

    bin_num = set(ids).intersection(numbers)
    
    for c in bin_num:
      if (df[c].dtype not in ['datetime64[ns]', 'object'] and
          df[c].skew() > 0.2):
          ids.remove(c)

    #--------------------------------------------------------------------------#

    # loop: if there are empty cells
    if sum(df.isnull().sum()) != 0:

      if len(ids) > 0:
        for c in ids:
          if c not in datetime:
            df = df.drop(columns=[c])
            print(f'''We have removed the columns {ids} because are likely id columns
          or will have a very poor predictive performance.''')

      high_nan = [c for c in df.columns if 
                                df[c].isnull().sum()/len(df[c]) > 0.35]
      df = df.drop(columns=[c for c in high_nan])

      if len(high_nan) == 0:
        print('''We have not found variables with a
                large number of empty points''')
      else:
        print(f'''We have removed the following columns because has
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

      #after smart NaN management, we drop the rest of NaN
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
        
      df = df.dropna()
      for c in ids:
        if c not in datetime:
          df = df.drop(columns=[c])
          print(f'''We have not found empty cells in the dataset, but we have
        removed the columns {ids} because are likely id columns or will have
        a very poor predictive performance.''')

      for col in df.columns:
        if df[col].dtype == 'object' and col not in ids:
          df[col] = df[col].astype('category')
          

    return df



  def nbEncode(self, df):
    """
        Returns an encoded dataset 

        Params:
              df(pd.DataFrame)
              input data

        Returns:
      
          A pandas dataframe wiht encoded categorical features. 
    """
    df = self.nbFillNan(df)
    cont_col = [
        c for c in df.columns if (
          df[c].dtype in [int , float] and
          np.unique(df[c]).shape[0] / len(df) > 0.85
          )
        ]
    le = preprocessing.LabelEncoder()
    
    for c in df.columns:
      if (isinstance(df[c].dtype, CategoricalDtype) and
          c not in cont_col):
        df[c] = le.fit_transform(df[c])
        df[c] = df[c].astype('category')
      
    return df