import numpy as np
import pandas as pd
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, \
    QuantileTransformer, KBinsDiscretizer
from vgan import GAN


def test_columns_type():
  data = [
        [31, 'Mike', 'A', 10, 0.04, 1324],
        [22, 'Jules', 'B', 15, 0.21, 3400], 
        [23, 'Anne', 'C', 140, 0.12, 2500],
        [34, 'John', 'D', 1, 0.04, 5600],
        [25, 'Liz', 'E', 32, 0.11, 12305], 
        [26, 'Tom', 'F', 110, 0.05, 21600],
        [27, 'Martha', 'G', 10, 0.01, 54600],
        [38, 'Lilly', 'H', 55, 0.99, 40000], 
        [29, 'Melissa', 'I', 44, 0.21, 989]
        ]
  df = pd.DataFrame(
      data, 
      columns=['Age', 'Name', 'Letter', 'x', 'y', 'z']
      ) 
  categorical = ['Name', 'Letter']
  for c in categorical:
    df[c] = df[c].astype('category')

  numerical = ['Age', 'x', 'y', 'z']
  for c in numerical:
    df[c] = df[c].astype(float)

  numerical_columns_selector = selector(
        dtype_exclude=CategoricalDtype
  )
  categorical_columns_selector = selector(
      dtype_include=CategoricalDtype
  )
  numerical_columns = numerical_columns_selector(df)
  categorical_columns = categorical_columns_selector(df)
  
  assert numerical == numerical_columns
  assert categorical == categorical_columns



def test_data_transformation():
  data = [
          [31, 1, 1, 10, 0.04, 1324],
          [22, 2, 2, 15, 0.21, 3400], 
          [23, 3, 3, 140, 0.12, 2500],
          [34, 4, 3, 1, 0.04, 5600],
          [25, 5, 3, 32, 0.11, 12305], 
          [26, 6, 1, 110, 0.05, 21600],
          [27, 7, 2, 10, 0.01, 54600],
          [38, 8, 4, 55, 0.99, 40000], 
          [29, 9, 1, 44, 0.21, 989]
          ]
  df = pd.DataFrame(
    data, 
    columns=['Age', 'Name', 'Letter', 'x', 'y', 'z']
    ) 
  categorical = ['Name', 'Letter']
  for c in categorical:
    df[c] = df[c].astype('category')

  numerical = ['Age', 'x', 'y', 'z']
  for c in numerical:
    df[c] = df[c].astype(float)

  numerical_columns_selector = selector(
      dtype_exclude=CategoricalDtype
  )
  categorical_columns_selector = selector(
    dtype_include=CategoricalDtype
  )
  numerical_columns = numerical_columns_selector(df)
  categorical_columns = categorical_columns_selector(df)

  categorical_scaler = make_pipeline(
      MinMaxScaler(
          feature_range=(-1, 1),
          clip=True
      )
  )
  if len(df) > 99:
      n_quantiles = 100
  else:
      n_quantiles = len(df)

  numerical_scaler = make_pipeline(
      # A quantile transform will map a variable’s
      # probability distribution to another probability
      # distribution.By performing a rank
      # transformation, a quantile transform smooths out
      # unusual distributions and is less influenced by
      # outliers than scaling methods.
      # Ref: https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-transformer
      QuantileTransformer(
          n_quantiles=n_quantiles,
          output_distribution='uniform',
      ),
      MinMaxScaler(
          feature_range=(-1, 1),
          clip=True
      )
  )
  scaled_X = df.copy()
  for cat_c in categorical_columns:
      scaled_X[cat_c] = categorical_scaler.fit_transform(
          np.array(df[cat_c]).reshape(-1, 1)
      ).flatten()

  for num_c in numerical_columns:
      scaled_X[num_c] = numerical_scaler.fit_transform(
          np.array(df[num_c]).reshape(-1, 1)
      ).flatten()

  assert sum(scaled_X.min())/6 == -1
  assert sum(scaled_X.max())/6 == 1



def test_synthetic_data():
  data = [
          [31, 1, 1, 10, 0.04, 1324],
          [22, 2, 2, 15, 0.21, 3400], 
          [23, 3, 3, 140, 0.12, 2500],
          [34, 4, 3, 1, 0.04, 5600],
          [25, 5, 3, 32, 0.11, 12305], 
          [26, 6, 1, 110, 0.05, 21600],
          [27, 7, 2, 10, 0.01, 54600],
          [38, 8, 4, 55, 0.99, 40000], 
          [29, 9, 1, 44, 0.21, 989]
          ]
  df = pd.DataFrame(
    data, 
    columns=['Age', 'Name', 'Letter', 'x', 'y', 'z']
    ) 
  categorical = ['Name', 'Letter']
  for c in categorical:
    df[c] = df[c].astype('category')

  numerical = ['Age', 'x', 'y', 'z']
  for c in numerical:
    df[c] = df[c].astype(float)

  numerical_columns_selector = selector(
      dtype_exclude=CategoricalDtype
  )
  categorical_columns_selector = selector(
    dtype_include=CategoricalDtype
  )
  numerical_columns = numerical_columns_selector(df)
  categorical_columns = categorical_columns_selector(df)

  categorical_scaler = make_pipeline(
      MinMaxScaler(
          feature_range=(-1, 1),
          clip=True
      )
  )

  if len(df) > 99:
      n_quantiles = 100
  else:
      n_quantiles = len(df)

  numerical_scaler = make_pipeline(
      QuantileTransformer(
          n_quantiles=n_quantiles,
          output_distribution='uniform',
      ),
      MinMaxScaler(
          feature_range=(-1, 1),
          clip=True
      )
  )
  scaled_X = df.copy()
  for cat_c in categorical_columns:
      scaled_X[cat_c] = categorical_scaler.fit_transform(
          np.array(df[cat_c]).reshape(-1, 1)
      ).flatten()

  for num_c in numerical_columns:
      scaled_X[num_c] = numerical_scaler.fit_transform(
          np.array(df[num_c]).reshape(-1, 1)
      ).flatten()

  scaled_data = scaled_X
  gan = GAN(
            number_of_features=6,
            learning_rate=0.0002,
            dropout=0.5,
        )
  G_loss, D_loss = gan.train(
      scaled_data=np.array(scaled_X),
      epochs=1,
      batch_size=52,
      )
  samples = 10
  x_synthetic,\
          y_synthetic = gan.create_fake_samples(
              batch_size=samples
          )
  newdf = pd.DataFrame(
          x_synthetic,
          columns=df.columns
      )
  for cat_c in categorical_columns:
      if np.unique(df[cat_c]).shape[0] > 1:
          newdf[cat_c] = categorical_scaler.inverse_transform(
              np.array(
                  newdf[cat_c]).reshape(-1, 1)
          )
          kbins = KBinsDiscretizer(
              n_bins=np.unique(df[cat_c]).shape[0],
              encode='ordinal',
              strategy='uniform'
          )
          newdf[cat_c] = kbins.fit_transform(
              np.array(newdf[cat_c]).reshape(-1, 1)
          ).astype(int)
          newdf[cat_c] = newdf[cat_c].astype('category')
      else:
          pass

  for num_c in numerical_columns:
      newdf[num_c] = numerical_scaler.inverse_transform(
          np.array(
              newdf[num_c]).reshape(-1, 1)
      ).flatten().astype('float64')

  for cat_c in categorical_columns:
      if np.unique(df[cat_c]).shape[0] == 2:
          newdf[cat_c].replace(
              [np.unique(newdf[cat_c])[0],
                np.unique(newdf[cat_c])[1]],
              [np.unique(df[cat_c])[0],
                np.unique(df[cat_c])[1]],
              inplace=True
          )
      else:
          pass

  assert df.dtypes.astype('str').tolist() == newdf.dtypes.astype('str').tolist()
  assert len(newdf) == samples

  if __name__ == '__main__':
    pytest.main([__file__])