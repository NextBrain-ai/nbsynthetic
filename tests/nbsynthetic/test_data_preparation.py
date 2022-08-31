import numpy as np
import pandas as pd
from pandas.core.arrays.categorical import CategoricalDtype
from data_preparation import SmartBrain
SB = SmartBrain()


def test_nb_types():
	data = [
	['20-12-2019', 'A', 10, 0.34], 
	['01-02-2010', 'B', 15, 0.21], 
	['20-02-2021', 'C', 14, 0.01]
	]
	df = pd.DataFrame(
		data,
		columns=['Date', 'Letter', 'x', 'y']
		)
	desired_types_list = ['datetime64[ns]', 
												'object', 
												'int64', 
												'float64'
												]
	original_types_list = SB.nbTypes(df).dtypes.astype('str').tolist()
	
	assert original_types_list == desired_types_list


def test_nb_fillna():
	data = [
	[1, '20-12-2019', np.nan, 10, 0.04, 1324],
	[2, '01-02-2010', 'B', 15, 0.21, 3400], 
	[3, '20-02-2021', np.nan, 140, np.nan, 2500],
	[4, '12-12-2019', 'C', 1, 0.04, 5600],
	[5, '05-04-2015', 'D', 32, 0.11, 12305], 
	[6, np.nan, 'E', 110, np.nan, 21600],
	[7, '11-10-1980', 'F', 10, 0.01, 54600],
	[8, '01-01-2010', 'G', 55, 0.99, 40000], 
	[9, '20-09-2011', 'H', 44, np.nan, 989]
	]
	df = pd.DataFrame(
		data, 
		columns=['Id', 'Date', 'Letter', 'x', 'y', 'z']
		) 
	newdf = SB.nbFillNan(df)
	desired_len = 6
	obtained_len = len(newdf)
	desired_n_columns = 5
	obtained_n_columns=len(newdf.columns)
	
	assert desired_len == obtained_len


def test_nb_encode():
	data = [
	[1, '20-12-2019', np.nan, 10, 0.04, 1324],
	[2, '01-02-2010', 'B', 15, 0.21, 3400], 
	[3, '20-02-2021', np.nan, 140, np.nan, 2500],
	[4, '12-12-2019', 'C', 1, 0.04, 5600],
	[5, '05-04-2015', 'D', 32, 0.11, 12305], 
	[6, np.nan, 'E', 110, np.nan, 21600],
	[7, '11-10-1980', 'F', 10, 0.01, 54600],
	[8, '01-01-2010', 'G', 55, 0.99, 40000], 
	[9, '20-09-2011', 'H', 44, np.nan, 989]
	]
	df = pd.DataFrame(
		data, 
		columns=['Id', 'Date', 'Letter', 'x', 'y', 'z']
		)

	newdf = SB.nbEncode(df)
	desired_len = 6
	obtained_len = len(newdf)
	desired_category = CategoricalDtype
	obtained_category =  df['Letter'].dtype
	
	assert desired_category == obtained_category