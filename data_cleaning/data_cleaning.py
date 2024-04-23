import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Reading data
def read_data(filename,separator="\c"):
    df = pd.read_csv(filename, sep=separator, engine='python')

    return df

# View data information
def data_overview(df):
    print(df.head(5))
    print(df.info())

def descritptive_stats(df):
    

data = read_data("Ames_Housing_Data1.tsv","\t")
# data_structure_information(data)

hous_num = data.select_dtypes(include = ['float64', 'int64'])
data_overview(hous_num)