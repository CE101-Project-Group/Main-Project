import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataHelper:
    @staticmethod
    def getMissingDataInformation(dataframe):
        total = dataframe.isnull().sum().sort_values(ascending=False)
        percent = (dataframe.isnull().sum()/dataframe.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        return missing_data

    @staticmethod
    def standardizeData(dataframe):
        return StandardScaler().fit_transform(dataframe);