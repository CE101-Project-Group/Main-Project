import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
class DataHelper:
    @staticmethod
    def getMissingDataInformation(dataframe):
        total = dataframe.isnull().sum().sort_values(ascending=False)
        percent = (dataframe.isnull().sum()/dataframe.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        return missing_data

    @staticmethod
    def standardizeData(dataframe):
        warnings.filterwarnings("ignore")
        return StandardScaler().fit_transform(dataframe)

    @staticmethod
    def processInput(train):
        # Convert potential numerical  categorical data to numerical data
        ordinal = [
            ('ExterQual', ['Po', 'Fa', 'TA', 'Gd', 'Ex'], [1, 2, 3, 4, 5], lambda x: 0),
            ('ExterCond', ['Po', 'Fa', 'TA', 'Gd', 'Ex'], [1, 2, 3, 4, 5], lambda x: 0),
            ('BsmtQual', ['Po', 'Fa', 'TA', 'Gd', 'Ex'], [1, 2, 3, 4, 5], lambda x: 0),
            ('BsmtCond',  ['Po', 'Fa', 'TA', 'Gd', 'Ex'], [1, 2, 3, 4, 5], lambda x: 0),
            ('HeatingQC', ['Po', 'Fa', 'TA', 'Gd', 'Ex'], [1, 2, 3, 4, 5], lambda x: 0),
            ('KitchenQual', ['Po', 'Fa', 'TA', 'Gd', 'Ex'], [1, 2, 3, 4, 5], lambda x: 0),
            ('FireplaceQu', ['Po', 'Fa', 'TA', 'Gd', 'Ex'], [1, 2, 3, 4, 5], lambda x: 0),
            ('GarageQual', ['Po', 'Fa', 'TA', 'Gd', 'Ex'], [1, 2, 3, 4, 5], lambda x: 0),
            ('GarageCond', ['Po', 'Fa', 'TA', 'Gd', 'Ex'], [1, 2, 3, 4, 5], lambda x: 0),
            ('PoolQC', ['Po', 'Fa', 'TA', 'Gd', 'Ex'], [1, 2, 3, 4, 5], lambda x: 0),
            ('Street', ['Pave', 'Grvl'], [1, 2], lambda x: 0),
            ('Alley', ['Pave', 'Grvl'], [1, 2], lambda x: 0),
            ('LotShape', ['IR3', 'IR2', 'IR1', 'Reg'], [1, 2, 3, 4], lambda x: 0),
            ('Utilities', ['ELO', 'NoSeWa', 'NoSewr', 'AllPub'], [0, 1, 2, 3], lambda x: 0),
            ('LandSlope', ['Sev', 'Mod', 'Gtl'], [1, 2, 3], lambda x: 0),
            ('BsmtExposure', ['No', 'Mn', 'Av', 'Gd'], [1, 2, 3, 4], lambda x: 0),
            ('BsmtFinType1', ['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], [1, 2, 3, 4, 5, 6], lambda x: 0),
            ('BsmtFinType2', ['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], [1, 2, 3, 4, 5, 6], lambda x: 0),
            ('GarageFinish', ['Unf', 'RFn', 'Fin'], [1, 2, 3], lambda x: 0),
        #    ('PavedDrive', ['N', 'P', 'Y'], [1, 2, 3], lambda x: 0),
            ('Fence', ['MnWw', 'GdWo', 'MnPrv', 'GdPrv'], [1, 2, 3, 4], lambda x: 0)
        ]


        def convert(df, ordinal):
            for ordinalData in ordinal:
                df[ordinalData[0]] = df[ordinalData[0]].fillna(ordinalData[3](df[ordinalData[0]]))
                df[ordinalData[0]] = df[ordinalData[0]].replace(ordinalData[1], ordinalData[2])
            return df

        train = convert(train, ordinal=ordinal)

        # Create dummy values
        train = pd.get_dummies(train)

        # Remove columns with more than 60% missing rows
        # train = train.dropna(axis='columns', how='any', thresh=train.shape[0] * 0.6, subset=None, inplace=False)

        # Fill missing data with average
        train = train.fillna(train.mean());

        #Fill missing data with 0
        #train = train.fillna(0)

        # Remove columns with all the same value
        # train = train.drop(train.std()[(train.std() == 0)].index, axis=1)
        return train
