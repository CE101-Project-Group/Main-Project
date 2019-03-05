from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras import backend as K
from keras import callbacks
from keras import metrics
from keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy
import pandas as pd
from DataHelper import DataHelper as dh

class NeuralNetwork:
    def __init__(self, features, output_columns):
        self.features = features
        self.output_columns = output_columns
        self.seed = 4834
        numpy.random.seed(self.seed)
        self.model = Sequential()
        return
    def buildModel(self, column_count):
        self.model.add(Dense(10, input_dim=column_count, activation='relu'))
        self.model.add(Dense(30, activation='relu'))
        self.model.add(Dense(40, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(loss="mean_squared_error", optimizer="adam", metrics=[metrics.mae])
    def train(self, dataframe):
        x = dataframe[self.features]
        x = dh.standardizeData(x)
        y = dataframe[self.output_columns].values

        #split rows 1/3 for test 2/3 for train
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=self.seed)
        self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=250, batch_size=16)        
        return
    def summary(self):
        return self.model.summary()
    def save(self, name):
        self.model.save(name)
        return
    def load(self, name):
        self.model = load_model(name)
        return
    def predict(self, dataframe):
        result = pd.DataFrame()
        result['Id'] = dataframe['Id'].values.tolist()
        del dataframe['Id']
        x_test = dh.standardizeData(dataframe)
        result['SalePrice'] = self.model.predict(x_test)
        return result
    def loss(self, feature, row, target):
        return