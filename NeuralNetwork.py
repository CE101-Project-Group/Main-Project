from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.utils import to_categorical
from keras import backend as K
from keras import callbacks
from keras import metrics
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
import numpy
from sklearn.preprocessing import RobustScaler
import pandas as pd
from DataHelper import DataHelper as dh
from TensorErrorFunctions import TensorErrorFunctions as tef


class NeuralNetwork:
    def __init__(self, features, output_columns, model = None):
        self.features = features
        self.output_columns = output_columns
        self.seed = 4834
        numpy.random.seed(self.seed)
        if model == None:
            self.model = self.buildModel() 
        else:
            self.model = model
        return
    def buildModel(self):
        model = Sequential()
        model.add(Dense(10, input_dim=len(self.features), activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(40, activation='relu'))
        model.add(Dense(1))
        model.compile(loss="mean_squared_error", optimizer="Adam", metrics=[tef.max_error, tef.mean_diff])
        return model
    def train(self, dataframe, evaluate=False, model=None):
        if model == None:
            model = self.model

        x = dataframe[self.features]
        x = dh.standardizeData(x)
        y = dataframe[self.output_columns].values

        #split rows 1/3 for test 2/3 for train
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=self.seed)
        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1400, batch_size=8, verbose=0 if evaluate else 1)
        if evaluate:
            return model.evaluate(x_test, y_test,verbose=0)
        else:
            return
    def summary(self):
        return self.model.summary()
    def save(self, name):
        self.model.save(name)
        return
    def load(self, name):
        self.model = load_model(name, custom_objects={'mean_diff': tef.mean_diff, 'max_error': tef.max_error, 'min_error': tef.min_error})
        return
    def predict(self, dataframe, training_data):
        result = pd.DataFrame()
        result['Id'] = dataframe['Id'].values.tolist()
        del dataframe['Id']
        x_test = dh.standardizeData(dataframe)
        x_train = dh.standardizeData(training_data)
        result['SalePrice'] = self.model.predict(x_test)
        return result
    def loss(self, feature, row, target):
        return