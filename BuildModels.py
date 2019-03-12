from TensorErrorFunctions import TensorErrorFunctions as tef
import keras
from keras.models import Sequential
from keras.layers import Dense, ThresholdedReLU, Dropout, Activation, BatchNormalization
from keras import regularizers

class BuildModels:
    @staticmethod
    def QV2(columns):
        model = Sequential()
        model.add(Dense(int(columns/2), input_dim=columns, activation='relu'))
        model.add(ThresholdedReLU())
        model.add(Dense(int(columns/5), activation='relu'))
        model.add(Dense(1))
        model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adadelta(), metrics=[tef.max_error, tef.mean_diff])
        return model
    @staticmethod
    def KQV15(columns):
        model = Sequential()
        model.add(Dense(int(columns/1.5), input_dim=columns, activation='relu'))
        model.add(ThresholdedReLU())
        model.add(Dense(int(columns/2), activation='relu'))
        model.add(ThresholdedReLU())
        model.add(Dense(int(columns/5), activation='relu'))
        model.add(Dense(1))
        model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adadelta(), metrics=[tef.max_error, tef.mean_diff])
        return model        