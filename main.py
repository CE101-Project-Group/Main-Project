import pandas as pd
import numpy
import seaborn as sns
from DataHelper import DataHelper as dh
from NeuralNetwork import NeuralNetwork

def test(model):
    train = pd.read_csv("input/train.csv", index_col=0)
    train = pd.get_dummies(train)
    train = train.fillna(train.mean())

    test = pd.read_csv("input/test.csv")
    ignore_list = ['SalePrice']
    test = pd.get_dummies(test)
    test = test.fillna(test.mean());
    
    features_test = list(set(test.columns.values) - set(ignore_list))
    features_train = list(set(train.columns.values) - set(ignore_list))
    
    missing_columns = set(features_train) - set(features_test)

    for col in missing_columns:
        test[col] = 0

    features = list(set(test.columns.values) - set(ignore_list))
    network = NeuralNetwork(features, ['SalePrice'])
    network.load(model)
    prediction = network.predict(test)
    print(prediction)

def train(model):
    # Build and train model
    train = pd.read_csv("input/train.csv", index_col=0)
    ignore_list = ['id', 'SalePrice']

    # Print missing data information
    #print(dh.getMissingDataInformation(train))

    # Create dummy values
    train = pd.get_dummies(train)

    # Fill missing data with average
    train = train.fillna(train.mean());

    #Fill missing data with 0
    #train = train.fillna(0)


    features = list(set(train.columns.values) - set(ignore_list))
    network = NeuralNetwork(features, ['SalePrice'])
    network.buildModel(len(features))
    network.train(train)
    network.save(model)


    
if __name__ == "__main__":
    test('model.h5')
    #train('model.h5')