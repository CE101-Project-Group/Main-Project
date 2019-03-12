import pandas as pd
import numpy
import os
import seaborn as sns
from DataHelper import DataHelper as dh
from NeuralNetwork import NeuralNetwork
from BuildModels import BuildModels

def test(model):
    train = pd.read_csv("input/train.csv", index_col=0)
    train = train.drop(train[(train['GrLivArea'] > 4112) & (train['SalePrice'] < 300000)].index)
    train = dh.processInput(train)

    test = pd.read_csv("input/test.csv")
    test = dh.processInput(test)
    ignore_list = ['SalePrice']
    
    features_test = list(set(test.columns.values) - set(ignore_list))
    features_train = list(set(train.columns.values) - set(ignore_list))
    
    missing_columns = set(features_train) - set(features_test)

    for col in missing_columns:
        test[col] = 0

    features_test = list(set(test.columns.values) - set(ignore_list))
    features_train = list(set(train.columns.values) - set(ignore_list))
    
    additional_columns = set(features_test) - set(features_train)

    for col in additional_columns:
        if col != 'Id':
            del test[col]

    features = list(set(test.columns.values) - set(ignore_list))
    network = NeuralNetwork(features, ['SalePrice'])
    network.load(model)
    prediction = network.predict(test,train)

    result = pd.DataFrame({'ID': prediction.Id, 'SalePrice': prediction.SalePrice})
    result.to_csv('submission.csv', index=False)
    print(prediction)
def train(model):
    # Build and train model
    train = pd.read_csv("input/train.csv", index_col=0)
    ignore_list = ['id', 'SalePrice']

    # Print missing data information
    #print(dh.getMissingDataInformation(train))
    
    # Remove outliers
    train = train.drop(train[(train['GrLivArea'] > 4112) & (train['SalePrice'] < 300000)].index)
    train = dh.processInput(train)
    
    features = list(set(train.columns.values) - set(ignore_list))
    network = NeuralNetwork(features, ['SalePrice'], model=BuildModels.KQV15(len(features)))

    # Get all models from BuildModels class
    '''
    kmodels = {}
    for key in BuildModels.__dict__:
        if isinstance(BuildModels.__dict__[key], staticmethod):
            kmodels[key] = BuildModels.__dict__[key].__func__(len(features))

    for key, kmodel in kmodels.items():
        scores = network.train(train, evaluate=True, model=kmodel)
        print('Model ' + key + '\nMax Error: ' + str(scores[1]) + '\nMean Error: ' + str(scores[2]))
    '''
    network.train(train)
    network.save(model)

if __name__ == "__main__":
    #if os.path.exists('model.h5'):
    #    os.remove('model.h5')
    #if os.path.exists('submission.csv'):
    #    os.remove('submission.csv')
    train('QV2.h5')
    test('QV2.h5')