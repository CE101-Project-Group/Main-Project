from keras import backend as K
class TensorErrorFunctions:
    def mean_diff(y_true, y_pred):
        return K.mean(K.abs(y_true - y_pred))
    def max_error(y_true, y_pred):
        return K.max(K.abs(y_true - y_pred))
    def min_error(y_true, y_pred):
        return K.min(K.abs(y_true - y_pred))
