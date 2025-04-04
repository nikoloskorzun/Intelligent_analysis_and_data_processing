import pandas as pd
import numpy as np


from typing import Callable, Any
class Model:
    _name: str = 'Модель'
    def __init__(self, model):
        self.model = model

    def fit(self, x, y):
        return self.model.fit(x, y)
    
    def predict(self, x):
        return self.model.predict(x)
        
    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, name: str):
        self._name = name
    def __str__(self):
        return f"Модель {self.name}"
    def __repl__(self):
        return self.__str__
    def test_model(self, dataset, scores: dict[str, Callable[[np.array, np.array], Any]], test_type="train"):
        if test_type == "train":
            X = dataset.X_train
            Y = dataset.Y_train
        elif test_type == "test":
            X = dataset.X_test
            Y = dataset.Y_test
        elif test_type == "valid":
            X = dataset.X_valid
            Y = dataset.Y_valid
        else:
            raise "error"
        Y_predict = self.model.predict(X)
        results = {}
        
        for s in scores:
            results[s] = scores[s](Y_predict, Y)
        return results
    
      