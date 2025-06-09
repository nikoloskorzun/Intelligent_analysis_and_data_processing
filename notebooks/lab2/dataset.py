import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from typing import Callable, Any
class Dataset:
    _target_features: list[str]
    _data : pd.DataFrame
    _data_X : pd.DataFrame
    _data_Y : pd.DataFrame
    _name: str = 'Датасет'
   
    
    _data_X_valid : np.array
    _data_Y_valid : np.array
    
    _data_X_test : np.array
    _data_Y_test : np.array
    
    _data_X_train : np.array
    _data_Y_train : np.array

    _share_split_test: np.float64
    _share_split_valid: np.float64
    _seed : int
    def __init__(self, df:pd.DataFrame, target_features:list[str], share_split_test=0.2, share_split_valid = 0.1, seed=1337):
        self._target_features = target_features
        self._data = df
        self._set_X_Y()
        self.set_share_split_dataset(share_split_test, share_split_valid, seed)

    def set_dataset(self, df:pd.DataFrame):
        self._data = df
        self._set_X_Y()
    def set_share_split_dataset(self, share_split_test=0.2, share_split_valid = 0.1, seed=1337):
        assert share_split_test + share_split_valid <= 0.5
        assert share_split_test != 0.0
        self._share_split_test = share_split_test
        self._share_split_valid = share_split_valid
        self._seed = seed
        self._set_TRAIN_TEST_VALID()
        
        
        
    def _set_TRAIN_TEST_VALID(self):
        self._data_X_train, self._data_X_test, self._data_Y_train, self._data_Y_test = train_test_split(self.X, self.Y, test_size=(self._share_split_test + self._share_split_valid), random_state=self._seed)
        self._data_X_test, self._data_X_valid, self._data_Y_test, self._data_Y_valid = train_test_split(self._data_X_test, self._data_Y_test, test_size=(self._share_split_valid / (self._share_split_test + self._share_split_valid)), random_state=self._seed)
    
        
        
        
        
    def _set_X_Y(self):
        self._data_X = self._data.drop(self._target_features, axis=1)
        self._data_Y = self._data[self._target_features]
        



    
    @property
    def X(self)->pd.DataFrame:
        return self._data_X
    @property
    def Y(self)->pd.DataFrame:
        return self._data_Y

    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, name: str):
        self._name = name
    def __str__(self):
        return f"Датасет {self.name} c {self.X.shape[1]} + 1 признаками"
    def __repl__(self):
        return self.__str__
    def performe_operation(self, func: Callable[[pd.DataFrame, ...], pd.DataFrame], **params: Any):                
        self._data = func(self._data, **params)
        self._set_X_Y()

    @property
    def X_test(self)->pd.DataFrame:
        return self._data_X_test
    @property
    def Y_test(self)->pd.DataFrame:
        return self._data_Y_test

    @property
    def X_train(self)->pd.DataFrame:
        return self._data_X_train
    @property
    def Y_train(self)->pd.DataFrame:
        return self._data_Y_train    
        
    @property
    def X_valid(self)->pd.DataFrame:
        return self._data_X_valid
    @property
    def Y_valid(self)->pd.DataFrame:
        return self._data_Y_valid

  