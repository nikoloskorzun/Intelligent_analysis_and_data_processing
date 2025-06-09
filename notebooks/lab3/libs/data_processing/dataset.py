import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Optional, Union, List, Tuple

class Dataset(pd.DataFrame):
    _target_features: List[str] = []
    _name: str = 'Датасет'
    
    # Приватные атрибуты для хранения разделенных данных
    _data_X_valid: Optional[pd.DataFrame] = None
    _data_Y_valid: Optional[pd.DataFrame] = None
    _data_X_test: Optional[pd.DataFrame] = None
    _data_Y_test: Optional[pd.DataFrame] = None
    _data_X_train: Optional[pd.DataFrame] = None
    _data_Y_train: Optional[pd.DataFrame] = None

    def __init__(
        self, 
        data: Optional[Union[pd.DataFrame, dict, List]] = None, 
        name: str = 'Датасет',
        target_features: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Инициализация Dataset.
        
        Parameters:
            data: Входные данные (DataFrame, dict, list и т.д.)
            name: Название датасета
            target_features: Список целевых признаков
            **kwargs: Дополнительные аргументы для pd.DataFrame
        """
        super().__init__(data, **kwargs)
        self._name = name
        self._target_features = target_features if target_features is not None else []

    @property
    def name(self) -> str:
        """Название датасета"""
        return self._name
    
    @name.setter
    def name(self, value: str) -> None:
        self._name = value
        
    @property
    def target_features(self) -> List[str]:
        """Список целевых признаков"""
        return self._target_features
    
    @target_features.setter
    def target_features(self, value: List[str]) -> None:
        self._target_features = value
        
    # Свойства для доступа к данным
    @property
    def X_train(self) -> pd.DataFrame:
        """Признаки тренировочного набора"""
        if self._data_X_train is None:
            raise ValueError("Данные не разделены. Сначала вызовите split_data()")
        return self._data_X_train
    
    @property
    def Y_train(self) -> pd.DataFrame:
        """Целевые переменные тренировочного набора"""
        if self._data_Y_train is None:
            raise ValueError("Данные не разделены. Сначала вызовите split_data()")
        return self._data_Y_train
    
    @property
    def X_valid(self) -> pd.DataFrame:
        """Признаки валидационного набора"""
        if self._data_X_valid is None:
            raise ValueError("Данные не разделены. Сначала вызовите split_data()")
        return self._data_X_valid
    
    @property
    def Y_valid(self) -> pd.DataFrame:
        """Целевые переменные валидационного набора"""
        if self._data_Y_valid is None:
            raise ValueError("Данные не разделены. Сначала вызовите split_data()")
        return self._data_Y_valid
    
    @property
    def X_test(self) -> pd.DataFrame:
        """Признаки тестового набора"""
        if self._data_X_test is None:
            raise ValueError("Данные не разделены. Сначала вызовите split_data()")
        return self._data_X_test
    
    @property
    def Y_test(self) -> pd.DataFrame:
        """Целевые переменные тестового набора"""
        if self._data_Y_test is None:
            raise ValueError("Данные не разделены. Сначала вызовите split_data()")
        return self._data_Y_test

    def split_data(
        self, 
        test_size: float = 0.2, 
        valid_size: float = 0.1,
        random_state: Optional[int] = None,
        shuffle: bool = True
    ) -> None:
        """
        Разделение данных на train/valid/test
        
        Parameters:
            test_size: Доля тестовой выборки
            valid_size: Доля валидационной выборки (от оставшихся после test данных)
            random_state: Seed для воспроизводимости
            shuffle: Перемешивать ли данные перед разделением
        """
        if not self._target_features:
            raise ValueError("Не заданы целевые признаки (target_features)")
            
        # Получаем признаки и целевую переменную
        X = self.drop(columns=self._target_features)
        y = self[self._target_features]
        
        # Сначала разделяем на train+valid и test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=shuffle
        )
        
        # Затем разделяем train+valid на train и valid
        val_size_adjusted = valid_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, 
            random_state=random_state, shuffle=shuffle
        )

        self._data_X_train = X_train
        self._data_Y_train = y_train
        self._data_X_valid = X_val
        self._data_Y_valid = y_val
        self._data_X_test = X_test
        self._data_Y_test = y_test


        """ old
        # Сохраняем разделенные данные как DataFrame
        self._data_X_train = pd.DataFrame(X_train, columns=self.columns.drop(self._target_features))
        self._data_Y_train = pd.DataFrame(y_train, columns=self._target_features)
        self._data_X_valid = pd.DataFrame(X_val, columns=self.columns.drop(self._target_features))
        self._data_Y_valid = pd.DataFrame(y_val, columns=self._target_features)
        self._data_X_test = pd.DataFrame(X_test, columns=self.columns.drop(self._target_features))
        self._data_Y_test = pd.DataFrame(y_test, columns=self._target_features)
        """
    def get_train_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Получить тренировочные данные"""
        return self.X_train, self.Y_train
        
    def get_valid_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Получить валидационные данные"""
        return self.X_valid, self.Y_valid
        
    def get_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Получить тестовые данные"""
        return self.X_test, self.Y_test

