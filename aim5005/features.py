import numpy as np
from typing import List, Tuple
### YOU MANY NOT ADD ANY MORE IMPORTS (you may add more typing imports)

class MinMaxScaler:
    def _init_(self):
        self.minimum = None
        self.maximum = None
        
    def _check_is_array(self, x:np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it'a not a np.ndarray and return. If it can't be cast raise an error
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        assert isinstance(x, np.ndarray), "Expected the input to be a list"
        return x
        
    
    def fit(self, x:np.ndarray) -> None:   
        x = self._check_is_array(x)
        self.minimum=x.min(axis=0)
        self.maximum=x.max(axis=0)
        
    def transform(self, x:np.ndarray) -> list:
        """
        MinMax Scale the given vector
        """
        x = self._check_is_array(x)
        diff_max_min = self.maximum - self.minimum
        
        # TODO: There is a bug here... Look carefully! 
        return (x-self.minimum)/diff_max_min
    
    def fit_transform(self, x:list) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)

class StandardScaler:
    def _init_(self):
        self.mean = None
        self.scale = None

    def fit(self, X):
        X = np.array(X)
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)
        self.scale[self.scale==0] = 1
        return self

    def transform(self, X):
        if self.mean is None or self.scale is None:
            raise ValueError("StandardScaler has not been fitted")
        X = np.array(X)
        return (X - self.mean) / self.scale

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    

class LabelEncoder:
    def _init_(self):
        self.classes_ = None

    def fit(self, X):
        self.classes_ = np.unique(X)
        return self

    def transform(self, X):
        if self.classes_ is None:
            raise ValueError("LabelEncoder has not been fitted.")
        return np.array([np.where(self.classes_ == label)[0][0] for label in X])

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)