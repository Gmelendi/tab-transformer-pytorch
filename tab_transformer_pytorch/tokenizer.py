import pandas as pd
from pandas.api.types import union_categoricals

class TabTokenizer:

    def __init__(self) -> None:

        self.itos = {}
        self.stoi = {}
        self.vocab = {}

    def fit(self, data: pd.DataFrame):

        data = data.select_dtypes(include='category')
        for col in data.columns:
            self.vocab[col] = list(data[col].cat.categories)
        
        i = 0
        for col in self.vocab:
            
        self.stoi = {i:token for i, token in enumerate(all_categories.cat.categories)}
        self.itos = {token:i for i, token in enumerate(all_categories.cat.categories)}
        
    def encode(self, x: pd.DataFrame):

        if len(self.stoi) == 0:
            raise ValueError('Must call .fit previous to encode')
        


