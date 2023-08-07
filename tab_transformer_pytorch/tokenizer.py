import pandas as pd

class TabTokenizer:

    def __init__(self) -> None:

        self.itos = {}
        self.stoi = {}
        self.cod2cat = {}
        self._cats = {}

    def fit(self, data: pd.DataFrame):

        data = data.select_dtypes(include='category')
        offsets = data.nunique().cumsum().shift(fill_value=0)

        for col in data.columns:
            self._cats[col] = data[col].cat.categories
            self.cod2cat[col] = {code:code + offsets.loc[col] for code in data[col].cat.codes.unique()}
        
    def encode(self, x: pd.DataFrame):

        if len(self.stoi) == 0:
            raise ValueError('Must call .fit previous to encode')
        


