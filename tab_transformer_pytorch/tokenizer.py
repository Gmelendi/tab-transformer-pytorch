from __future__ import annotations
import pandas as pd
from typing import Dict

class TabTokenizer:
    """Tokenizer class for tabular data.
    
    Tokenize categorical columns from a pandas DataFrame.
    The tokenization is perform using `cat.codes` API from pandas. In order to prevent the overlap of codes
    from different columns an offset is applied sequentially to each column.
    """

    def __init__(self) -> None:

        self.offsets = None
        self._cats: Dict[str, pd.Categorical] = {}

    @property
    def vocab_len(self):
        return sum(len(col) + 1 for col in self._cats.values())

    def fit(self, data: pd.DataFrame) -> TabTokenizer:

        data = data.select_dtypes(include='category')
        self.offsets = data.nunique().add(1).cumsum().shift(fill_value=0).add(1)

        for col in data.columns:
            self._cats[col] = data[col].cat.categories

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:

        if len(self._cats) == 0 or self.offsets is None:
            raise AssertionError('Must call .fit before transform')
        
        for col in data.select_dtypes(include='category'):
            data[col] = data[col].cat.set_categories(self._cats[col])
            data[col] = data[col].cat.codes + self.offsets.loc[col]

        return data

