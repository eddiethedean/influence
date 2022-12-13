import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_chunked


class DataFrameChunks:
    def __init__(self, generator):
        self.generator = generator
        self.n = 0
        
    def __next__(self) -> pd.DataFrame:
        chunk = next(self.generator)
        new_n = self.n + len(chunk)-1
        index = range(self.n, new_n+1)
        self.n = new_n + 1
        return pd.DataFrame(chunk, index=index)
    
    def __iter__(self):
        return self


def top_scores(row, row_date, dates, n=10):
    df = pd.DataFrame({
        'x_index': row.name,
        'score': row,
        'y_date':  dates
    })
    df = df[df['y_date'] < row_date]
    df.index.name = 'y_index'
    return df.sort_values('score').head(n).drop('y_date', axis=1).reset_index()  # type: ignore


def distance_generator(df, time_col, n_jobs=-1):
    return pairwise_distances_chunked(df.drop(time_col, axis=1), n_jobs=n_jobs)


def distances(df: pd.DataFrame, time_col: str, top_n=10, n_jobs=-1) -> pd.DataFrame:
    out = pd.DataFrame({'y_index': [], 'x_index': [], 'score': []})
    for chunk in distances_chunked(df, time_col, top_n, n_jobs):
        out = pd.concat([out, chunk])
    return out.reset_index(drop=True)


def distances_chunked(df: pd.DataFrame, time_col: str, top_n=10, n_jobs=-1):
    generator = distance_generator(df, time_col, n_jobs=n_jobs)
    df_generator = DataFrameChunks(generator)
    for _, chunk in enumerate(df_generator):
        for row_i, row in chunk.iterrows():
            scores = top_scores(row, df.iloc[row_i][time_col], df[time_col], n=top_n)  # type: ignore
            yield scores