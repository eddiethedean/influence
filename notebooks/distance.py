from typing import Generator
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances_chunked


def get_scores(row, n=10) -> list[dict]:
    return [
        {'x_index': row.name,
         'y_index': i,
         'score': score}
            for i, score in row.sort_values().drop(row.name).head(n).items()]


def distances_chunked(numbers_df: pd.DataFrame, date_col: str, top_n=10, n_jobs=-1) -> Generator[pd.DataFrame, None, None]:
    generator = pairwise_distances_chunked(numbers_df.drop(date_col, axis=1), n_jobs=-1)
    index = 0
    scores = []
    for chunk in generator:
        count = len(chunk)
        indexes = range(index, count + index)
        index = count + index
        results = pd.DataFrame(chunk, index=indexes)
        for i, row in results.iterrows():
            scores.extend(get_scores(row))
        yield pd.DataFrame(scores)


def distances(numbers_df: pd.DataFrame, date_col: str, top_n=10, n_jobs=-1) -> pd.DataFrame:
    out = pd.DataFrame()
    for chunk in distances_chunked(numbers_df, date_col, top_n, n_jobs):
        pd.concat([out, chunk])
    return out