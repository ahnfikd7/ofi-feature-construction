import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def load_data(path: str) -> pd.DataFrame:
    """
    Load the LOB CSV, parse timestamps, and sort by symbol + time.
    """
    df = pd.read_csv(path, parse_dates=['ts_event'])
    df.sort_values(['symbol', 'ts_event'], inplace=True)
    return df

def compute_level_ofi(df: pd.DataFrame, level: int) -> pd.DataFrame:
    """
    Compute OFI at a single depth level.
    level: 0–9 (formatted as two digits in column names)
    Returns a DataFrame with columns ['symbol','ts_event','ofi_L{level}'].
    """
    lvl = f"{level:02d}"
    bid_sz = f'bid_sz_{lvl}'
    ask_sz = f'ask_sz_{lvl}'

    # compute size changes per symbol
    delta_bid = df.groupby('symbol')[bid_sz].diff().fillna(0)
    delta_ask = df.groupby('symbol')[ask_sz].diff().fillna(0)

    return pd.DataFrame({
        'symbol': df['symbol'],
        'ts_event': df['ts_event'],
        f'ofi_L{level}': delta_bid - delta_ask
    })

def compute_best_ofi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket the level-0 OFI into 1-minute intervals.
    """
    lvl0 = compute_level_ofi(df, 0)
    lvl0.set_index('ts_event', inplace=True)
    best = (
        lvl0
        .groupby('symbol')
        .resample('1T')[f'ofi_L0']
        .sum()
        .reset_index()
        .rename(columns={'ofi_L0': 'best_ofi'})
    )
    return best

def compute_multilevel_ofi(df: pd.DataFrame, max_level: int = 9) -> pd.DataFrame:
    """
    Compute levels 0–max_level, then bucket each into 1-minute sums.
    Returns one DataFrame with columns ['symbol','ts_event','ofi_L0',…,'ofi_L9'].
    """
    ofi_list = []
    for lvl in range(max_level + 1):
        tmp = compute_level_ofi(df, lvl)
        tmp.set_index('ts_event', inplace=True)
        b = (
            tmp
            .groupby('symbol')
            .resample('1T')[f'ofi_L{lvl}']
            .sum()
            .rename(f'ofi_L{lvl}')
        )
        ofi_list.append(b)
    # merge all levels on index ['symbol','ts_event']
    merged = pd.concat(ofi_list, axis=1).reset_index()
    return merged

def compute_integrated_ofi(multilevel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run a 1-component PCA on the multi-level OFI vector to get Integrated OFI.
    """
    ofi_cols = [c for c in multilevel_df.columns if c.startswith('ofi_L')]
    X = multilevel_df[ofi_cols].fillna(0).values
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(X).flatten()
    out = multilevel_df[['symbol','ts_event']].copy()
    out['integrated_ofi'] = pc1
    return out

def compute_cross_asset_ofi(best_ofi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compress all symbols' best_ofi at each minute into one series via PCA.
    """
    # pivot: rows=timestamp, cols=symbol
    pivot = best_ofi_df.pivot(index='ts_event', columns='symbol', values='best_ofi').fillna(0)
    pca = PCA(n_components=1)
    # fit_transform gives shape (n_times, 1)
    comp = pca.fit_transform(pivot.values)[:,0]
    return pd.DataFrame({
        'ts_event': pivot.index,
        'cross_asset_ofi': comp
    })

def main():
    df = load_data('first_25000_rows.csv')

    best = compute_best_ofi(df)
    best.to_csv('best_ofi.csv', index=False)

    multi = compute_multilevel_ofi(df)
    multi.to_csv('multilevel_ofi.csv', index=False)

    integrated = compute_integrated_ofi(multi)
    integrated.to_csv('integrated_ofi.csv', index=False)

    cross = compute_cross_asset_ofi(best)
    cross.to_csv('cross_asset_ofi.csv', index=False)

if __name__ == '__main__':
    main()
