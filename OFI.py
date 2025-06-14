import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw limit order book CSV.
    Assumes columns include:
      - 'timestamp' (datetime or string)
      - 'bid_price_{i}', 'bid_size_{i}', 'ask_price_{i}', 'ask_size_{i}' for i=1..M
    """
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def compute_level_ofi(df: pd.DataFrame, level: int) -> pd.Series:
    """
    Compute per-event OFI at a given depth level.
    Implements Cont et al. logic:
      delta_bid = +size change if same price; +full new size if price improves; 0 if price worsens
      delta_ask = -size change if same price; -full new size if price improves for ask side; 0 if price worsens
    """
    bp = df[f'bid_price_{level}']
    bs = df[f'bid_size_{level}']
    ap = df[f'ask_price_{level}']
    asc = df[f'ask_size_{level}']

    # shift previous values
    bp_prev = bp.shift(1)
    bs_prev = bs.shift(1)
    ap_prev = ap.shift(1)
    asc_prev = asc.shift(1)

    # bid side changes
    delta_bid = np.where(
        bp > bp_prev,  # bid price improved
        bs,
        np.where(
            bp == bp_prev,  # price unchanged
            bs - bs_prev,
            0  # worsened
        )
    )

    # ask side changes
    delta_ask = np.where(
        ap < ap_prev,  # ask price improved (lower)
        -asc,
        np.where(
            ap == ap_prev,
            -(asc - asc_prev),
            0
        )
    )

    ofi_level = delta_bid + delta_ask
    # first row NaN -> zero
    ofi_level.iloc[0] = 0
    return pd.Series(ofi_level, index=df.index)


def bucket_ofi(ofi: pd.Series, timestamps: pd.Series, freq: str = '1T') -> pd.Series:
    """
    Aggregate per-event OFI into time buckets (e.g., 1-minute '1T').
    Returns a Series indexed by bucket start timestamp.
    """
    # assign each event to its time bucket
    buckets = timestamps.dt.floor(freq)
    return ofi.groupby(buckets).sum()


def compute_best_ofi(df: pd.DataFrame, freq: str = '1T') -> pd.Series:
    """
    Best-level OFI: aggregated OFI at level 1.
    """
    ofi1 = compute_level_ofi(df, level=1)
    return bucket_ofi(ofi1, df['timestamp'], freq)


def compute_multilevel_ofi(df: pd.DataFrame, max_level: int = 10, freq: str = '1T') -> pd.DataFrame:
    """
    Multi-level OFI: compute OFI at each level 1..max_level,
    and bucket into freq. Returns DataFrame with columns OFI_1..OFI_max.
    """
    ofi_dict = {}
    for lvl in range(1, max_level + 1):
        ofi_series = compute_level_ofi(df, level=lvl)
        ofi_bucketed = bucket_ofi(ofi_series, df['timestamp'], freq)
        ofi_dict[f'OFI_{lvl}'] = ofi_bucketed
    return pd.DataFrame(ofi_dict).fillna(0)


def compute_integrated_ofi(multilevel_df: pd.DataFrame) -> pd.Series:
    """
    Integrated OFI: project multi-level OFI onto 1st principal component.
    Returns a Series indexed as multilevel_df.
    """
    pca = PCA(n_components=1)
    component = pca.fit_transform(multilevel_df.values)
    # flatten and wrap as Series
    return pd.Series(component.flatten(), index=multilevel_df.index, name='Integrated_OFI')


def compute_cross_asset_ofi(all_ofi: dict) -> pd.DataFrame:
    """
    Cross-Asset OFI: for each symbol, aggregate other symbols' OFI.
    all_ofi: dict of symbol -> OFI Series (indexed by timestamp).
    Returns DataFrame: rows indexed by timestamp, columns CrossOFI_<symbol>.
    Here we compress each set of other-symbol OFIs into first PCA component.
    """
    cross_ofi = {}
    # full OFI DataFrame: symbols x timestamps
    ofi_df = pd.DataFrame(all_ofi)
    symbols = ofi_df.columns.tolist()
    for sym in symbols:
        others = ofi_df.drop(columns=[sym]).fillna(0)
        # PCA to 1st component
        pca = PCA(n_components=1)
        comp = pca.fit_transform(others.values)
        cross_ofi[f'CrossOFI_{sym}'] = comp.flatten()
    return pd.DataFrame(cross_ofi, index=ofi_df.index)


def main():
    # Example pipeline
    # 1. Load data
    df = load_data('first_25000_rows.csv')

    # 2. Best-Level OFI
    best_ofi = compute_best_ofi(df)
    best_ofi.to_csv('best_level_ofi.csv', header=True)

    # 3. Multi-Level OFI
    multilevel_ofi = compute_multilevel_ofi(df, max_level=10)
    multilevel_ofi.to_csv('multilevel_ofi.csv')

    # 4. Integrated OFI
    integrated = compute_integrated_ofi(multilevel_ofi)
    integrated.to_csv('integrated_ofi.csv', header=True)

    # 5. Cross-Asset OFI (example: same symbol only)
    all_ofi = {'SYM': best_ofi}  
    cross = compute_cross_asset_ofi(all_ofi)
    cross.to_csv('cross_asset_ofi.csv')

    print("OFI feature files saved:",
          "best_level_ofi.csv, multilevel_ofi.csv, integrated_ofi.csv, cross_asset_ofi.csv")

if __name__ == '__main__':
    main()
