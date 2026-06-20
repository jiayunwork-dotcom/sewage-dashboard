import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

INFLOW_INDICATORS = ['COD_in', 'BOD5_in', 'SS_in', 'TN_in', 'TP_in', 'NH3N_in', 'pH_in']
PROCESS_PARAMS = ['曝气量', '污泥回流比', '剩余污泥排放量', 'HRT', 'DO设定值', '碳源投加量']
OUTFLOW_INDICATORS = ['COD_out', 'BOD5_out', 'SS_out', 'TN_out', 'TP_out', 'NH3N_out', 'pH_out']
KEY_PREDICTORS = ['COD_out', 'TN_out', 'TP_out']

STANDARDS = {
    'COD_out': 50.0,
    'TN_out': 15.0,
    'TP_out': 0.5
}

DISPLAY_NAMES = {
    'COD_in': '进水COD(mg/L)', 'BOD5_in': '进水BOD5(mg/L)', 'SS_in': '进水SS(mg/L)',
    'TN_in': '进水TN(mg/L)', 'TP_in': '进水TP(mg/L)', 'NH3N_in': '进水NH3-N(mg/L)', 'pH_in': '进水pH',
    'COD_out': '出水COD(mg/L)', 'BOD5_out': '出水BOD5(mg/L)', 'SS_out': '出水SS(mg/L)',
    'TN_out': '出水TN(mg/L)', 'TP_out': '出水TP(mg/L)', 'NH3N_out': '出水NH3-N(mg/L)', 'pH_out': '出水pH',
    '曝气量': '曝气量(m³/h)', '污泥回流比': '污泥回流比(%)', '剩余污泥排放量': '剩余污泥排放量(m³/d)',
    'HRT': 'HRT(h)', 'DO设定值': 'DO设定值(mg/L)', '碳源投加量': '碳源投加量(L/h)',
    'SRT': '污泥龄SRT(d)', 'MLSS': 'MLSS(mg/L)', 'SV30': 'SV30(%)',
    '处理水量': '处理水量(m³/d)', '水温': '水温(℃)'
}

SEASON_MAP = {1: '冬季', 2: '冬季', 3: '春季', 4: '春季', 5: '春季',
              6: '夏季', 7: '夏季', 8: '夏季', 9: '秋季', 10: '秋季',
              11: '秋季', 12: '冬季'}


def detect_time_column(df: pd.DataFrame) -> str:
    time_keywords = ['time', 'date', 'datetime', '时间', '日期', '时间戳']
    for col in df.columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in time_keywords):
            return col
    if pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
        return df.columns[0]
    try:
        pd.to_datetime(df.iloc[:, 0])
        return df.columns[0]
    except (ValueError, TypeError):
        raise ValueError('未找到时间列，请确保CSV包含时间/日期列')


def load_and_clean_csv(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    time_col = detect_time_column(df)
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)
    df = df.set_index(time_col)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        df[col] = df[col].interpolate(method='linear', limit_direction='both')
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())

    outlier_mask = pd.DataFrame(False, index=df.index, columns=numeric_cols)
    for col in numeric_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        if std_val > 0:
            outlier_mask[col] = (df[col] - mean_val).abs() > 3 * std_val

    df = df.reset_index()
    outlier_mask = outlier_mask.reset_index()
    return df, outlier_mask


def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    stats = df[numeric_cols].agg(['mean', 'median', 'std', 'min', 'max']).T
    stats.columns = ['均值', '中位数', '标准差', '最小值', '最大值']
    stats = stats.round(4)
    stats['指标名称'] = [DISPLAY_NAMES.get(c, c) for c in stats.index]
    return stats.reset_index().rename(columns={'index': '指标'})[['指标', '指标名称', '均值', '中位数', '标准差', '最小值', '最大值']]


def create_sliding_windows(df: pd.DataFrame, lookback: int = 24, horizon: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    feature_cols = [c for c in INFLOW_INDICATORS + PROCESS_PARAMS if c in df.columns]
    target_cols = [c for c in KEY_PREDICTORS if c in df.columns]

    data = df[feature_cols + target_cols].values.astype(np.float32)
    n_samples = len(data) - lookback - horizon + 1

    X = np.zeros((n_samples, lookback, len(feature_cols)), dtype=np.float32)
    y = np.zeros((n_samples, horizon, len(target_cols)), dtype=np.float32)

    for i in range(n_samples):
        X[i] = data[i:i + lookback, :len(feature_cols)]
        y[i] = data[i + lookback:i + lookback + horizon, len(feature_cols):]

    return X, y


def train_test_split_time(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2) -> Tuple:
    split_idx = int(len(X) * (1 - test_ratio))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def normalize_data(df: pd.DataFrame, train_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, Dict]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if train_df is None:
        means = df[numeric_cols].mean()
        stds = df[numeric_cols].std()
        stds = stds.replace(0, 1)
    else:
        means = train_df[numeric_cols].mean()
        stds = train_df[numeric_cols].std()
        stds = stds.replace(0, 1)

    result = df.copy()
    result[numeric_cols] = (result[numeric_cols] - means) / stds
    return result, {'means': means, 'stds': stds}


def denormalize(values: np.ndarray, means: pd.Series, stds: pd.Series, cols: List[str]) -> np.ndarray:
    means_arr = means[cols].values.astype(np.float32)
    stds_arr = stds[cols].values.astype(np.float32)
    return values * stds_arr + means_arr
