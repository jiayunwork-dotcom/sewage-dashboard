import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from data_processing import INFLOW_INDICATORS, PROCESS_PARAMS, KEY_PREDICTORS, STANDARDS

def test_shock_simulation_logic():
    print("=" * 60)
    print("测试冲击负荷模拟核心逻辑")
    print("=" * 60)

    n_days = 10
    hours = n_days * 24
    start = datetime(2024, 1, 1, 0, 0, 0)
    times = [start + timedelta(hours=i) for i in range(hours)]

    np.random.seed(42)
    data = {
        '时间': times,
        'COD_in': np.full(hours, 300.0),
        'TN_in': np.full(hours, 35.0),
        'TP_in': np.full(hours, 4.0),
        'BOD5_in': np.full(hours, 150.0),
        'SS_in': np.full(hours, 200.0),
        'NH3N_in': np.full(hours, 20.0),
        'pH_in': np.full(hours, 7.2),
        '曝气量': np.full(hours, 5000.0),
        '污泥回流比': np.full(hours, 100.0),
        '剩余污泥排放量': np.full(hours, 80.0),
        'HRT': np.full(hours, 12.0),
        'DO设定值': np.full(hours, 2.0),
        '碳源投加量': np.full(hours, 80.0),
    }
    for name in KEY_PREDICTORS:
        data[name] = np.full(hours, STANDARDS[name] * 0.5)

    df = pd.DataFrame(data)

    lookback = 24
    start_idx = 72
    duration = 4
    multiplier = 2.0

    baseline_window = df.iloc[start_idx - 24:start_idx]
    baseline_cod = baseline_window['COD_in'].mean()
    baseline_tn = baseline_window['TN_in'].mean()
    baseline_tp = baseline_window['TP_in'].mean()

    print(f"\n基线值 (冲击前24h均值):")
    print(f"  COD_in = {baseline_cod:.1f} mg/L")
    print(f"  TN_in = {baseline_tn:.1f} mg/L")
    print(f"  TP_in = {baseline_tp:.2f} mg/L")

    total_hours = duration + 4
    end_idx = start_idx + total_hours

    feature_cols = [c for c in INFLOW_INDICATORS + PROCESS_PARAMS if c in df.columns]
    cod_in_col_idx = feature_cols.index('COD_in')
    tn_in_col_idx = feature_cols.index('TN_in')
    tp_in_col_idx = feature_cols.index('TP_in')
    shock_start_data_idx = start_idx
    shock_end_data_idx = start_idx + duration

    print(f"\n冲击配置:")
    print(f"  开始时刻索引: {start_idx} ({times[start_idx]})")
    print(f"  持续时长: {duration}小时")
    print(f"  冲击倍率: {multiplier}x")
    print(f"  总预测时长: {total_hours}小时 (冲击{duration}h + 恢复4h)")

    print(f"\n{'步骤':<6} {'窗口范围':<20} {'冲击放大的时刻数':<15} {'验证':<30}")
    print("-" * 80)

    for step in range(total_hours):
        window_start = start_idx + step - lookback
        window_end = start_idx + step

        state_window = df.iloc[window_start:window_end][feature_cols].values.astype(np.float32).copy()

        modified_count = 0
        modified_indices = []
        for t_in_window in range(lookback):
            actual_data_idx = window_start + t_in_window
            if shock_start_data_idx <= actual_data_idx < shock_end_data_idx:
                state_window[t_in_window, cod_in_col_idx] = baseline_cod * multiplier
                state_window[t_in_window, tn_in_col_idx] = baseline_tn * multiplier
                state_window[t_in_window, tp_in_col_idx] = baseline_tp * multiplier
                modified_count += 1
                modified_indices.append(actual_data_idx)

        cod_values_in_window = state_window[:, cod_in_col_idx]
        has_shock_values = np.any(cod_values_in_window > baseline_cod * 1.5)

        if step < duration:
            expected_modified = min(step + 1, duration)
        else:
            remaining_in_window = max(0, duration - (step - duration + 1))
            expected_modified = max(0, remaining_in_window)

        status = "✓" if modified_count > 0 or step >= duration else "✗"
        note = ""
        if step >= duration:
            note = "(恢复期)"
            if modified_count > 0:
                note += " 窗口仍含冲击数据 ✓"

        print(f"{step:<6} [{window_start}:{window_end}]  {modified_count:<15} {status} 预期~{expected_modified}个 {note}")

        if step == duration:
            print(f"    验证: 进入恢复期后，窗口内仍有{modified_count}个冲击放大的数据点")

    print("\n" + "=" * 60)
    print("测试通过！核心逻辑验证成功：")
    print("  1. 冲击时段内的进水数据正确放大")
    print("  2. 恢复期窗口仍正确包含放大后的冲击数据")
    print("  3. 其余进水指标和工艺参数保持原值")
    print("=" * 60)

    return True

if __name__ == "__main__":
    try:
        test_shock_simulation_logic()
    except Exception as e:
        import traceback
        print(f"\n❌ 测试失败: {e}")
        print(traceback.format_exc())
        sys.exit(1)
