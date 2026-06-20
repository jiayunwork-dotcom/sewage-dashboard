import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from typing import Dict, Tuple, Optional, List
from data_processing import INFLOW_INDICATORS, PROCESS_PARAMS, KEY_PREDICTORS, STANDARDS


BLOWER_COEFF = 0.0015
RETURN_COEFF = 0.8
CARBON_COST = 2.5

PARAM_BOUNDS = [
    (2000, 8000),
    (50, 150),
    (0, 200),
]

PARAM_NAMES_IN_OPT = ['曝气量', '污泥回流比', '碳源投加量']

DEFAULT_DAILY_FLOW = 50000


def compute_energy(aeration: float, return_ratio: float, carbon_dosage: float) -> float:
    return (BLOWER_COEFF * aeration +
            RETURN_COEFF * return_ratio +
            CARBON_COST * carbon_dosage)


def compute_daily_cost(energy_per_ton: float, daily_flow: float = DEFAULT_DAILY_FLOW) -> float:
    return energy_per_ton * daily_flow * 0.6


def predict_outflow_wrapper(predict_fn, current_state: np.ndarray,
                            process_params: np.ndarray, lookback: int = 24) -> np.ndarray:
    new_state = current_state.copy()
    param_indices = [PROCESS_PARAMS.index(name) for name in PARAM_NAMES_IN_OPT]
    for i, idx in enumerate(param_indices):
        new_state[-1, len(INFLOW_INDICATORS) + idx] = process_params[i]
    X_input = new_state.reshape(1, lookback, -1)
    return predict_fn(X_input)[0]


def objective(params: np.ndarray, predict_fn, current_state: np.ndarray,
              standards: Dict, penalty_weight: float = 1e6) -> float:
    energy = compute_energy(params[0], params[1], params[2])
    predictions = predict_outflow_wrapper(predict_fn, current_state, params)

    penalty = 0.0
    for i, name in enumerate(KEY_PREDICTORS):
        std = standards.get(name, np.inf)
        for h in range(predictions.shape[0]):
            if predictions[h, i] > std:
                penalty += (predictions[h, i] - std) ** 2

    return energy + penalty_weight * penalty


def constraint_check_fn(predict_fn, current_state: np.ndarray, params: np.ndarray,
                        standards: Dict) -> Tuple[bool, List[str]]:
    predictions = predict_outflow_wrapper(predict_fn, current_state, params)
    violations = []
    feasible = True
    for i, name in enumerate(KEY_PREDICTORS):
        std = standards.get(name, np.inf)
        for h in range(predictions.shape[0]):
            if predictions[h, i] > std:
                feasible = False
                violations.append(f'{name.replace("_out", "")}(t+{h+1}h)')
                break
    return feasible, violations


def optimize_process(predict_fn, current_state: np.ndarray,
                     current_params: Dict[str, float],
                     daily_flow: float = DEFAULT_DAILY_FLOW,
                     max_iter: int = 100) -> Dict:
    bounds = PARAM_BOUNDS

    result = differential_evolution(
        objective,
        bounds,
        args=(predict_fn, current_state, STANDARDS),
        maxiter=max_iter,
        popsize=15,
        tol=1e-6,
        seed=42,
        polish=True
    )

    opt_params = result.x
    opt_energy = compute_energy(opt_params[0], opt_params[1], opt_params[2])
    opt_predictions = predict_outflow_wrapper(predict_fn, current_state, opt_params)

    feasible, violations = constraint_check_fn(predict_fn, current_state, opt_params, STANDARDS)

    current_p = [current_params.get(name, 0) for name in PARAM_NAMES_IN_OPT]
    current_energy = compute_energy(current_p[0], current_p[1], current_p[2])
    current_predictions = predict_outflow_wrapper(predict_fn, current_state, np.array(current_p))

    result_dict = {
        'feasible': feasible,
        'violations': violations,
        'opt_params': {
            '曝气量(m³/h)': float(opt_params[0]),
            '污泥回流比(%)': float(opt_params[1]),
            '碳源投加量(L/h)': float(opt_params[2]),
        },
        'current_params': {
            '曝气量(m³/h)': float(current_p[0]),
            '污泥回流比(%)': float(current_p[1]),
            '碳源投加量(L/h)': float(current_p[2]),
        },
        'adjustments': {
            '曝气量': float(opt_params[0] - current_p[0]),
            '污泥回流比': float(opt_params[1] - current_p[1]),
            '碳源投加量': float(opt_params[2] - current_p[2]),
        },
        'opt_energy_per_ton': float(opt_energy),
        'current_energy_per_ton': float(current_energy),
        'energy_saving_per_ton': float(current_energy - opt_energy),
        'daily_cost_saving': float((current_energy - opt_energy) * daily_flow * 0.6),
        'opt_predictions': opt_predictions,
        'current_predictions': current_predictions,
        'tight_constraints': _find_tight_constraints(predict_fn, current_state, current_p, opt_predictions) if not feasible else None
    }

    return result_dict


def _find_tight_constraints(predict_fn, current_state: np.ndarray, current_p: List,
                            opt_predictions: np.ndarray) -> List[str]:
    tight = []
    for i, name in enumerate(KEY_PREDICTORS):
        std = STANDARDS.get(name, np.inf)
        max_val = np.max(opt_predictions[:, i])
        if max_val > std:
            tight.append(f'{name.replace("_out", "")}(标准{std}mg/L，预测最高{max_val:.2f}mg/L)')
    return tight


def generate_suggestion_text(opt_result: Dict) -> str:
    if not opt_result['feasible']:
        text = '⚠️ 警告：在当前工艺参数可调范围内无法找到满足出水达标的方案，建议启动应急措施。\n'
        if opt_result['tight_constraints']:
            text += '约束过紧的指标：\n'
            for c in opt_result['tight_constraints']:
                text += f'  • {c}\n'
        text += '\n建议措施：\n'
        text += '  1. 联系上游企业排查异常排放源\n'
        text += '  2. 增加碳源应急投加罐储量\n'
        text += '  3. 启动备用曝气系统\n'
        return text

    lines = ['✅ 已找到最优工艺参数组合，调整建议如下：\n']
    for name, delta in opt_result['adjustments'].items():
        cur = opt_result['current_params'].get(f'{name}(m³/h)') or \
              opt_result['current_params'].get(f'{name}(%)') or \
              opt_result['current_params'].get(f'{name}(L/h)')
        opt = opt_result['opt_params'].get(f'{name}(m³/h)') or \
              opt_result['opt_params'].get(f'{name}(%)') or \
              opt_result['opt_params'].get(f'{name}(L/h)')
        direction = '↑增加' if delta > 0 else ('↓减少' if delta < 0 else '→不变')
        unit = 'm³/h' if '曝气量' in name else ('%' if '回流比' in name else 'L/h')
        lines.append(f'  • {name}: {direction} {abs(delta):.1f}{unit}  (当前{cur:.1f} → 建议{opt:.1f})')

    lines.append(f'\n💡 能耗优化效果：')
    lines.append(f'  • 吨水综合能耗: {opt_result["current_energy_per_ton"]:.4f} → {opt_result["opt_energy_per_ton"]:.4f} kWh/m³')
    lines.append(f'  • 预计日运行成本节约: ¥{opt_result["daily_cost_saving"]:.2f}')
    return '\n'.join(lines)
