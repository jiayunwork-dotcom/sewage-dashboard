import os
import io
import base64
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import dash_bootstrap_components as dbc

from data_processing import (
    load_and_clean_csv, compute_statistics, create_sliding_windows,
    train_test_split_time, INFLOW_INDICATORS, PROCESS_PARAMS, OUTFLOW_INDICATORS,
    KEY_PREDICTORS, STANDARDS, DISPLAY_NAMES, SEASON_MAP
)
from models import (
    train_lstm, train_xgboost, train_fusion, predict_lstm, predict_xgboost,
    predict_fusion, compute_metrics, estimate_violation_probability
)
from optimization import (
    optimize_process, generate_suggestion_text, compute_energy, compute_daily_cost,
    BLOWER_COEFF, RETURN_COEFF, CARBON_COST, DEFAULT_DAILY_FLOW
)
from report_export import generate_daily_report


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)
server = app.server
app.title = '城镇污水处理厂出水水质预测与工艺调控分析平台'


GLOBAL_STATE = {
    'df': None, 'outlier_mask': None, 'time_col': None,
    'X_train': None, 'X_test': None, 'y_train': None, 'y_test': None,
    'lstm_model': None, 'xgb_model': None,
    'lstm_result': None, 'xgb_result': None, 'fusion_result': None,
    'fusion_lstm_weight': 0.5,
    'scaler': None, 'feature_cols': None, 'target_cols': None,
    'last_prediction': None, 'last_prediction_times': None
}

COLOR_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def generate_sample_data():
    np.random.seed(42)
    n_days = 90
    hours = n_days * 24
    start = datetime(2024, 1, 1, 0, 0, 0)
    times = [start + timedelta(hours=i) for i in range(hours)]

    t = np.arange(hours)
    diurnal = np.sin(2 * np.pi * t / 24) * 0.15 + 1
    weekly = np.sin(2 * np.pi * t / (24*7)) * 0.1 + 1
    seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * t / hours)

    def with_noise(base, scale, min_v=None, max_v=None):
        v = base * diurnal * weekly * seasonal_factor * (1 + np.random.randn(hours) * scale)
        if min_v is not None:
            v = np.maximum(v, min_v)
        if max_v is not None:
            v = np.minimum(v, max_v)
        return v

    cod_in = with_noise(350, 0.12, 150, 600)
    bod5_in = cod_in * (0.4 + np.random.randn(hours) * 0.03)
    ss_in = with_noise(250, 0.15, 100, 450)
    tn_in = with_noise(40, 0.1, 20, 70)
    tp_in = with_noise(5, 0.12, 2, 10)
    nh3n_in = tn_in * (0.55 + np.random.randn(hours) * 0.05)
    ph_in = 7.2 + np.random.randn(hours) * 0.2
    ph_in = np.clip(ph_in, 6.5, 8.5)

    removal_cod = 0.85 + np.random.randn(hours) * 0.03
    removal_bod5 = 0.92 + np.random.randn(hours) * 0.02
    removal_ss = 0.94 + np.random.randn(hours) * 0.02
    removal_tn = 0.65 + np.random.randn(hours) * 0.05
    removal_tp = 0.90 + np.random.randn(hours) * 0.03
    removal_nh3n = 0.88 + np.random.randn(hours) * 0.04

    winter = (np.array([dt.month for dt in times])[:, None] == np.array([12, 1, 2])).any(axis=1)
    removal_nh3n[winter] *= (0.85 + np.random.randn(winter.sum()) * 0.05)
    removal_tn[winter] *= 0.9

    cod_out = cod_in * (1 - removal_cod)
    bod5_out = bod5_in * (1 - removal_bod5)
    ss_out = ss_in * (1 - removal_ss)
    tn_out = tn_in * (1 - removal_tn)
    tp_out = tp_in * (1 - removal_tp)
    nh3n_out = nh3n_in * (1 - removal_nh3n)
    ph_out = ph_in + np.random.randn(hours) * 0.15

    aeration = 5000 + (cod_in - 350) * 5 + np.random.randn(hours) * 300
    aeration = np.clip(aeration, 2500, 8000)
    return_ratio = 100 + (tn_in - 40) * 0.8 + np.random.randn(hours) * 10
    return_ratio = np.clip(return_ratio, 55, 145)
    waste_sludge = 80 + (ss_in - 250) * 0.1 + np.random.randn(hours) * 15
    waste_sludge = np.clip(waste_sludge, 30, 200)
    hrt = np.full(hours, 12.0) + np.random.randn(hours) * 0.5
    do_set = 2.0 + (cod_in - 350) * 0.003 + np.random.randn(hours) * 0.3
    do_set = np.clip(do_set, 0.5, 4.0)
    carbon = 80 + (tn_in - 40) * 2.5 + np.random.randn(hours) * 20
    carbon = np.clip(carbon, 0, 195)

    mlss = 3000 + (ss_in - 250) * 2 + np.random.randn(hours) * 300
    mlss = np.clip(mlss, 1500, 5000)
    sv30 = mlss * (0.009 + np.random.randn(hours) * 0.0015)
    sv30 = np.clip(sv30, 15, 90)

    temp = 15 + 10 * np.sin(2 * np.pi * t / hours - np.pi/2) + np.random.randn(hours) * 2
    flow = (DEFAULT_DAILY_FLOW/24) * diurnal * (1 + np.random.randn(hours) * 0.05)

    shock_idx = np.random.choice(np.arange(hours), size=15, replace=False)
    for idx in shock_idx:
        s = slice(max(0, idx-2), min(hours, idx+3))
        cod_in[s] *= 1.6
        tn_in[s] *= 1.5
        nh3n_in[s] *= 1.55

    df = pd.DataFrame({
        '时间': times,
        'COD_in': np.round(cod_in, 2),
        'BOD5_in': np.round(bod5_in, 2),
        'SS_in': np.round(ss_in, 2),
        'TN_in': np.round(tn_in, 2),
        'TP_in': np.round(tp_in, 3),
        'NH3N_in': np.round(nh3n_in, 3),
        'pH_in': np.round(ph_in, 2),
        '曝气量': np.round(aeration, 1),
        '污泥回流比': np.round(return_ratio, 1),
        '剩余污泥排放量': np.round(waste_sludge, 1),
        'HRT': np.round(hrt, 2),
        'DO设定值': np.round(do_set, 2),
        '碳源投加量': np.round(carbon, 1),
        'COD_out': np.round(cod_out, 2),
        'BOD5_out': np.round(bod5_out, 2),
        'SS_out': np.round(ss_out, 2),
        'TN_out': np.round(tn_out, 2),
        'TP_out': np.round(tp_out, 3),
        'NH3N_out': np.round(nh3n_out, 3),
        'pH_out': np.round(ph_out, 2),
        'MLSS': np.round(mlss, 0),
        'SV30': np.round(sv30, 1),
        '水温': np.round(temp, 1),
        '处理水量': np.round(flow, 1),
    })
    return df


def header():
    return dbc.Navbar(
        dbc.Container([
            html.H3('🏭 城镇污水处理厂出水水质预测与工艺调控分析平台',
                    className='text-white mb-0'),
        ], fluid=True),
        color='#1a5276', dark=True, className='mb-4'
    )




OVERVIEW_LAYOUT = html.Div([
    dbc.Card([
        dbc.CardHeader('数据上传', className='bg-info text-white'),
        dbc.CardBody([
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    '拖拽CSV文件到此处 或 ', html.A('点击选择文件', className='text-primary')
                ]),
                style={'width': '100%', 'height': '60px', 'lineHeight': '60px',
                       'borderWidth': '1px', 'borderStyle': 'dashed',
                       'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px 0',
                       'backgroundColor': '#f8f9fa'},
                multiple=False
            ),
            html.Div(id='upload-status', className='text-muted'),
            html.Div([
                dbc.Button('使用示例数据（生成模拟数据）', id='use-sample-btn',
                           color='secondary', size='sm', className='mt-2'),
            ]),
        ])
    ], className='mb-4'),

    dbc.Card([
        dbc.CardHeader('统计摘要', className='bg-light'),
        dbc.CardBody([
            dash_table.DataTable(
                id='stats-table',
                columns=[], data=[],
                style_table={'overflowX': 'auto'},
                style_header={'backgroundColor': '#d6eaf8', 'fontWeight': 'bold'},
                style_cell={'textAlign': 'center', 'padding': '8px', 'fontSize': '12px'},
                style_data_conditional=[
                    {'if': {'column_id': '指标名称'}, 'textAlign': 'left'}
                ],
                page_size=25
            )
        ])
    ], className='mb-4'),

    dbc.Card([
        dbc.CardHeader('时间序列趋势图', className='bg-light'),
        dbc.CardBody([
            dcc.Dropdown(
                id='ts-metrics',
                options=[], value=[], multi=True,
                placeholder='选择要显示的指标（可多选叠加）',
                className='mb-3'
            ),
            dcc.Graph(id='ts-chart', style={'height': '420px'})
        ])
    ], className='mb-4'),
])


PREDICTION_LAYOUT = html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('模型参数设置', className='bg-primary text-white'),
                dbc.CardBody([
                    dbc.Label('选择模型'),
                    dcc.Dropdown(
                        id='model-select',
                        options=[
                            {'label': 'LSTM 时序网络', 'value': 'lstm'},
                            {'label': 'XGBoost 梯度提升树', 'value': 'xgb'},
                            {'label': '加权融合 (LSTM+XGBoost)', 'value': 'fusion'},
                        ], value='fusion', clearable=False, className='mb-3'
                    ),
                    html.Div(id='lstm-params-div', children=[
                        dbc.Label('LSTM 隐藏层单元数'),
                        dcc.Dropdown(
                            id='lstm-hidden',
                            options=[{'label': str(v), 'value': v} for v in [32, 64, 128]],
                            value=64, clearable=False, className='mb-3'
                        )
                    ]),
                    html.Div(id='xgb-params-div', children=[
                        dbc.Label('XGBoost 树数量'),
                        dcc.Dropdown(
                            id='xgb-estimators',
                            options=[{'label': str(v), 'value': v} for v in [50, 100, 200]],
                            value=100, clearable=False, className='mb-3'
                        )
                    ]),
                    html.Div(id='fusion-params-div', children=[
                        dbc.Label('融合权重 - LSTM占比'),
                        dcc.Slider(min=0, max=1, step=0.1, value=0.5,
                                   id='fusion-weight',
                                   marks={i/10: f'{i*10}%' for i in range(0, 11)}),
                        html.Br()
                    ]),
                    dbc.Button('🚀 训练模型', id='train-btn', color='success',
                               className='w-100'),
                    dcc.Loading(id='train-loading', type='circle',
                                children=html.Div(id='train-status', className='mt-3'))
                ])
            ])
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('测试集预测对比', className='bg-light'),
                dbc.CardBody([
                    dcc.Dropdown(
                        id='pred-target-select',
                        options=[{'label': DISPLAY_NAMES.get(c, c), 'value': c} for c in KEY_PREDICTORS],
                        value=KEY_PREDICTORS[0], clearable=False, className='mb-3'
                    ),
                    dcc.Graph(id='pred-chart', style={'height': '380px'})
                ])
            ], className='mb-3'),
            dbc.Card([
                dbc.CardHeader('模型误差指标', className='bg-light'),
                dbc.CardBody([
                    dash_table.DataTable(
                        id='metrics-table',
                        columns=[{'name': c, 'id': c} for c in ['指标', 'RMSE', 'MAE', 'MAPE(%)']],
                        data=[],
                        style_table={'overflowX': 'auto'},
                        style_header={'backgroundColor': '#d5f5e3', 'fontWeight': 'bold'},
                        style_cell={'textAlign': 'center', 'padding': '8px', 'fontSize': '12px'}
                    )
                ])
            ])
        ], width=8),
    ], className='mb-4')
])


WARNING_LAYOUT = html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('预警设置', className='bg-warning text-dark'),
                dbc.CardBody([
                    dbc.Label('选择预警起始时刻'),
                    dcc.Dropdown(id='warning-start-time', options=[],
                                 placeholder='默认使用数据最后时刻', className='mb-3'),
                    dbc.Label('按指标筛选'),
                    dcc.Dropdown(
                        id='warning-filter-indicator',
                        options=[{'label': '全部指标', 'value': 'all'}] +
                                [{'label': DISPLAY_NAMES.get(c, c), 'value': c} for c in KEY_PREDICTORS],
                        value='all', clearable=False, className='mb-2'
                    ),
                    dbc.Label('按预警等级筛选'),
                    dcc.Dropdown(
                        id='warning-filter-level',
                        options=[
                            {'label': '全部', 'value': 'all'},
                            {'label': '🔴 红色预警', 'value': 'red'},
                            {'label': '🟡 黄色预警', 'value': 'yellow'},
                            {'label': '✅ 正常', 'value': 'normal'},
                        ], value='all', clearable=False, className='mb-2'
                    ),
                    html.Hr(),
                    dbc.Button('🔍 执行预警分析', id='run-warning-btn', color='warning',
                               className='w-100'),
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('未来4小时预警时间轴', className='bg-light'),
                dbc.CardBody([
                    dcc.Loading(
                        id='warning-loading', type='circle',
                        children=html.Div(id='warning-cards', className='mt-2',
                                          style={'maxHeight': '500px', 'overflowY': 'auto'})
                    )
                ])
            ])
        ], width=9),
    ], className='mb-4')
])


OPTIMIZATION_LAYOUT = html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('当前工艺参数', className='bg-success text-white'),
                dbc.CardBody([
                    html.Div([
                        dbc.Label(f'{DISPLAY_NAMES["曝气量"]} (2000~8000)'),
                        dbc.Input(id='opt-p-0', type='number', value=5000, min=2000, max=8000, step=50)
                    ], className='mb-3'),
                    html.Div([
                        dbc.Label(f'{DISPLAY_NAMES["污泥回流比"]} (50~150)'),
                        dbc.Input(id='opt-p-1', type='number', value=100, min=50, max=150, step=1)
                    ], className='mb-3'),
                    html.Div([
                        dbc.Label(f'{DISPLAY_NAMES["碳源投加量"]} (0~200)'),
                        dbc.Input(id='opt-p-2', type='number', value=80, min=0, max=200, step=5)
                    ], className='mb-3'),
                ])
            ], className='mb-3'),
            dbc.Card([
                dbc.CardHeader('优化参数', className='bg-light'),
                dbc.CardBody([
                    dbc.Label('日处理水量 (m³/d)'),
                    dbc.Input(id='opt-daily-flow', type='number',
                              value=DEFAULT_DAILY_FLOW, className='mb-3'),
                    html.Hr(),
                    dbc.Button('🔬 运行工艺优化', id='run-opt-btn', color='primary',
                               className='w-100'),
                ])
            ])
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('优化结果与调整建议', className='bg-light'),
                dbc.CardBody([
                    dcc.Loading(id='opt-loading', type='circle',
                                children=html.Div(id='opt-result-text',
                                                  style={'whiteSpace': 'pre-line',
                                                         'fontSize': '13px'})),
                    html.Hr(),
                    html.Div(id='opt-details')
                ])
            ])
        ], width=8),
    ], className='mb-4')
])


ENERGY_LAYOUT = html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('三项主要能耗日趋势', className='bg-light'),
                dbc.CardBody([
                    dcc.Graph(id='energy-trend-chart', style={'height': '350px'})
                ])
            ], className='mb-3'),
            dbc.Card([
                dbc.CardHeader('吨水综合能耗月度趋势', className='bg-light'),
                dbc.CardBody([
                    dcc.Graph(id='energy-monthly-chart', style={'height': '300px'})
                ])
            ])
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('能耗构成占比', className='bg-light'),
                dbc.CardBody([
                    dcc.Graph(id='energy-pie-chart', style={'height': '300px'})
                ])
            ], className='mb-3'),
            dbc.Card([
                dbc.CardHeader('能耗异常日归因分析', className='bg-light'),
                dbc.CardBody([
                    html.Div(id='energy-anomaly-text',
                             style={'fontSize': '12px', 'maxHeight': '300px', 'overflowY': 'auto'})
                ])
            ])
        ], width=4),
    ], className='mb-4')
])


SLUDGE_LAYOUT = html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('污泥龄 SRT 每日趋势', className='bg-light'),
                dbc.CardBody([
                    dcc.Graph(id='srt-chart', style={'height': '420px'})
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('MLSS 与 SV30 关联（膨胀预警：SV30>70%）', className='bg-light'),
                dbc.CardBody([
                    dcc.Graph(id='mlss-sv30-chart', style={'height': '420px'})
                ])
            ])
        ], width=6),
    ], className='mb-4')
])


SEASONAL_LAYOUT = html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('四季指标均值对比', className='bg-light'),
                dbc.CardBody([
                    dcc.Dropdown(
                        id='season-metric-select',
                        options=[{'label': DISPLAY_NAMES.get(c, c), 'value': c}
                                 for c in OUTFLOW_INDICATORS] +
                                [{'label': '吨水能耗', 'value': '吨水能耗'},
                                 {'label': '平均碳源药耗', 'value': '碳源药耗'}],
                        value=OUTFLOW_INDICATORS[0], clearable=False, className='mb-3'
                    ),
                    dcc.Graph(id='season-bar-chart', style={'height': '380px'})
                ])
            ])
        ], width=7),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('冬季低温期硝化效率分析', className='bg-light'),
                dbc.CardBody([
                    dcc.Graph(id='winter-nitrification-chart', style={'height': '380px'})
                ])
            ])
        ], width=5),
    ], className='mb-4'),
    dbc.Card([
        dbc.CardHeader('四季综合对比表', className='bg-light'),
        dbc.CardBody([
            dash_table.DataTable(
                id='season-table',
                columns=[], data=[],
                style_table={'overflowX': 'auto'},
                style_header={'backgroundColor': '#fad7a0', 'fontWeight': 'bold'},
                style_cell={'textAlign': 'center', 'padding': '8px', 'fontSize': '12px'}
            )
        ])
    ], className='mb-4')
])


REPORT_LAYOUT = html.Div([
    dbc.Card([
        dbc.CardHeader('PDF 运行日报生成', className='bg-info text-white'),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label('选择报告日期'),
                    dcc.Dropdown(id='report-date-select', options=[],
                                 placeholder='选择日期', className='mb-3'),
                    dbc.Button('📥 生成并下载PDF日报', id='generate-report-btn',
                               color='primary'),
                    dcc.Download(id='download-report'),
                ], width=6),
                dbc.Col([
                    html.H6('📋 报告内容包括：', className='text-muted mb-3'),
                    html.Ul([
                        html.Li('当日进出水指标摘要（均值/极值/达标率）'),
                        html.Li('预警事件列表（时间/等级/超标指标/超标概率）'),
                        html.Li('工艺参数调控建议与优化效果评估'),
                        html.Li('能耗统计与吨水成本分析'),
                    ], className='text-muted', style={'fontSize': '13px'})
                ], width=6),
            ])
        ])
    ], className='mb-5')
])


TABS_IDS = ['tab-overview', 'tab-prediction', 'tab-warning', 'tab-optimization',
            'tab-energy', 'tab-sludge', 'tab-seasonal', 'tab-report']

app.layout = html.Div([
    header(),
    dcc.Store(id='last-model-store', data=None),
    dbc.Container([
        dbc.Tabs([
            dbc.Tab(OVERVIEW_LAYOUT, label='📊 数据概览', tab_id='tab-overview'),
            dbc.Tab(PREDICTION_LAYOUT, label='🔮 出水预测', tab_id='tab-prediction'),
            dbc.Tab(WARNING_LAYOUT, label='⚠️ 达标预警', tab_id='tab-warning'),
            dbc.Tab(OPTIMIZATION_LAYOUT, label='⚙️ 工艺优化', tab_id='tab-optimization'),
            dbc.Tab(ENERGY_LAYOUT, label='⚡ 能耗分析', tab_id='tab-energy'),
            dbc.Tab(SLUDGE_LAYOUT, label='🧪 污泥管理', tab_id='tab-sludge'),
            dbc.Tab(SEASONAL_LAYOUT, label='🌡️ 季节对比', tab_id='tab-seasonal'),
            dbc.Tab(REPORT_LAYOUT, label='📄 报告导出', tab_id='tab-report'),
        ], id='main-tabs', active_tab='tab-overview', className='mb-4', persistence=True),
    ], fluid=True)
])


@app.callback(
    Output('upload-status', 'children'),
    Output('stats-table', 'columns'),
    Output('stats-table', 'data'),
    Output('ts-metrics', 'options'),
    Output('ts-metrics', 'value'),
    Output('warning-start-time', 'options'),
    Output('report-date-select', 'options'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified'),
    Input('use-sample-btn', 'n_clicks'),
    prevent_initial_call=True
)
def handle_upload(contents, filename, last_modified, sample_n):
    ctx = callback_context
    triggered = ctx.triggered[0]['prop_id'].split('.')[0]

    try:
        if triggered == 'use-sample-btn' or (triggered == 'upload-data' and contents is None):
            df = generate_sample_data()
            time_col = '时间'
            outlier_mask = pd.DataFrame(False, index=df.index, columns=df.select_dtypes(include=[np.number]).columns)
            status = f'✅ 已加载模拟示例数据，共 {len(df)} 条记录（90天小时级）'
        else:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            tmp_path = f'temp_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'
            with open(tmp_path, 'wb') as f:
                f.write(decoded)
            df, outlier_mask = load_and_clean_csv(tmp_path)
            time_col = df.columns[0]
            try:
                os.remove(tmp_path)
            except:
                pass
            status = f'✅ 文件 "{filename}" 上传成功，共 {len(df)} 条记录'

        GLOBAL_STATE['df'] = df
        GLOBAL_STATE['outlier_mask'] = outlier_mask
        GLOBAL_STATE['time_col'] = time_col

        stats_df = compute_statistics(df)
        stats_cols = [{'name': c, 'id': c} for c in stats_df.columns]
        stats_data = stats_df.to_dict('records')

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        ts_options = [{'label': DISPLAY_NAMES.get(c, c), 'value': c} for c in numeric_cols]
        ts_default = [KEY_PREDICTORS[0]] if KEY_PREDICTORS[0] in numeric_cols else ([numeric_cols[0]] if numeric_cols else [])

        times = pd.to_datetime(df[time_col]).tolist()
        warn_opts = [{'label': t.strftime('%Y-%m-%d %H:00'), 'value': i} for i, t in enumerate(times)]
        dates = sorted(set(t.strftime('%Y-%m-%d') for t in pd.to_datetime(df[time_col])))
        date_opts = [{'label': d, 'value': d} for d in dates]

        return status, stats_cols, stats_data, ts_options, ts_default, warn_opts, date_opts

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return f'❌ 错误: {str(e)}', [], [], [], [], [], []


@app.callback(
    Output('ts-chart', 'figure'),
    Input('ts-metrics', 'value'),
    prevent_initial_call=True
)
def update_ts_chart(selected_metrics):
    if GLOBAL_STATE['df'] is None or not selected_metrics:
        return go.Figure()

    df = GLOBAL_STATE['df']
    time_col = GLOBAL_STATE['time_col']
    fig = go.Figure()

    for i, metric in enumerate(selected_metrics):
        if metric not in df.columns:
            continue
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        y_vals = df[metric].values
        x_vals = pd.to_datetime(df[time_col]).values
        if GLOBAL_STATE['outlier_mask'] is not None and metric in GLOBAL_STATE['outlier_mask'].columns:
            mask = GLOBAL_STATE['outlier_mask'][metric].values
            normal_x, normal_y = x_vals[~mask], y_vals[~mask]
            outlier_x, outlier_y = x_vals[mask], y_vals[mask]
            fig.add_trace(go.Scatter(x=normal_x, y=normal_y, mode='lines',
                                     name=DISPLAY_NAMES.get(metric, metric),
                                     line=dict(color=color, width=1.5)))
            if len(outlier_x) > 0:
                fig.add_trace(go.Scatter(x=outlier_x, y=outlier_y, mode='markers',
                                         name=f'{DISPLAY_NAMES.get(metric, metric)} 异常(3σ)',
                                         marker=dict(color='orange', size=6, symbol='x')))
        else:
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines',
                                     name=DISPLAY_NAMES.get(metric, metric),
                                     line=dict(color=color, width=1.5)))

    fig.update_layout(
        template='plotly_white', hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=20, t=40, b=40)
    )
    return fig


@app.callback(
    Output('train-status', 'children'),
    Output('pred-chart', 'figure'),
    Output('metrics-table', 'data'),
    Output('lstm-params-div', 'style'),
    Output('xgb-params-div', 'style'),
    Output('fusion-params-div', 'style'),
    Output('last-model-store', 'data'),
    Input('train-btn', 'n_clicks'),
    State('model-select', 'value'),
    State('lstm-hidden', 'value'),
    State('xgb-estimators', 'value'),
    State('fusion-weight', 'value'),
    prevent_initial_call=True
)
def train_models(n_clicks, model_type, lstm_hidden, xgb_est, fusion_w):
    if GLOBAL_STATE['df'] is None:
        return '请先上传数据', go.Figure(), [], {}, {}, {}, {}

    try:
        df = GLOBAL_STATE['df'].copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        all_needed = INFLOW_INDICATORS + PROCESS_PARAMS + KEY_PREDICTORS
        missing = [c for c in all_needed if c not in numeric_cols]
        if missing:
            return f'缺少必要字段: {missing}', go.Figure(), [], {}, {}, {}, {}

        lookback, horizon = 24, 4
        X, y = create_sliding_windows(df, lookback=lookback, horizon=horizon)
        X_train, X_test, y_train, y_test = train_test_split_time(X, y, test_ratio=0.2)

        GLOBAL_STATE['X_train'], GLOBAL_STATE['X_test'] = X_train, X_test
        GLOBAL_STATE['y_train'], GLOBAL_STATE['y_test'] = y_train, y_test

        lstm_style = {'display': 'block' if model_type in ['lstm', 'fusion'] else 'none'}
        xgb_style = {'display': 'block' if model_type in ['xgb', 'fusion'] else 'none'}
        fusion_style = {'display': 'block' if model_type == 'fusion' else 'none'}

        train_status_parts = []

        if model_type in ['lstm', 'fusion']:
            lstm_model, lstm_result = train_lstm(
                X_train, y_train, X_test, y_test, hidden_dim=lstm_hidden,
                epochs=50, batch_size=32, verbose=False
            )
            GLOBAL_STATE['lstm_model'] = lstm_model
            GLOBAL_STATE['lstm_result'] = lstm_result
            train_status_parts.append(f'✅ LSTM 训练完成 (hidden={lstm_hidden})')

        if model_type in ['xgb', 'fusion']:
            xgb_model, xgb_result = train_xgboost(
                X_train, y_train, X_test, y_test, n_estimators=xgb_est
            )
            GLOBAL_STATE['xgb_model'] = xgb_model
            GLOBAL_STATE['xgb_result'] = xgb_result
            train_status_parts.append(f'✅ XGBoost 训练完成 (trees={xgb_est})')

        if model_type == 'fusion':
            GLOBAL_STATE['fusion_lstm_weight'] = fusion_w
            fusion_result = train_fusion(
                GLOBAL_STATE['lstm_model'], GLOBAL_STATE['xgb_model'],
                GLOBAL_STATE['lstm_result']['test_pred'],
                GLOBAL_STATE['xgb_result']['test_pred'],
                y_test, lstm_weight=fusion_w
            )
            GLOBAL_STATE['fusion_result'] = fusion_result
            train_status_parts.append(f'✅ 融合完成 (LSTM权重 {fusion_w:.0%})')

        if model_type == 'lstm':
            test_pred = GLOBAL_STATE['lstm_result']['test_pred']
            metrics = GLOBAL_STATE['lstm_result']['metrics']
        elif model_type == 'xgb':
            test_pred = GLOBAL_STATE['xgb_result']['test_pred']
            metrics = GLOBAL_STATE['xgb_result']['metrics']
        else:
            test_pred = GLOBAL_STATE['fusion_result']['test_pred']
            metrics = GLOBAL_STATE['fusion_result']['metrics']

        GLOBAL_STATE['last_model_type'] = model_type

        metrics_data = []
        for name in KEY_PREDICTORS:
            m = metrics.get(name, {})
            metrics_data.append({
                '指标': DISPLAY_NAMES.get(name, name),
                'RMSE': f"{m.get('RMSE', 0):.4f}",
                'MAE': f"{m.get('MAE', 0):.4f}",
                'MAPE(%)': f"{m.get('MAPE', 0):.2f}"
            })

        target_idx = KEY_PREDICTORS.index(KEY_PREDICTORS[0])
        time_col = GLOBAL_STATE['time_col']
        df_times = pd.to_datetime(df[time_col]).values
        start_test = len(X_train) + lookback

        fig = go.Figure()
        n_display = min(300, len(test_pred))
        display_start = max(0, len(test_pred) - n_display)
        x_idx = np.arange(display_start, len(test_pred))
        t_idx = [start_test + i for i in x_idx]

        true_avg = y_test[x_idx, :, target_idx].mean(axis=1)
        pred_avg = test_pred[x_idx, :, target_idx].mean(axis=1)
        x_times = [str(df_times[i])[:13] for i in t_idx if i < len(df_times)]
        min_len = min(len(x_times), len(true_avg), len(pred_avg))

        fig.add_trace(go.Scatter(
            x=x_times[:min_len], y=true_avg[:min_len], mode='lines+markers',
            name='真实值', line=dict(color='#1f77b4', width=2), marker=dict(size=4)
        ))
        fig.add_trace(go.Scatter(
            x=x_times[:min_len], y=pred_avg[:min_len], mode='lines+markers',
            name='预测值', line=dict(color='#d62728', width=2, dash='dash'), marker=dict(size=4)
        ))

        std_val = STANDARDS.get(KEY_PREDICTORS[target_idx])
        if std_val:
            fig.add_hline(y=std_val, line_dash='dot', line_color='green',
                          annotation_text=f'GB-1A 标准 {std_val}mg/L',
                          annotation_position='bottom right')

        fig.update_layout(
            title=f'{DISPLAY_NAMES.get(KEY_PREDICTORS[target_idx])} 测试集预测对比 (未来4h均值)',
            template='plotly_white', hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            xaxis_tickangle=45, margin=dict(l=40, r=20, t=60, b=60)
        )

        status_html = html.Div([
            html.P(s, style={'margin': '2px 0', 'color': '#27ae60', 'fontWeight': 'bold'})
            for s in train_status_parts
        ] + [html.P(f'训练集: {len(X_train)} | 测试集: {len(X_test)} | 窗口: 过去24h→未来4h',
                    className='text-muted mt-2 fs-7')])

        return status_html, fig, metrics_data, lstm_style, xgb_style, fusion_style, {'type': model_type}

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return f'❌ 训练出错: {str(e)}\n{tb}', go.Figure(), [], {}, {}, {}, {}


@app.callback(
    Output('pred-chart', 'figure', allow_duplicate=True),
    Input('pred-target-select', 'value'),
    State('last-model-store', 'data'),
    prevent_initial_call=True
)
def update_pred_chart_target(target, store):
    if not store or GLOBAL_STATE.get('last_model_type') is None:
        raise dash.exceptions.PreventUpdate
    try:
        model_type = GLOBAL_STATE['last_model_type']
        if model_type == 'lstm':
            test_pred = GLOBAL_STATE['lstm_result']['test_pred']
        elif model_type == 'xgb':
            test_pred = GLOBAL_STATE['xgb_result']['test_pred']
        else:
            test_pred = GLOBAL_STATE['fusion_result']['test_pred']
        y_test = GLOBAL_STATE['y_test']
        X_train = GLOBAL_STATE['X_train']

        target_idx = KEY_PREDICTORS.index(target)
        time_col = GLOBAL_STATE['time_col']
        df_times = pd.to_datetime(GLOBAL_STATE['df'][time_col]).values
        start_test = len(X_train) + 24

        n_display = min(300, len(test_pred))
        display_start = max(0, len(test_pred) - n_display)
        x_idx = np.arange(display_start, len(test_pred))
        t_idx = [start_test + i for i in x_idx]

        true_avg = y_test[x_idx, :, target_idx].mean(axis=1)
        pred_avg = test_pred[x_idx, :, target_idx].mean(axis=1)
        x_times = [str(df_times[i])[:13] for i in t_idx if i < len(df_times)]
        min_len = min(len(x_times), len(true_avg), len(pred_avg))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_times[:min_len], y=true_avg[:min_len], mode='lines+markers',
            name='真实值', line=dict(color='#1f77b4', width=2), marker=dict(size=4)
        ))
        fig.add_trace(go.Scatter(
            x=x_times[:min_len], y=pred_avg[:min_len], mode='lines+markers',
            name='预测值', line=dict(color='#d62728', width=2, dash='dash'), marker=dict(size=4)
        ))
        std_val = STANDARDS.get(target)
        if std_val:
            fig.add_hline(y=std_val, line_dash='dot', line_color='green',
                          annotation_text=f'GB-1A {std_val}mg/L')
        fig.update_layout(
            title=f'{DISPLAY_NAMES.get(target)} 测试集预测对比 (未来4h均值)',
            template='plotly_white', hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            xaxis_tickangle=45, margin=dict(l=40, r=20, t=60, b=60)
        )
        return fig
    except:
        raise dash.exceptions.PreventUpdate


def _get_predict_fn(model_type):
    if model_type == 'lstm':
        return lambda X: predict_lstm(GLOBAL_STATE['lstm_model'], X)
    elif model_type == 'xgb':
        return lambda X: predict_xgboost(GLOBAL_STATE['xgb_model'], X)
    else:
        w = GLOBAL_STATE.get('fusion_lstm_weight', 0.5)
        return lambda X: predict_fusion(GLOBAL_STATE['lstm_model'], GLOBAL_STATE['xgb_model'], X, w)


@app.callback(
    Output('warning-cards', 'children'),
    Input('run-warning-btn', 'n_clicks'),
    State('warning-start-time', 'value'),
    State('warning-filter-indicator', 'value'),
    State('warning-filter-level', 'value'),
    prevent_initial_call=True
)
def run_warning(n_clicks, start_idx, filter_ind, filter_level):
    if GLOBAL_STATE.get('lstm_model') is None and GLOBAL_STATE.get('xgb_model') is None:
        return html.Div('⚠️ 请先在「出水预测」标签页训练模型',
                        className='text-warning p-4 rounded',
                        style={'backgroundColor': '#fef9e7'})
    if GLOBAL_STATE['df'] is None:
        return html.Div('⚠️ 请先上传数据', className='text-warning')

    try:
        df = GLOBAL_STATE['df']
        time_col = GLOBAL_STATE['time_col']
        lookback, horizon = 24, 4

        if start_idx is None:
            start_idx = len(df) - lookback - horizon
        start_idx = max(lookback, min(int(start_idx), len(df) - lookback - horizon))

        model_type = GLOBAL_STATE.get('last_model_type', 'lstm')
        predict_fn = _get_predict_fn(model_type)

        feature_cols = [c for c in INFLOW_INDICATORS + PROCESS_PARAMS if c in df.columns]
        state_window = df.iloc[start_idx - lookback:start_idx][feature_cols].values.astype(np.float32)
        X_input = state_window.reshape(1, lookback, -1)
        prediction = predict_fn(X_input)[0]

        base_time = pd.to_datetime(df.iloc[start_idx][time_col])
        pred_times = [base_time + timedelta(hours=h + 1) for h in range(horizon)]

        if model_type == 'lstm':
            test_pred = GLOBAL_STATE['lstm_result']['test_pred']
        elif model_type == 'xgb':
            test_pred = GLOBAL_STATE['xgb_result']['test_pred']
        else:
            test_pred = GLOBAL_STATE['fusion_result']['test_pred']
        y_test = GLOBAL_STATE['y_test']

        viola_prob = estimate_violation_probability(model_type, y_test, test_pred,
                                                     prediction.reshape(1, horizon, len(KEY_PREDICTORS)), STANDARDS)

        warning_events = []
        for h in range(horizon):
            for i, name in enumerate(KEY_PREDICTORS):
                val = float(prediction[h, i])
                std = STANDARDS[name]
                if val > std:
                    level = 'red'; level_text = '🔴 红色预警'; color = '#fdecea'
                elif val > std * 0.9:
                    level = 'yellow'; level_text = '🟡 黄色预警'; color = '#fef9e7'
                else:
                    level = 'normal'; level_text = '✅ 正常'; color = '#eafaf1'

                prob = viola_prob.get(name, [0.5] * horizon)[h]

                if (filter_ind == 'all' or filter_ind == name) and \
                   (filter_level == 'all' or filter_level == level):
                    warning_events.append({
                        'time': pred_times[h], 'level': level, 'level_text': level_text,
                        'color': color, 'indicator': name, 'value': val, 'standard': std,
                        'probability': prob, 'horizon': h
                    })

        if not warning_events:
            return html.Div('🎉 当前筛选条件下无预警，出水预测全部达标',
                            className='text-success p-4 rounded fw-bold',
                            style={'backgroundColor': '#eafaf1'})

        cards = []
        for ev in sorted(warning_events, key=lambda x: x['time']):
            ind_disp = DISPLAY_NAMES.get(ev['indicator'], ev['indicator'])
            exceed_ratio = (ev['value'] - ev['standard']) / ev['standard'] * 100
            border_color = '#e74c3c' if ev['level'] == 'red' else ('#f39c12' if ev['level'] == 'yellow' else '#27ae60')
            cards.append(dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6(f"⏰ {ev['time'].strftime('%Y-%m-%d %H:00')} "
                                    f"<small class='text-muted'>(t+{ev['horizon']+1}h)</small>",
                                    className='mb-1'),
                            html.Span(ev['level_text'], className='fw-bold'),
                        ], width=5),
                        dbc.Col([
                            html.P(f'指标: <b>{ind_disp}</b>　|　'
                                   f'预测值: <b style="color:#d62728">{ev["value"]:.3f}</b> mg/L　|　'
                                   f'GB-1A标准: {ev["standard"]} mg/L',
                                   style={'margin': '4px 0', 'fontSize': '13px'},
                                   dangerously_allow_html=True),
                            html.P(f'超标幅度: <b>{exceed_ratio:+.1f}%</b>　|　'
                                   f'超标概率: <b>{ev["probability"]*100:.0f}%</b>　'
                                   f'(基于测试集超标频次估算)',
                                   style={'margin': '4px 0', 'fontSize': '13px'},
                                   dangerously_allow_html=True)
                        ], width=7)
                    ])
                ])
            ], style={'backgroundColor': ev['color'], 'marginBottom': '10px',
                      'borderLeft': f'4px solid {border_color}'}))

        return html.Div(cards)

    except Exception as e:
        import traceback
        return html.Div(f'❌ 预警出错: {str(e)}\n{traceback.format_exc()}',
                        className='text-danger', style={'whiteSpace': 'pre-wrap'})


@app.callback(
    Output('opt-result-text', 'children'),
    Output('opt-details', 'children'),
    Input('run-opt-btn', 'n_clicks'),
    State('warning-start-time', 'value'),
    State('opt-p-0', 'value'),
    State('opt-p-1', 'value'),
    State('opt-p-2', 'value'),
    State('opt-daily-flow', 'value'),
    prevent_initial_call=True
)
def run_optimization(n_clicks, start_idx, cur_aeration, cur_return, cur_carbon, daily_flow):
    if GLOBAL_STATE.get('lstm_model') is None and GLOBAL_STATE.get('xgb_model') is None:
        return '⚠️ 请先训练出水预测模型', None
    if GLOBAL_STATE['df'] is None:
        return '⚠️ 请先上传数据', None

    try:
        df = GLOBAL_STATE['df']
        lookback = 24
        if start_idx is None:
            start_idx = len(df) - lookback - 4
        start_idx = max(lookback, min(int(start_idx), len(df) - lookback - 4))

        model_type = GLOBAL_STATE.get('last_model_type', 'lstm')
        predict_fn = _get_predict_fn(model_type)

        feature_cols = [c for c in INFLOW_INDICATORS + PROCESS_PARAMS if c in df.columns]
        state_window = df.iloc[start_idx - lookback:start_idx][feature_cols].values.astype(np.float32)

        current_params = {
            '曝气量': float(cur_aeration or 5000),
            '污泥回流比': float(cur_return or 100),
            '碳源投加量': float(cur_carbon or 80)
        }

        opt_result = optimize_process(
            predict_fn, state_window, current_params,
            daily_flow=float(daily_flow or DEFAULT_DAILY_FLOW),
            max_iter=60
        )

        suggestion_text = generate_suggestion_text(opt_result)

        rows = []
        for p, unit, delta in [
            ('曝气量', 'm³/h', opt_result['adjustments']['曝气量']),
            ('污泥回流比', '%', opt_result['adjustments']['污泥回流比']),
            ('碳源投加量', 'L/h', opt_result['adjustments']['碳源投加量']),
        ]:
            cur_v = opt_result['current_params'][f'{p}({unit})']
            opt_v = opt_result['opt_params'][f'{p}({unit})']
            delta_color = 'green' if delta < 0 else ('#d62728' if delta > 0 else 'grey')
            arrow = '↑' if delta > 0 else ('↓' if delta < 0 else '→')
            rows.append(html.Tr([
                html.Td(f'{p} ({unit})'),
                html.Td(f'{cur_v:.1f}'),
                html.Td(html.B(f'{opt_v:.1f}', className='text-primary')),
                html.Td(f'{arrow} {abs(delta):.1f} {unit}', style={'color': delta_color, 'fontWeight': 'bold'})
            ]))

        pred_rows = []
        for i, ind in enumerate(KEY_PREDICTORS):
            cur_mean = opt_result['current_predictions'][:, i].mean()
            opt_mean = opt_result['opt_predictions'][:, i].mean()
            std = STANDARDS[ind]
            cur_color = '#d62728' if cur_mean > std else '#2c3e50'
            opt_color = '#27ae60' if opt_mean <= std else '#d62728'
            pred_rows.append(html.Tr([
                html.Td(DISPLAY_NAMES.get(ind)),
                html.Td(f'{std} mg/L'),
                html.Td(f'{cur_mean:.3f} mg/L', style={'color': cur_color, 'fontWeight': 'bold'}),
                html.Td(f'{opt_mean:.3f} mg/L', style={'color': opt_color, 'fontWeight': 'bold'})
            ]))

        param_compare = html.Div([
            html.H6('📌 参数调整方案', className='mt-3 mb-2 text-primary'),
            dbc.Table([
                html.Thead(html.Tr([html.Th('参数'), html.Th('当前值'), html.Th('建议值'), html.Th('调整方向')])),
                html.Tbody(rows)
            ], bordered=True, hover=True, striped=True, size='sm'),
            html.H6('📊 出水预测对比 (优化前 → 优化后)', className='mt-4 mb-2 text-primary'),
            dbc.Table([
                html.Thead(html.Tr([html.Th('指标'), html.Th('GB-1A标准'), html.Th('当前预测均值'), html.Th('优化后预测均值')])),
                html.Tbody(pred_rows)
            ], bordered=True, hover=True, striped=True, size='sm'),
            html.Div([
                html.Hr(),
                html.P([
                    '💡 目标函数：min 总能耗 = ',
                    html.Code(f'{BLOWER_COEFF}×曝气量 + {RETURN_COEFF}×回流比 + {CARBON_COST}×碳源量'),
                    '　约束：出水满足GB-1A标准',
                ], className='text-muted fs-7', style={'fontSize': '12px'})
            ])
        ])

        return suggestion_text, param_compare

    except Exception as e:
        import traceback
        return f'❌ 优化出错: {str(e)}\n{traceback.format_exc()}', None


@app.callback(
    Output('energy-trend-chart', 'figure'),
    Output('energy-pie-chart', 'figure'),
    Output('energy-monthly-chart', 'figure'),
    Output('energy-anomaly-text', 'children'),
    Input('main-tabs', 'active_tab'),
    State('stats-table', 'data'),
    prevent_initial_call=True
)
def update_energy(tab, stats_data):
    if tab != 'tab-energy' or GLOBAL_STATE['df'] is None:
        return go.Figure(), go.Figure(), go.Figure(), '加载中...'

    try:
        df = GLOBAL_STATE['df'].copy()
        time_col = GLOBAL_STATE['time_col']
        df[time_col] = pd.to_datetime(df[time_col])
        df['date'] = df[time_col].dt.date

        aerate = df.get('曝气量', pd.Series(5000, index=df.index))
        ret = df.get('污泥回流比', pd.Series(100, index=df.index))
        carbon = df.get('碳源投加量', pd.Series(80, index=df.index))
        flow_hourly = df.get('处理水量', pd.Series(DEFAULT_DAILY_FLOW/24, index=df.index))

        df['鼓风机能耗'] = BLOWER_COEFF * aerate
        df['回流泵能耗'] = RETURN_COEFF * ret
        df['加药能耗'] = CARBON_COST * carbon
        df['总能耗瞬时'] = df['鼓风机能耗'] + df['回流泵能耗'] + df['加药能耗']

        daily = df.groupby('date').agg({
            '鼓风机能耗': 'sum', '回流泵能耗': 'sum', '加药能耗': 'sum',
            '总能耗瞬时': 'sum', '处理水量': lambda x: x.sum() if '处理水量' in df.columns else DEFAULT_DAILY_FLOW
        }).reset_index()
        daily['吨水能耗'] = daily['总能耗瞬时'] / (daily['处理水量'].replace(0, np.nan))
        daily['date'] = pd.to_datetime(daily['date'])

        fig_trend = go.Figure()
        for col, color, name in [
            ('鼓风机能耗', '#1f77b4', '鼓风机'),
            ('回流泵能耗', '#ff7f0e', '回流泵'),
            ('加药能耗', '#2ca02c', '加药系统')
        ]:
            fig_trend.add_trace(go.Scatter(
                x=daily['date'], y=daily[col] / 1000, mode='lines',
                name=name+' (kWh/d)', line=dict(color=color, width=2),
                stackgroup='one' if col == '鼓风机能耗' else None
            ))
        fig_trend.update_layout(title='三项主要设备日能耗趋势 (kWh/日)', template='plotly_white',
                                hovermode='x unified', yaxis_title='kWh/日',
                                margin=dict(l=60, r=20, t=60, b=40))

        pie_labels = ['鼓风机能耗', '回流泵能耗', '加药能耗']
        pie_vals = [daily['鼓风机能耗'].sum(), daily['回流泵能耗'].sum(), daily['加药能耗'].sum()]
        fig_pie = go.Figure(data=[go.Pie(
            labels=pie_labels, values=pie_vals, hole=0.45,
            marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c']),
            textinfo='label+percent', textfont_size=12
        )])
        fig_pie.update_layout(title='能耗构成占比', template='plotly_white',
                              margin=dict(l=10, r=10, t=60, b=10), showlegend=False)

        daily['month'] = daily['date'].dt.to_period('M').astype(str)
        monthly = daily.groupby('month')['吨水能耗'].mean().reset_index()

        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Bar(
            x=monthly['month'], y=monthly['吨水能耗'],
            name='吨水综合能耗 (kWh/m³)',
            marker=dict(color=monthly['吨水能耗'], colorscale='Viridis',
                        showscale=True, colorbar=dict(title='kWh/m³'))
        ))
        fig_monthly.update_layout(title='吨水综合能耗月度趋势 (kWh/m³)', template='plotly_white',
                                  yaxis_title='kWh/m³', margin=dict(l=60, r=60, t=60, b=40))

        monthly_energy = daily['吨水能耗'].mean()
        anomaly_days = daily[daily['吨水能耗'] > monthly_energy * 1.5]

        parts = []
        if len(anomaly_days) > 0:
            parts.append(html.P(f'🔍 发现 {len(anomaly_days)} 个能耗异常日 (>月均×1.5)',
                                className='text-danger fw-bold'))
            inflow_mean = df.get('COD_in', pd.Series(0)).mean()
            flow_mean = daily['处理水量'].mean()
            for _, row in anomaly_days.head(8).iterrows():
                day_mask = pd.to_datetime(df[time_col]).dt.date == row['date'].date()
                day_cod = df.loc[day_mask, 'COD_in'].mean() if 'COD_in' in df.columns else np.nan
                day_flow = row['处理水量']
                cod_ratio = (day_cod / inflow_mean - 1) * 100 if inflow_mean > 0 else 0
                flow_ratio = (day_flow / flow_mean - 1) * 100 if flow_mean > 0 else 0
                parts.append(html.P(
                    f"📅 {row['date'].strftime('%Y-%m-%d')}: "
                    f"能耗={row['吨水能耗']:.3f} kWh/m³<br>"
                    f"COD负荷{'↑' if cod_ratio>0 else '↓'}{abs(cod_ratio):.0f}%　"
                    f"水量{'↑' if flow_ratio>0 else '↓'}{abs(flow_ratio):.0f}%",
                    style={'fontSize': '12px', 'margin': '4px 0', 'borderLeft': '3px solid #e74c3c',
                           'paddingLeft': '8px'},
                    dangerously_allow_html=True
                ))
        else:
            parts.append(html.P('✅ 未发现显著能耗异常日', className='text-success'))
        parts.append(html.Hr())
        parts.append(html.P(f'📊 全周期月均吨水能耗: <b>{monthly_energy:.4f}</b> kWh/m³',
                            className='fs-7', dangerously_allow_html=True))

        return fig_trend, fig_pie, fig_monthly, html.Div(parts)

    except Exception as e:
        import traceback
        return go.Figure(), go.Figure(), go.Figure(), f'出错: {str(e)}'


@app.callback(
    Output('srt-chart', 'figure'),
    Output('mlss-sv30-chart', 'figure'),
    Input('main-tabs', 'active_tab'),
    prevent_initial_call=True
)
def update_sludge(tab):
    if tab != 'tab-sludge' or GLOBAL_STATE['df'] is None:
        return go.Figure(), go.Figure()

    try:
        df = GLOBAL_STATE['df'].copy()
        time_col = GLOBAL_STATE['time_col']
        df[time_col] = pd.to_datetime(df[time_col])
        df['date'] = df[time_col].dt.date

        reactor_volume = 10000
        waste = df.get('剩余污泥排放量', pd.Series(50, index=df.index))
        mlss = df.get('MLSS', pd.Series(3000, index=df.index))

        daily = df.groupby('date').agg({
            '剩余污泥排放量': 'sum', 'MLSS': 'mean'
        }).reset_index()
        daily.columns = ['date', '日排泥量_m3', 'MLSS_均值']
        daily['MLSS_kg'] = reactor_volume * daily['MLSS_均值'] / 1000
        daily['排泥_MLSS_kg'] = daily['日排泥量_m3'] * daily['MLSS_均值'] / 1000
        daily['SRT'] = daily['MLSS_kg'] / daily['排泥_MLSS_kg'].replace(0, np.nan)
        daily['SRT'] = daily['SRT'].clip(lower=0.5, upper=60)
        daily['date'] = pd.to_datetime(daily['date'])

        fig_srt = go.Figure()
        fig_srt.add_trace(go.Scatter(
            x=daily['date'], y=daily['SRT'], mode='lines+markers',
            name='SRT', line=dict(color='#8e44ad', width=2),
            marker=dict(size=4, color=['#e74c3c' if (v < 8 or v > 25) else '#8e44ad' for v in daily['SRT']])
        ))
        fig_srt.add_hrect(y0=0, y1=8, fillcolor='#e74c3c', opacity=0.08,
                          annotation_text='SRT<8天 (泥龄过短)', annotation_position='top left')
        fig_srt.add_hrect(y0=25, y1=60, fillcolor='#e74c3c', opacity=0.08,
                          annotation_text='SRT>25天 (泥龄过长)', annotation_position='top left')
        fig_srt.add_hline(y=8, line_dash='dash', line_color='#e74c3c', opacity=0.7)
        fig_srt.add_hline(y=25, line_dash='dash', line_color='#e74c3c', opacity=0.7)
        fig_srt.update_layout(
            title='污泥龄 SRT 每日趋势 (正常区间: 8~25天)',
            template='plotly_white', yaxis_title='天数',
            yaxis=dict(range=[0, 40]),
            margin=dict(l=50, r=20, t=60, b=40)
        )

        sv30_vals = df.get('SV30', pd.Series(30, index=df.index)).values
        mlss_vals = df.get('MLSS', pd.Series(3000, index=df.index)).values
        valid = ~(np.isnan(sv30_vals) | np.isnan(mlss_vals))
        sv30_v = sv30_vals[valid]
        mlss_v = mlss_vals[valid]

        colors = ['#e74c3c' if v > 70 else '#27ae60' for v in sv30_v]
        fig_mlss = go.Figure()
        fig_mlss.add_trace(go.Scatter(
            x=mlss_v, y=sv30_v, mode='markers',
            marker=dict(color=colors, size=6, opacity=0.7,
                        line=dict(width=0.5, color='DarkSlateGrey')),
            name='数据点',
            text=[f'MLSS={m:.0f}mg/L<br>SV30={s:.1f}%<br>' + ('膨胀风险' if s > 70 else '正常')
                  for s, m in zip(sv30_v, mlss_v)],
            hovertemplate='%{text}<extra></extra>'
        ))

        if len(mlss_v) > 5:
            try:
                z = np.polyfit(mlss_v, sv30_v, 1)
                x_fit = np.linspace(mlss_v.min(), mlss_v.max(), 100)
                y_fit = np.poly1d(z)(x_fit)
                fig_mlss.add_trace(go.Scatter(
                    x=x_fit, y=y_fit, mode='lines',
                    name=f'拟合: SV30 = {z[0]:.4f}·MLSS + {z[1]:.2f}',
                    line=dict(color='#2c3e50', dash='dot', width=2)
                ))
            except:
                pass

        fig_mlss.add_hrect(y0=70, y1=100, fillcolor='#e74c3c', opacity=0.08,
                           annotation_text='SV30>70% 膨胀预警')
        fig_mlss.add_hline(y=70, line_dash='dash', line_color='#e74c3c',
                           annotation_text='膨胀预警线 70%')
        fig_mlss.update_layout(
            title='MLSS vs SV30 散点图 (红色=SV30>70% 污泥膨胀风险)',
            template='plotly_white',
            xaxis_title='MLSS (mg/L)', yaxis_title='SV30 (%)',
            yaxis=dict(range=[0, min(100, sv30_v.max()*1.2) if len(sv30_v)>0 else [0,100]]),
            margin=dict(l=50, r=20, t=60, b=40)
        )

        return fig_srt, fig_mlss

    except Exception as e:
        import traceback
        return go.Figure(), go.Figure()


@app.callback(
    Output('season-bar-chart', 'figure'),
    Output('winter-nitrification-chart', 'figure'),
    Output('season-table', 'columns'),
    Output('season-table', 'data'),
    Input('season-metric-select', 'value'),
    State('main-tabs', 'active_tab'),
    prevent_initial_call=True
)
def update_seasonal(metric, tab):
    if tab != 'tab-seasonal' or GLOBAL_STATE['df'] is None:
        return go.Figure(), go.Figure(), [], []

    try:
        df = GLOBAL_STATE['df'].copy()
        time_col = GLOBAL_STATE['time_col']
        df[time_col] = pd.to_datetime(df[time_col])
        df['month'] = df[time_col].dt.month
        df['season'] = df['month'].map(SEASON_MAP)
        df['date'] = df[time_col].dt.date

        seasons_order = ['春季', '夏季', '秋季', '冬季']
        outflow_cols = [c for c in OUTFLOW_INDICATORS if c in df.columns]

        season_stats = df.groupby('season')[outflow_cols].mean()

        daily = df.groupby('date').agg({c: 'mean' for c in outflow_cols}).reset_index()
        daily['date'] = pd.to_datetime(daily['date'])
        daily['month'] = daily['date'].dt.month
        daily['season'] = daily['month'].map(SEASON_MAP)

        compliance = {}
        for season in seasons_order:
            sd = daily[daily['season'] == season]
            row = {}
            for c in outflow_cols:
                if len(sd) == 0:
                    row[c] = np.nan
                    continue
                std_c = STANDARDS.get(c, None)
                if std_c is not None:
                    rate = (sd[c] <= std_c).mean() * 100
                elif c == 'pH_out':
                    rate = ((sd[c] >= 6) & (sd[c] <= 9)).mean() * 100
                else:
                    rate = 100.0
                row[c] = rate
            compliance[season] = row

        aerate = df.get('曝气量', pd.Series(5000))
        ret = df.get('污泥回流比', pd.Series(100))
        carbon = df.get('碳源投加量', pd.Series(80))
        df['吨水能耗_瞬时'] = BLOWER_COEFF * aerate + RETURN_COEFF * ret + CARBON_COST * carbon
        season_energy = df.groupby('season')['吨水能耗_瞬时'].mean() * 24 / (DEFAULT_DAILY_FLOW / 24) * 24
        season_energy = df.groupby('season').apply(
            lambda g: (BLOWER_COEFF * g['曝气量'].mean() if '曝气量' in g else BLOWER_COEFF*5000) +
                      (RETURN_COEFF * g['污泥回流比'].mean() if '污泥回流比' in g else RETURN_COEFF*100) +
                      (CARBON_COST * g['碳源投加量'].mean() if '碳源投加量' in g else CARBON_COST*80)
        ) if not season_energy.any() else season_energy

        season_carbon = df.groupby('season')['碳源投加量'].mean() if '碳源投加量' in df.columns else \
                        pd.Series({s: 80 for s in seasons_order})

        if metric in outflow_cols:
            vals = [season_stats.loc[s, metric] if s in season_stats.index else 0 for s in seasons_order]
            y_title = DISPLAY_NAMES.get(metric, metric)
            if metric in STANDARDS:
                y_title += f' (GB-1A ≤{STANDARDS[metric]})'
        elif metric == '吨水能耗':
            vals = [season_energy.loc[s] if s in season_energy.index else 0 for s in seasons_order]
            y_title = '吨水综合能耗 (kWh/m³)'
        else:
            vals = [season_carbon.loc[s] if s in season_carbon.index else 80 for s in seasons_order]
            y_title = '平均碳源投加量 (L/h)'

        bar_colors = {'春季': '#27ae60', '夏季': '#e67e22', '秋季': '#d4ac0d', '冬季': '#3498db'}
        colors_list = [bar_colors[s] for s in seasons_order]

        fig_bar = go.Figure(go.Bar(
            x=seasons_order, y=vals,
            marker=dict(color=colors_list, line=dict(color='white', width=2)),
            text=[f'{v:.2f}' for v in vals], textposition='outside',
            width=0.6
        ))
        if metric in STANDARDS:
            fig_bar.add_hline(y=STANDARDS[metric], line_dash='dash', line_color='#e74c3c',
                              annotation_text=f'GB-1A标准 {STANDARDS[metric]}')
        fig_bar.update_layout(
            title=f'四季 {y_title} 均值对比',
            template='plotly_white', yaxis_title=y_title,
            margin=dict(l=60, r=20, t=60, b=40)
        )

        winter_mask = df['month'].isin([12, 1, 2])
        winter_df = df[winter_mask]
        nh3n = winter_df.get('NH3N_out', pd.Series(dtype=float))
        temp = winter_df.get('水温', pd.Series(dtype=float))

        fig_winter = go.Figure()
        if len(nh3n.dropna()) > 0 and len(temp.dropna()) > 5:
            valid = ~nh3n.isna() & ~temp.isna()
            nh3n_v = nh3n[valid].values
            temp_v = temp[valid].values
            c_win = ['#e74c3c' if v > 5 else '#3498db' for v in nh3n_v]
            fig_winter.add_trace(go.Scatter(
                x=temp_v, y=nh3n_v, mode='markers',
                marker=dict(color=c_win, size=6, opacity=0.7,
                            line=dict(width=0.5, color='DarkSlateGrey')),
                text=[f'水温={t:.1f}℃<br>出水NH3-N={n:.2f}mg/L<br>'+
                      ('超标' if n>5 else '达标')
                      for t, n in zip(temp_v, nh3n_v)],
                hovertemplate='%{text}<extra></extra>',
                name='样本点'
            ))
            try:
                z = np.polyfit(temp_v, nh3n_v, 1)
                x_fit = np.linspace(temp_v.min(), temp_v.max(), 50)
                y_fit = np.poly1d(z)(x_fit)
                trend_desc = '负相关(水温↑硝化↑)' if z[0] < 0 else '正相关'
                fig_winter.add_trace(go.Scatter(
                    x=x_fit, y=y_fit, mode='lines',
                    name=f'拟合线({trend_desc}) NH3-N = {z[0]:.3f}·T + {z[1]:.2f}',
                    line=dict(color='#2c3e50', dash='dot', width=2.5)
                ))
            except:
                pass

        fig_winter.add_hrect(y0=5, y1=20, fillcolor='#e74c3c', opacity=0.08,
                             annotation_text='NH3-N>5mg/L 超标')
        fig_winter.add_hline(y=5, line_dash='dash', line_color='#e74c3c',
                             annotation_text='GB-1A 标准 5mg/L')
        fig_winter.update_layout(
            title='冬季(12~2月)低温期硝化效率分析<br><sup>水温 vs 出水氨氮 — 验证低温抑制效应</sup>',
            template='plotly_white',
            xaxis_title='水温 (℃)', yaxis_title='出水NH3-N (mg/L)',
            margin=dict(l=50, r=20, t=80, b=40)
        )

        table_data = []
        for season in seasons_order:
            row = {'季节': season}
            for c in outflow_cols:
                dn = DISPLAY_NAMES.get(c, c).replace('出水', '')
                sv = season_stats.loc[season, c] if (season in season_stats.index and c in season_stats.columns) else np.nan
                cv = compliance.get(season, {}).get(c, np.nan)
                row[f'{dn} 均值'] = f'{sv:.2f}' if not np.isnan(sv) else '-'
                row[f'{dn} 达标率%'] = f'{cv:.1f}' if not np.isnan(cv) else '-'
            if season in season_energy.index:
                row['吨水能耗(kWh/m³)'] = f'{season_energy.loc[season]:.3f}'
            table_data.append(row)

        table_cols = [{'name': c, 'id': c} for c in table_data[0].keys()] if table_data else []

        return fig_bar, fig_winter, table_cols, table_data

    except Exception as e:
        import traceback
        return go.Figure(), go.Figure(), [], [f'错误: {str(e)}']


@app.callback(
    Output('download-report', 'data'),
    Input('generate-report-btn', 'n_clicks'),
    State('report-date-select', 'value'),
    State('warning-start-time', 'value'),
    prevent_initial_call=True
)
def export_report(n_clicks, date_str, warn_idx):
    if GLOBAL_STATE['df'] is None or not date_str:
        raise dash.exceptions.PreventUpdate

    try:
        df = GLOBAL_STATE['df'].copy()
        time_col = GLOBAL_STATE['time_col']

        warnings_list = []
        suggestions_text = '本报告由系统自动生成，工艺建议请结合现场实际情况使用。'
        energy_stats = {}

        try:
            if GLOBAL_STATE.get('lstm_model') is not None or GLOBAL_STATE.get('xgb_model') is not None:
                lookback, horizon = 24, 4
                model_type = GLOBAL_STATE.get('last_model_type', 'lstm')
                predict_fn = _get_predict_fn(model_type)

                dates = pd.to_datetime(df[time_col]).dt.strftime('%Y-%m-%d')
                day_mask = dates == date_str
                day_indices = np.where(day_mask)[0]

                warnings_list = []
                if len(day_indices) > 0:
                    for start_hour in [0, 6, 12, 18]:
                        candidate_start = day_indices[0] + start_hour * 1
                        if candidate_start >= lookback and candidate_start < len(df) - horizon:
                            feature_cols = [c for c in INFLOW_INDICATORS + PROCESS_PARAMS if c in df.columns]
                            window = df.iloc[candidate_start - lookback:candidate_start][feature_cols].values.astype(np.float32)
                            pred = predict_fn(window.reshape(1, lookback, -1))[0]
                            base_time = pd.to_datetime(df.iloc[candidate_start][time_col])
                            for h in range(horizon):
                                for i, name in enumerate(KEY_PREDICTORS):
                                    val = float(pred[h, i])
                                    std = STANDARDS[name]
                                    if val > std * 0.9:
                                        level = '红色' if val > std else '黄色'
                                        warnings_list.append({
                                            'time': (base_time + timedelta(hours=h + 1)).strftime('%Y-%m-%d %H:00'),
                                            'level': f'{level}预警',
                                            'indicator': DISPLAY_NAMES.get(name, name),
                                            'value': f'{val:.3f} mg/L',
                                            'standard': f'{std} mg/L',
                                            'probability': f'{min(max(val/std, 0.5), 0.99)*100:.0f}%'
                                        })

                if model_type == 'lstm':
                    test_pred = GLOBAL_STATE['lstm_result']['test_pred']
                elif model_type == 'xgb':
                    test_pred = GLOBAL_STATE['xgb_result']['test_pred']
                else:
                    test_pred = GLOBAL_STATE['fusion_result']['test_pred']
                y_test = GLOBAL_STATE['y_test']
                metrics = compute_metrics(y_test, test_pred)
                avg_mape = np.mean([m['MAPE'] for m in metrics.values()])
                suggestions_text += f'\n\n当前模型整体MAPE={avg_mape:.2f}%，建议每7天重新训练一次模型。'
        except Exception as ex:
            suggestions_text += f'\n(模型分析部分暂不可用: {str(ex)[:50]})'

        try:
            aerate = df.get('曝气量', pd.Series(5000)).mean()
            ret_p = df.get('污泥回流比', pd.Series(100)).mean()
            carb = df.get('碳源投加量', pd.Series(80)).mean()
            daily_e = (BLOWER_COEFF * aerate + RETURN_COEFF * ret_p + CARBON_COST * carb) * 24
            energy_stats = {
                '日总能耗 (kWh)': daily_e,
                '吨水能耗 (kWh/m³)': daily_e / DEFAULT_DAILY_FLOW,
                '日运行成本估算 (¥)': daily_e * 0.6,
                '鼓风机能耗占比 (%)': f'{BLOWER_COEFF * aerate / (BLOWER_COEFF*aerate+RETURN_COEFF*ret_p+CARBON_COST*carb)*100:.1f}',
                '回流泵能耗占比 (%)': f'{RETURN_COEFF * ret_p / (BLOWER_COEFF*aerate+RETURN_COEFF*ret_p+CARBON_COST*carb)*100:.1f}',
                '加药能耗占比 (%)': f'{CARBON_COST * carb / (BLOWER_COEFF*aerate+RETURN_COEFF*ret_p+CARBON_COST*carb)*100:.1f}',
            }
        except:
            pass

        pdf_bytes = generate_daily_report(df, date_str, warnings_list, suggestions_text, energy_stats)

        return dcc.send_bytes(pdf_bytes, f"污水厂运行日报_{date_str}.pdf")

    except Exception as e:
        import traceback
        return dict(content=str(e), filename='error.txt', type='text/plain')


if __name__ == '__main__':
    print('='*70)
    print('  城镇污水处理厂出水水质预测与工艺调控分析平台')
    print('='*70)
    print('  技术栈: Dash + Plotly + PyTorch(LSTM) + XGBoost + SciPy优化')
    print('  启动后访问: http://127.0.0.1:8050/')
    print('='*70)
    app.run(debug=False, host='0.0.0.0', port=8050)


