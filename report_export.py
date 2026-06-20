import os
import numpy as np
import pandas as pd
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io


def _register_chinese_font():
    font_paths = [
        'C:/Windows/Fonts/msyh.ttc',
        'C:/Windows/Fonts/msyh.ttf',
        'C:/Windows/Fonts/simhei.ttf',
        'C:/Windows/Fonts/simsun.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/System/Library/Fonts/PingFang.ttc',
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                pdfmetrics.registerFont(TTFont('ChineseFont', fp))
                return True
            except:
                continue
    return False


def generate_daily_report(df: pd.DataFrame, date_str: str,
                          warnings_list: list, suggestions: str,
                          energy_stats: dict, output_path: str = None) -> bytes:
    has_font = _register_chinese_font()
    font_name = 'ChineseFont' if has_font else 'Helvetica'

    if output_path is None:
        buffer = io.BytesIO()
    else:
        buffer = output_path

    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Heading1'],
        fontName=font_name, fontSize=18, leading=24,
        alignment=TA_CENTER, spaceAfter=20, textColor=colors.HexColor('#1a5276')
    )
    h2_style = ParagraphStyle(
        'CustomH2', parent=styles['Heading2'],
        fontName=font_name, fontSize=13, leading=18,
        spaceBefore=15, spaceAfter=8, textColor=colors.HexColor('#2874a6')
    )
    h3_style = ParagraphStyle(
        'CustomH3', parent=styles['Heading3'],
        fontName=font_name, fontSize=11, leading=14,
        spaceBefore=10, spaceAfter=6, textColor=colors.HexColor('#2e86c1')
    )
    body_style = ParagraphStyle(
        'CustomBody', parent=styles['BodyText'],
        fontName=font_name, fontSize=10, leading=14,
        alignment=TA_LEFT, spaceAfter=6
    )
    warn_style = ParagraphStyle(
        'WarnStyle', parent=body_style,
        textColor=colors.HexColor('#c0392b'), fontSize=10
    )

    story = []

    story.append(Paragraph('城镇污水处理厂运行日报', title_style))
    story.append(Paragraph(f'报告日期：{date_str}　　生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                           ParagraphStyle('Subtitle', parent=body_style, alignment=TA_CENTER,
                                          textColor=colors.grey)))
    story.append(Spacer(1, 1*cm))

    date_mask = pd.to_datetime(df.iloc[:, 0]).dt.date == pd.to_datetime(date_str).date()
    day_data = df[date_mask]

    story.append(Paragraph('一、进出水指标摘要', h2_style))
    inflow_cols = [c for c in ['COD_in', 'BOD5_in', 'SS_in', 'TN_in', 'TP_in', 'NH3N_in', 'pH_in'] if c in day_data.columns]
    outflow_cols = [c for c in ['COD_out', 'BOD5_out', 'SS_out', 'TN_out', 'TP_out', 'NH3N_out', 'pH_out'] if c in day_data.columns]

    if len(day_data) > 0:
        table_data = [['指标类别', '指标', '日均值', '最大值', '最小值', '达标率(%)']]
        display_map = {'COD': 'COD', 'BOD5': 'BOD5', 'SS': 'SS', 'TN': 'TN', 'TP': 'TP', 'NH3N': 'NH3-N', 'pH': 'pH'}
        std_map = {'COD_out': 50, 'BOD5_out': 10, 'SS_out': 10, 'TN_out': 15, 'TP_out': 0.5, 'NH3N_out': 5, 'pH_out': None}
        ph_range = (6, 9)

        for c in inflow_cols:
            name = c.replace('_in', '')
            dn = display_map.get(name, name)
            vals = day_data[c].dropna()
            if len(vals) > 0:
                table_data.append(['进水', f'{dn}', f'{vals.mean():.2f}', f'{vals.max():.2f}', f'{vals.min():.2f}', '-'])

        for c in outflow_cols:
            name = c.replace('_out', '')
            dn = display_map.get(name, name)
            vals = day_data[c].dropna()
            if len(vals) > 0:
                std = std_map.get(c)
                if std is not None:
                    rate = (vals <= std).mean() * 100
                elif c == 'pH_out':
                    rate = ((vals >= ph_range[0]) & (vals <= ph_range[1])).mean() * 100
                else:
                    rate = 100.0
                table_data.append(['出水', f'{dn}', f'{vals.mean():.2f}', f'{vals.max():.2f}', f'{vals.min():.2f}', f'{rate:.1f}'])

        t = Table(table_data, colWidths=[2*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm])
        t.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#d6eaf8')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f4f6f7')]),
        ]))
        story.append(t)
    else:
        story.append(Paragraph('当日无数据', body_style))

    story.append(Spacer(1, 0.8*cm))
    story.append(Paragraph('二、预警事件列表', h2_style))

    if warnings_list and len(warnings_list) > 0:
        warn_data = [['序号', '预警时间', '预警等级', '超标指标', '预测值', '标准值', '超标概率']]
        level_color = {}
        for idx, w in enumerate(warnings_list, 1):
            warn_data.append([
                str(idx),
                w.get('time', ''),
                w.get('level', ''),
                w.get('indicator', ''),
                w.get('value', ''),
                w.get('standard', ''),
                w.get('probability', '')
            ])
        wt = Table(warn_data, colWidths=[1.2*cm, 3*cm, 2*cm, 2.5*cm, 2*cm, 2*cm, 2.3*cm])
        style_cmds = [
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#fadbd8')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]
        for idx, w in enumerate(warnings_list, 1):
            if '红色' in w.get('level', ''):
                style_cmds.append(('BACKGROUND', (0, idx), (-1, idx), colors.HexColor('#fdecea')))
            elif '黄色' in w.get('level', ''):
                style_cmds.append(('BACKGROUND', (0, idx), (-1, idx), colors.HexColor('#fef9e7')))
        wt.setStyle(TableStyle(style_cmds))
        story.append(wt)
    else:
        story.append(Paragraph('✅ 当日无预警事件，出水水质稳定达标。', body_style))

    story.append(Spacer(1, 0.8*cm))
    story.append(Paragraph('三、工艺调控建议', h2_style))
    for line in suggestions.split('\n'):
        if line.strip():
            if '警告' in line or '⚠️' in line:
                story.append(Paragraph(line, warn_style))
            else:
                story.append(Paragraph(line, body_style))

    story.append(Spacer(1, 0.8*cm))
    story.append(Paragraph('四、能耗统计', h2_style))
    if energy_stats:
        en_data = [['统计项', '数值', '单位']]
        for k, v in energy_stats.items():
            if isinstance(v, (int, float)):
                en_data.append([k, f'{v:.4f}' if isinstance(v, float) else str(v), ''])
            else:
                en_data.append([k, str(v), ''])
        et = Table(en_data, colWidths=[5*cm, 5*cm, 5*cm])
        et.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#d5f5e3')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(et)

    story.append(Spacer(1, 1*cm))
    story.append(Paragraph('————— 本报告由系统自动生成 —————',
                           ParagraphStyle('Footer', parent=body_style, alignment=TA_CENTER, textColor=colors.grey)))

    doc.build(story)

    if isinstance(buffer, io.BytesIO):
        return buffer.getvalue()
    return None
