#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立的示例数据生成脚本
可独立运行生成CSV格式的模拟污水厂运行数据
"""
import sys
sys.path.insert(0, '.')
from app import generate_sample_data

if __name__ == '__main__':
    print('正在生成模拟数据...')
    df = generate_sample_data()
    output_file = 'sample_sewage_data.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f'✅ 已生成 {len(df)} 条记录（90天×24小时）')
    print(f'📁 保存至: {output_file}')
    print(f'\n数据字段说明:')
    for c in df.columns:
        print(f'  • {c}')
