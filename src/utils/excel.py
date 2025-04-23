#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description: 
Version: 
Autor: dreamy-xay
Date: 2024-10-22 14:49:10
LastEditors: dreamy-xay
LastEditTime: 2024-10-22 14:49:13
"""
from openpyxl import load_workbook
from openpyxl.styles import Border, Side, Alignment


def format_excel(
    xlsx_path,
    offset=(1, 1),
    row_width_extend=2,
    row_height=20,
    alignment=("center", "center"),
    side_config={"border_style": "thin", "color": "000000"},
):
    # 使用 openpyxl 加载 Excel 文件
    wb = load_workbook(xlsx_path)

    # 对每个工作表进行一些自定义操作
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]  # 获取每个工作表对象

        # 设置实线边框（跳过最左侧列和最上侧行）
        border = Border(left=Side(**side_config), right=Side(**side_config), top=Side(**side_config), bottom=Side(**side_config))

        # 为所有单元格添加实线边框（但不包括最左列和最上行）
        for row in ws.iter_rows(min_row=1, max_row=ws.max_row, min_col=1, max_col=ws.max_column):
            for cell in row:
                cell.border = border

        # 向下偏移并向右偏移（插入空行和空列）
        for _ in range(offset[0]):  # 向下偏移
            ws.insert_rows(1)
        for _ in range(offset[1]):  # 向右偏移
            ws.insert_cols(1)

        # 自动调整列宽
        for col in ws.columns:
            max_length = 0
            column = col[0].column_letter  # 获取列字母
            for cell in col:
                if cell.value:
                    try:
                        max_length = max(max_length, len(str(cell.value)))
                    except:
                        pass
            if max_length != 0:
                adjusted_width = max_length + row_width_extend  # 加 2 为额外的空间
                ws.column_dimensions[column].width = adjusted_width

        # 自动调整行高
        for row in ws.iter_rows():
            ws.row_dimensions[row[0].row].height = row_height

        # 设置所有单元格内容居中
        for row in ws.iter_rows():
            for cell in row:
                cell.alignment = Alignment(horizontal=alignment[0], vertical=alignment[1])

    # 保存修改后的 Excel 文件
    wb.save(xlsx_path)
