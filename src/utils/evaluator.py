#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description:
Version:
Autor: dreamy-xay
Date: 2024-10-31 20:39:17
LastEditors: dreamy-xay
LastEditTime: 2024-11-16 16:36:28
"""
import re
from shapely.geometry import Polygon
from typing_extensions import Literal, Any
from table_recognition_metric import TEDS
from scitsr.eval import compare_rel, json2Relations
from . import cell_adj_relation
from .interpolate import split_unicom_area
from .quads_utils import build_relation_map

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except:
    pass


class BaseEvaluator(object):
    EPS = 1e-6

    def evaluate(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Subclasses should implement 'evaluate' method")

    @staticmethod
    def calc_f1_score(precision, recall):
        return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0


class TableStructEvaluator(BaseEvaluator):
    """table"""

    def __init__(self, pred_tables, gt_tables, eval_teds=True):
        self.eval_teds = eval_teds

        pred_logic_coords = [((item[0], item[1]), (item[2], item[3]), "") for pred_cells in pred_tables for _, item in pred_cells]
        gt_logic_coords = [((item[0], item[1]), (item[2], item[3]), "") for gt_cells in gt_tables for _, item in gt_cells]

        self.pred_table_structure = self.convert_html_by_structure(pred_logic_coords)
        self.gt_table_structure = self.convert_html_by_structure(gt_logic_coords)

        # self.pred_table_structure = self.generate_html_by_structure(pred_logic_coords)
        # self.gt_table_structure = self.generate_html_by_structure(gt_logic_coords)

    def evaluate(self, ignore_nodes=None):
        if not self.eval_teds:
            return self.calculate_bleu(self.gt_table_structure, self.pred_table_structure)

        try:
            return TEDS(structure_only=True, ignore_nodes=ignore_nodes)(self.pred_table_structure, self.gt_table_structure)
        except:  # gt 不存在表格：如果预测也不存在表格则返回1.0，否则返回0.0
            return float(self.pred_table_structure == self.gt_table_structure)

    @staticmethod
    def calculate_bleu(reference, candidate):
        """计算两个HTML序列之间的BLEU分数"""
        reference_tokens = [re.findall(r"<[^>]+>|[^<]+", reference)]
        candidate_tokens = re.findall(r"<[^>]+>|[^<]+", candidate)

        # 计算BLEU分数，使用4-gram
        smoothie = SmoothingFunction().method6
        bleu_score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothie)

        return bleu_score

    """ multitable
    def __init__(self, pred_tables, gt_tables):
        # 总共单元格数目
        self.num_pred_cells = sum([len(cells) for cells in pred_tables])
        self.num_gt_cells = sum([len(cells) for cells in gt_tables])

        # 获取排序后的 tables
        self.pred_tables, self.gt_tables, self.pred_table_bboxes, self.gt_table_bboxes = self._get_sorted_tables(pred_tables, gt_tables)

    def evaluate(self, ignore_nodes=None, table_iou_threshold=0.7):
        # 匹配表格列表
        table_pairs = self._match_tables(self.gt_table_bboxes, self.pred_table_bboxes, table_iou_threshold)

        # 初始化 TEDS
        teds = TEDS(structure_only=True, ignore_nodes=ignore_nodes)

        # 总分数（加权平均）
        p_score, r_sorce = 0.0, 0.0

        # 评估所有表格
        for gt_i, pred_i in table_pairs:
            gt_cells = self.gt_tables[gt_i]
            pred_cells = self.pred_tables[pred_i]

            pred_logic_coords = [((item[0], item[1]), (item[2], item[3]), "") for _, item in pred_cells]
            gt_logic_coords = [((item[0], item[1]), (item[2], item[3]), "") for _, item in gt_cells]

            pred_table_structure = self.convert_html_by_structure(pred_logic_coords)
            gt_table_structure = self.convert_html_by_structure(gt_logic_coords)

            score = teds(pred_table_structure, gt_table_structure)

            p_score += score * (len(pred_cells) / self.num_pred_cells if self.num_pred_cells > 0 else 0.0)
            r_sorce += score * (len(gt_cells) / self.num_gt_cells if self.num_gt_cells > 0 else 0.0)

        return 2 * (p_score * r_sorce) / (p_score + r_sorce) if p_score + r_sorce > 0 else 0.0
    """

    @staticmethod
    def _get_sorted_tables(pred_tables, gt_tables):
        # 计算对应表格bbox
        pred_table_bboxes = [TableStructEvaluator._get_table_bbox(table) for table in pred_tables]
        gt_table_bboxes = [TableStructEvaluator._get_table_bbox(table) for table in gt_tables]

        # 重新按照面积重大到小排序
        sorted_pred_bboxes = sorted(enumerate(pred_table_bboxes), key=lambda item: BoxEvaluator._calc_area(item[1]), reverse=True)
        pred_table_bboxes = [item[1] for item in sorted_pred_bboxes]
        pred_tables = [pred_tables[item[0]] for item in sorted_pred_bboxes]

        sorted_gt_bboxes = sorted(enumerate(gt_table_bboxes), key=lambda item: BoxEvaluator._calc_area(item[1]), reverse=True)
        gt_table_bboxes = [item[1] for item in sorted_gt_bboxes]
        gt_tables = [gt_tables[item[0]] for item in sorted_gt_bboxes]

        return pred_tables, gt_tables, pred_table_bboxes, gt_table_bboxes

    @staticmethod
    def _match_tables(gt_table_bboxes, pred_table_bboxes, table_iou_threshold=0.7):
        table_pairs = []

        # 计算交叠面积
        for i, gt_bbox in enumerate(gt_table_bboxes):
            gt_bbox_area = BoxEvaluator._calc_area(gt_bbox)
            for j, pred_bbox in enumerate(pred_table_bboxes):
                pred_bbox_area = BoxEvaluator._calc_area(pred_bbox)
                olp = BoxEvaluator._calc_overlap(gt_bbox, pred_bbox) / (min(gt_bbox_area, pred_bbox_area) + BaseEvaluator.EPS)
                if olp >= table_iou_threshold:
                    table_pairs.append((i, j))

        return table_pairs

    @staticmethod
    def _get_table_bbox(table):
        minx = min(min(pt[0] for pt in cell[0]) for cell in table)
        maxx = max(max(pt[0] for pt in cell[0]) for cell in table)
        miny = min(min(pt[1] for pt in cell[0]) for cell in table)
        maxy = max(max(pt[1] for pt in cell[0]) for cell in table)

        return [minx, miny, maxx, maxy]

    @staticmethod
    def convert_html_by_structure(data):
        if len(data) == 0:
            return "<html><body><table></table></body></html>"

        # 根据工作表和合并单元格范围构建 HTML 表格
        html = "<html><body>"

        rows = max(entry[1][1] for entry in data) + 1

        rows_cells = [[] for _ in range(rows)]
        for cell in data:
            start_row = cell[1][0]
            rows_cells[start_row].append(cell)
        for i in range(rows):
            rows_cells[i].sort(key=lambda cell: (cell[0][0], cell[0][1], cell[1][1]))

        html += "<table>"
        for row in range(rows):
            html += "<tr>"
            for colspan, rowspan, content in rows_cells[row]:
                colspan = colspan[1] - colspan[0] + 1
                rowspan = rowspan[1] - rowspan[0] + 1
                if colspan > 1 and rowspan > 1:
                    html += f"<td colspan='{colspan}' rowspan='{rowspan}'>{content}</td>"
                elif rowspan > 1:
                    html += f"<td rowspan='{rowspan}'>{content}</td>"
                elif colspan > 1:
                    html += f"<td colspan='{colspan}'>{content}</td>"
                else:
                    html += f"<td>{content}</td>"
            html += "</tr>"
        html += "</table>"

        html += "</body></html>"

        return html

    @staticmethod
    def generate_html_by_structure(data, preview=False, empty_content=""):
        """
        根据数据生成HTML表格

        Args:
            - data (list): 包含一个表格所有单元格数据，每个单元格是一个元组，包含行跨度、列跨度、内容和结构，例如：[[((start_col, end_col), (start_row, end_row), content)]]
            - preview (bool): 是否生成预览表格，默认为False
            - empty_content (str): 空单元格的内容，默认为空字符串

        Returns:
            - str: 生成的HTML表格字符串

        """
        if len(data) == 0:
            return "<html><body><table></table></body></html>"

        # 根据工作表和合并单元格范围构建 HTML 表格
        if preview:
            html = "<html><style> table { border-right: 1px solid #000000; border-bottom: 1px solid #000000; text-align: center; } table th { border-left: 1px solid #000000; border-top: 1px solid #000000; } table td { border-left: 1px solid #000000; border-top: 1px solid #000000; } </style><body>"
        else:
            html = "<html><body>"

        cols = max(entry[0][1] for entry in data) + 1
        rows = max(entry[1][1] for entry in data) + 1

        table = [[0 for _ in range(cols)] for _ in range(rows)]
        table_content = [[empty_content for _ in range(cols)] for _ in range(rows)]
        table_stucture = [[() for _ in range(cols)] for _ in range(rows)]

        # 遍历数据并填充单元格
        for colspan, rowspan, content in data:
            start_col = colspan[0]
            end_col = colspan[1]
            start_row = rowspan[0]
            end_row = rowspan[1]

            # 填充合并单元格
            try:
                for row in range(start_row, end_row + 1):
                    for col in range(start_col, end_col + 1):
                        table[row][col] = 1
                table[start_row][start_col] = 2
                table_content[start_row][start_col] = content
                table_stucture[start_row][start_col] = (end_row - start_row + 1, end_col - start_col + 1)
            except:
                print(start_row, end_row, start_col, end_col, rows, cols)

        # 合并空白
        for row in range(rows):
            start_col = None
            col_ranges = []
            for col in range(cols):
                if table[row][col] == 0:
                    if start_col is None:
                        start_col = col
                elif start_col is not None:
                    col_ranges.append((start_col, col))
                    start_col = None

            if start_col is not None:
                col_ranges.append((start_col, cols))

            for start_col, end_col in col_ranges:
                for col in range(start_col + 1, end_col):
                    table[row][col] = 1
                table[row][start_col] = 2
                table_stucture[row][start_col] = (1, end_col - start_col)

        html += "<table>"
        for row in range(rows):
            html += "<tr>"
            for col in range(cols):
                content = table_content[row][col]
                # 查找是否包含在合并范围内
                if table[row][col] > 0:
                    if table[row][col] == 2:
                        rowspan, colspan = table_stucture[row][col]

                        if colspan > 1 and rowspan > 1:
                            html += f"<td colspan='{colspan}' rowspan='{rowspan}'>{content}</td>"
                        elif rowspan > 1:
                            html += f"<td rowspan='{rowspan}'>{content}</td>"
                        elif colspan > 1:
                            html += f"<td colspan='{colspan}'>{content}</td>"
                        else:
                            html += f"<td>{content}</td>"
                else:
                    html += f"<td>{content}</td>"
            html += "</tr>"
        html += "</table>"

        html += "</body></html>"

        return html


class BoxEvaluator(BaseEvaluator):

    def __init__(self, pred_boxes, gt_boxes, data_format: Literal["xyxy", "xywh", "xy...", "nxy"] = "nxy", union_area=True):
        if data_format == "xyxy":
            self.pred_boxes = pred_boxes
            self.gt_boxes = gt_boxes
        else:
            self.pred_boxes = [self._normalized_box(box, data_format) for box in pred_boxes]
            self.gt_boxes = [self._normalized_box(box, data_format) for box in gt_boxes]

        self.union_area = union_area

        # print(f"Predict: {len(self.pred_boxes)}, GroundTruth: {len(self.gt_boxes)}")

        # 计算所有预测框和真实框的交并比
        self._calc_ious()

    def _calc_ious(self):
        self.ious = []

        pred_areas = [self._calc_area(box) for box in self.pred_boxes]
        gt_areas = [self._calc_area(box) for box in self.gt_boxes]

        for i, pred_box in enumerate(self.pred_boxes):
            ious = []
            for j, gt_box in enumerate(self.gt_boxes):
                inter_area = self._calc_overlap(pred_box, gt_box)
                if self.union_area:
                    ious.append(inter_area / (pred_areas[i] + gt_areas[j] - inter_area + self.EPS))
                else:
                    ious.append(inter_area / (min(pred_areas[i], gt_areas[j]) + self.EPS))
            self.ious.append(ious)

    @staticmethod
    def _calc_area(box):
        return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

    @staticmethod
    def _calc_overlap(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        return max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    def evaluate(self, iou_thresholds, save_match_indices=False):
        results = {}

        num_pred_boxes = len(self.pred_boxes)
        num_gt_boxes = len(self.gt_boxes)

        for threshold in iou_thresholds:
            true_positives = 0
            false_positives = 0
            false_negatives = num_gt_boxes
            matched_pred_indices = [None] * num_gt_boxes

            for i in range(num_pred_boxes):
                match_found = False
                for j in range(num_gt_boxes):
                    if matched_pred_indices[j] is not None:
                        continue

                    if self.ious[i][j] >= threshold:
                        matched_pred_indices[j] = i
                        match_found = True
                        break

                if match_found:
                    true_positives += 1
                    false_negatives -= 1
                else:
                    false_positives += 1

            precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0.0

            f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

            if save_match_indices:
                results[threshold] = {"P": precision, "R": recall, "F1": f1_score, "match_indices": matched_pred_indices}
            else:
                results[threshold] = {"P": precision, "R": recall, "F1": f1_score}

        return results

    @staticmethod
    def _normalized_box(box: list, origin_mode: Literal["xywh", "xy...", "nxy"] = "xywh"):
        if origin_mode == "xywh":
            return [box[0], box[1], box[0] + box[2], box[1] + box[3]]
        elif origin_mode == "xy...":
            minx = min(box[::2])
            miny = min(box[1::2])
            maxx = max(box[::2])
            maxy = max(box[1::2])
            return [minx, miny, maxx, maxy]
        elif origin_mode == "nxy":
            minx = min([pt[0] for pt in box])
            miny = min([pt[1] for pt in box])
            maxx = max([pt[0] for pt in box])
            maxy = max([pt[1] for pt in box])
            return [minx, miny, maxx, maxy]
        else:
            raise ValueError(f"Unknown origin_mode: {origin_mode}.")


class PolygonEvaluator(BoxEvaluator):

    def __init__(self, pred_polygons, gt_polygons, union_area=True):
        new_pred_polygons = []
        for pred_polygon in pred_polygons:
            try:
                new_pred_polygon = Polygon(pred_polygon)
            except:
                pass
            if new_pred_polygon.is_valid:
                new_pred_polygons.append(new_pred_polygon)

        new_gt_polygons = []
        for gt_polygon in gt_polygons:
            try:
                new_gt_polygon = Polygon(gt_polygon)
            except:
                pass
            if new_gt_polygon.is_valid:
                new_gt_polygons.append(new_gt_polygon)

        # num_inviad_pred = len(pred_polygons) - len(new_pred_polygons)
        # num_inviad_gt = len(gt_polygons) - len(new_gt_polygons)
        # if num_inviad_pred > 0:
        #     print(f"Number of inviad 'pred_polygon': {num_inviad_pred}/{len(pred_polygons)}")
        # if num_inviad_gt > 0:
        #     print(f"Number of inviad 'gt_polygon': {num_inviad_gt}/{len(gt_polygons)}")

        super().__init__(new_pred_polygons, new_gt_polygons, "xyxy", union_area)

    @staticmethod
    def _calc_area(polygon):
        return polygon.area

    @staticmethod
    def _calc_overlap(polygon1, polygon2):
        return polygon1.intersection(polygon2).area


class TableCoordsEvaluator(BaseEvaluator):

    def __init__(self, CellEvaluator, pred_cells, gt_cells, union_area=True):
        self.pred_cells = sorted(pred_cells, key=lambda cell: (cell[1][2], cell[1][0], cell[1][1], cell[1][3]))
        self.gt_cells = sorted(gt_cells, key=lambda cell: (cell[1][2], cell[1][0], cell[1][1], cell[1][3]))

        self.cell_evaluator = CellEvaluator([cell[0] for cell in self.pred_cells], [cell[0] for cell in self.gt_cells], union_area=union_area)

    def evaluate(self, iou_thresholds, is_detail=True):
        evaluate_results = self.cell_evaluator.evaluate(iou_thresholds, True)

        for threshold in iou_thresholds:
            match_indices = evaluate_results[threshold]["match_indices"]

            true_physical_coords = 0
            true_logical_coords = 0
            true_logical_coord_detail = [0, 0, 0, 0]

            #! test
            # true_logical_coord_cr = [0, 0]

            for i, j in enumerate(match_indices):
                if j is not None:
                    true_physical_coords += 1

                    if self.gt_cells[i][1] == self.pred_cells[j][1]:
                        true_logical_coords += 1

                    for k in range(4):
                        if self.gt_cells[i][1][k] == self.pred_cells[j][1][k]:
                            true_logical_coord_detail[k] += 1

                    #! test
                    # for k0, k1, k2 in [(0, 0, 1), (1, 2, 3)]:
                    #     if self.gt_cells[i][1][k1] == self.pred_cells[j][1][k1] and self.gt_cells[i][1][k2] == self.pred_cells[j][1][k2]:
                    #         true_logical_coord_cr[k0] += 1

            evaluate_results[threshold]["L_Acc"] = true_logical_coords / true_physical_coords if true_physical_coords > 0 else 0.0

            if is_detail:
                #! test
                # evaluate_results[threshold]["Lsc_Acc"] = true_logical_coord_cr[0] / true_physical_coords if true_physical_coords > 0 else 0.0
                # evaluate_results[threshold]["Lec_Acc"] = true_logical_coord_cr[1] / true_physical_coords if true_physical_coords > 0 else 0.0

                evaluate_results[threshold]["Lsc_Acc"] = true_logical_coord_detail[0] / true_physical_coords if true_physical_coords > 0 else 0.0
                evaluate_results[threshold]["Lec_Acc"] = true_logical_coord_detail[1] / true_physical_coords if true_physical_coords > 0 else 0.0
                evaluate_results[threshold]["Lsr_Acc"] = true_logical_coord_detail[2] / true_physical_coords if true_physical_coords > 0 else 0.0
                evaluate_results[threshold]["Ler_Acc"] = true_logical_coord_detail[3] / true_physical_coords if true_physical_coords > 0 else 0.0

            del evaluate_results[threshold]["match_indices"]

        return evaluate_results


# 物理坐标邻接关系
class PhyAdjRelationEvaluator(BaseEvaluator):
    NONE = 0
    TOP = 1
    LEFT = 2
    BOTTOM = -1
    RIGHT = -2

    def __init__(self, pred_cells, gt_cells):
        self.pred_cells = [cell[0] for cell in pred_cells]
        self.gt_cells = [cell[0] for cell in gt_cells]

        self.pred_quads = build_relation_map(self.pred_cells, 1, True)
        self.gt_quads = build_relation_map(self.gt_cells, 1, True)
        self.pred_quads.sort(key=lambda quad: quad.index)
        self.gt_quads.sort(key=lambda quad: quad.index)

        self.cell_evaluator = BoxEvaluator(self.pred_cells, self.gt_cells)

    def evaluate(self, iou_thresholds):
        evaluate_results = self.cell_evaluator.evaluate(iou_thresholds, True)

        # 初始化邻接关系信息
        num_pred_cells = len(self.pred_cells)
        num_gt_cells = len(self.gt_cells)

        pred_ar_pair = [[self.NONE] * num_pred_cells for _ in range(num_pred_cells)]
        gt_ar_pair = [[self.NONE] * num_gt_cells for _ in range(num_gt_cells)]
        pred_adj_relations, gt_adj_relations = 0, 0

        # 计算 pred_cells 的邻接关系
        for i in range(num_pred_cells):
            for j in range(i + 1, num_pred_cells):
                pred_ar_pair[i][j] = self.get_adj_relation(self.pred_quads[i], self.pred_quads[j])
                if pred_ar_pair[i][j] != self.NONE:
                    pred_ar_pair[j][i] = -pred_ar_pair[i][j]
                    pred_adj_relations += 1

        # 计算 gt_cells 的邻接关系
        for i in range(num_gt_cells):
            for j in range(i + 1, num_gt_cells):
                gt_ar_pair[i][j] = self.get_adj_relation(self.gt_quads[i], self.gt_quads[j])
                if gt_ar_pair[i][j] != self.NONE:
                    gt_adj_relations += 1

        # 开始计算
        for threshold in iou_thresholds:
            match_indices = evaluate_results[threshold]["match_indices"]

            # 统计正确邻接关系对数
            true_adj_relations = 0
            for gt_i, pred_i in enumerate(match_indices):
                if pred_i is None:
                    continue
                for gt_j in range(gt_i + 1, num_gt_cells):
                    pred_j = match_indices[gt_j]
                    if pred_j is None:
                        continue
                    if gt_ar_pair[gt_i][gt_j] != self.NONE and pred_ar_pair[pred_i][pred_j] == gt_ar_pair[gt_i][gt_j]:
                        true_adj_relations += 1

            precision = true_adj_relations / pred_adj_relations if pred_adj_relations > 0 else 0.0
            recall = true_adj_relations / gt_adj_relations if gt_adj_relations > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

            evaluate_results[threshold] = {"AR_P": precision, "AR_R": recall, "AR_F1": f1}

        return evaluate_results

    @staticmethod
    def get_adj_relation(quad1, quad2):
        for quad in [*quad1.forward.top, *quad1.backward.top]:
            if quad.index == quad2.index:
                return PhyAdjRelationEvaluator.TOP
        for quad in [*quad1.forward.left, *quad1.backward.left]:
            if quad.index == quad2.index:
                return PhyAdjRelationEvaluator.LEFT
        for quad in [*quad1.forward.bottom, *quad1.backward.bottom]:
            if quad.index == quad2.index:
                return PhyAdjRelationEvaluator.BOTTOM
        for quad in [*quad1.forward.right, *quad1.backward.right]:
            if quad.index == quad2.index:
                return PhyAdjRelationEvaluator.RIGHT
        return PhyAdjRelationEvaluator.NONE


class ScitsrRelationEvaluator(BaseEvaluator):
    """table"""

    def __init__(self, pred_cells, gt_cells):
        self.gt_relations = self.get_scitsr_relations([cell[1] for cell in pred_cells])
        self.pred_relations = self.get_scitsr_relations([cell[1] for cell in gt_cells])

    def evaluate(self, cmp_blank=True):
        correct_count = compare_rel(self.gt_relations, self.pred_relations, cmp_blank)

        num_pred_relations = len(self.pred_relations)
        num_gt_relations = len(self.gt_relations)

        precision = correct_count / num_pred_relations if num_pred_relations > 0 else 0
        recall = correct_count / num_gt_relations if num_gt_relations > 0 else 0
        f1_scorce = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        return {"P": precision, "R": recall, "F1": f1_scorce}

    """multiable
    def __init__(self, pred_tables, gt_tables):
        # 获取排序后的 tables
        self.pred_tables, self.gt_tables, self.pred_table_bboxes, self.gt_table_bboxes = TableStructEvaluator._get_sorted_tables(pred_tables, gt_tables)

    def evaluate(self, cmp_blank=True, table_iou_threshold=0.7):
        # 匹配表格列表
        table_pairs = TableStructEvaluator._match_tables(self.gt_table_bboxes, self.pred_table_bboxes, table_iou_threshold)
        not_match_gts = set(range(len(self.gt_table_bboxes))) - set([pair[0] for pair in table_pairs])
        not_match_preds = set(range(len(self.pred_table_bboxes))) - set([pair[1] for pair in table_pairs])

        # 总计数
        correct_count, num_pred_relations, num_gt_relations = 0, 0, 0

        # 评估所有匹配表格
        for gt_i, pred_i in table_pairs:
            gt_cells = self.gt_tables[gt_i]
            pred_cells = self.pred_tables[pred_i]

            pred_relations = self.get_scitsr_relations([cell[1] for cell in pred_cells])
            gt_relations = self.get_scitsr_relations([cell[1] for cell in gt_cells])

            correct_count += compare_rel(gt_relations, pred_relations, cmp_blank)
            num_pred_relations += len(pred_relations)
            num_gt_relations += len(gt_relations)

        # 评估未匹配表格
        for gt_i in not_match_gts:
            gt_cells = self.gt_tables[gt_i]
            num_gt_relations += len(self.get_scitsr_relations([cell[1] for cell in gt_cells]))
        for pred_i in not_match_preds:
            pred_cells = self.pred_tables[pred_i]
            num_pred_relations += len(self.get_scitsr_relations([cell[1] for cell in pred_cells]))

        precision = correct_count / num_pred_relations if num_pred_relations > 0 else 0
        recall = correct_count / num_gt_relations if num_gt_relations > 0 else 0
        f1_scorce = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        return {"P": precision, "R": recall, "F1": f1_scorce}
    """

    @staticmethod
    def get_scitsr_relations(cells):
        result = {"cells": []}
        _cells = result["cells"]
        for id, cell in enumerate(cells):
            if cell[0] > cell[1] or cell[2] > cell[3]:
                continue
            _cells.append({"id": id, "content": "empty", "start_col": cell[0], "end_col": cell[1], "start_row": cell[2], "end_row": cell[3]})
        return json2Relations(result, False)


# 逻辑坐标邻接关系
# source: ICDAR2019 evaluate
class LogAdjRelationEvaluator(BaseEvaluator):
    def __init__(self, pred_tables, gt_tables):
        # 计算预测表格结构
        self.pred_structure = [{"cells": [{"coords": cell[0], "structure": cell[1]} for cell in cells]} for cells in pred_tables]
        # 计算标准表格结构
        self.gt_structure = [{"cells": [{"coords": cell[0], "structure": cell[1]} for cell in cells]} for cells in gt_tables]

        # 构建单元格邻接关系评估器
        self.carte_cell_adj_relation_evaluator = cell_adj_relation.Carte(self.pred_structure, self.gt_structure)
        self.cell_adj_relation_evaluator = cell_adj_relation.Evaluate(self.pred_structure, self.gt_structure)

    def evaluate(self, iou_thresholds, table_iou_threshold=0.8):
        # return [None for _ in range(len(iou_thresholds))]
        return self.cell_adj_relation_evaluator.evaluate(iou_thresholds, table_iou_threshold)

    def evaluate_carte(self):
        return self.carte_cell_adj_relation_evaluator.evaluate()

    @staticmethod
    def average(evaluate_results, is_carte=False):
        # if is_carte:
        #     return {"P": 1.0, "R": 1.0, "F1": 1.0}
        # return [{"P": 1.0, "R": 1.0, "F1": 1.0} for _ in range(len(evaluate_results[0]))]
        if is_carte:
            return cell_adj_relation.Carte.compute_metric(evaluate_results)

        return cell_adj_relation.Evaluate.compute_metrics(evaluate_results)

    @staticmethod
    def split_tables(cells, adjacent_thresh):
        tables_indexes = split_unicom_area([cell[0] for cell in cells], adjacent_thresh)

        tables = []
        for table_indexes in tables_indexes:
            tables.append([cells[index] for index in table_indexes])

        return tables

    @staticmethod
    def split_tables_by_id(cells):
        """
        根据和标准表格的匹配度分割表格区域

        Args:
            - cells: 标准单元格列表，结构：[[cell..., table_id], ...]

        Returns:
            - tables: 预测表格列表
        """
        if len(cells) == 0:
            return [], False

        # 分离标准表格
        tables = {}
        for cell in cells:
            table_id = cell[-1]
            if table_id not in tables:
                tables[table_id] = []

            tables[table_id].append(cell[:-1])

        return list(tables.values()), cells[0][-1] is not None

    @staticmethod
    def split_tables_by_iou(pred_cells, gt_cells, iou_threshold):
        """
        根据和标准表格的匹配度分割表格区域

        Args:
            - pred_cells: 预测单元格列表，结构：[pred_cell, ...]， pred_cell=>[Physical coordinates, ...]
            - gt_cells: 标准单元格列表，结构：[[gt_cell..., table_id], ...]， gt_cell=>[Physical coordinates, ...]
            - iou_threshold: 匹配度阈值

        Returns:
            - pred_tables: 预测表格列表
            - gt_tables: 标准表格列表
        """
        # 分离标准表格
        gt_tables = {}
        for gt_cell in gt_cells:
            table_id = gt_cell[-1]
            if table_id not in gt_tables:
                gt_tables[table_id] = []

            gt_tables[table_id].append(gt_cell[:-1])

        # 计算iou
        ious = BoxEvaluator([cell[0] for cell in pred_cells], [cell[0] for cell in gt_cells]).ious

        # 分离预测表格
        pred_tables = {}
        unmatched_cells = []
        for i, pred_cell in enumerate(pred_cells):
            match_tables = []
            for j, (table_id, _) in enumerate(gt_cells):
                iou = ious[i][j]
                if iou >= iou_threshold:
                    match_tables.append((table_id, iou))

            match_tables.sort(key=lambda x: x[1], reverse=True)

            if len(match_tables) > 0:
                table_id = match_tables[0][0]

                if table_id not in pred_tables:
                    pred_tables[table_id] = []

                pred_tables[table_id].append(pred_cell)
            else:
                unmatched_cells.append([pred_cell])

        return [*pred_tables.values(), *unmatched_cells], list(gt_tables.values())
