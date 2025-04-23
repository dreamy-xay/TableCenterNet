#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description: 
Version: 
Autor: dreamy-xay
Date: 2024-12-06 15:38:09
LastEditors: dreamy-xay
LastEditTime: 2024-12-07 15:29:33
"""
from shapely.geometry import Polygon
import xml.dom.minidom
import numpy as np


def compute_poly_iou(poly1, poly2):
    if poly1 is None or poly2 is None:
        return 0

    try:
        intersection = poly1.intersection(poly2)
        iou = intersection.area / (poly1.area + poly2.area - intersection.area)
    except:
        iou = 0

    return iou


def compute_poly_overlap(poly1, poly2):
    if poly1 is None or poly2 is None:
        return 0

    try:
        overlap = poly1.intersection(poly2).area / poly1.area
    except:
        overlap = 0

    return overlap


def compute_poly_max_overlap(poly1, poly2):
    if poly1 is None or poly2 is None:
        return 0

    try:
        overlap = poly1.intersection(poly2).area / min(poly1.area, poly2.area)
    except:
        overlap = 0

    return overlap


class Cell(object):
    # @:param start_row : start row index of the Cell
    # @:param start_col : start column index of the Cell
    # @:param end-row : end row index of the Cell
    # @:param end-col : end column index of the Cell
    # @:param cell_box: bounding-box of the Cell (coordinates are saved as a string)
    # @:param content_box: bounding-box of the text content within Cell (unused variable)
    # @:param cell_id: unique id of the Cell

    def __init__(self, table_id, start_row, start_col, cell_box, end_row, end_col, content_box="", is_parse_cell_boox=False):
        self.start_row = int(start_row)
        self.start_col = int(start_col)
        self.cell_box = cell_box
        self.content_box = content_box
        self.table_id = table_id  # the table_id this cell belongs to
        # self._cell_name = cell_id    # specify the cell using passed-in cell_id
        self.cell_id = id(self)
        # self._region = region

        # check for end-row and end-col special case
        if end_row == -1:
            self.end_row = self.start_row
        else:
            self.end_row = int(end_row)
        if end_col == -1:
            self.end_col = self.start_col
        else:
            self.end_col = int(end_col)

        if is_parse_cell_boox:
            self._bbox, self._cell_box = self.parse_bbox()
            self._cell_box = np.array(self._cell_box, dtype=int)
        else:
            self._bbox = self.parse_bbox()[0]

    def __str__(self):
        return "CELL row=[%d, %d] col=[%d, %d] (coords=%s)" % (self.start_row, self.end_row, self.start_col, self.end_col, self.cell_box)

    def parse_bbox(self):
        cell_box = []
        for pt_str in self.cell_box.split():
            x, y = pt_str.split(",")
            cell_box.append([int(x), int(y)])

        try:
            _cell_box = Polygon(cell_box)
        except:
            _cell_box = None

        return _cell_box, cell_box

    # return the IoU value of two cell blocks
    def compute_cell_iou(self, another_cell):
        return compute_poly_iou(self._bbox, another_cell._bbox)

    # check if the two cell object denotes same cell area in table
    def check_same(self, another_cell):
        return self.start_row == another_cell.start_row and self.end_row == another_cell.end_row and self.start_col == another_cell.start_col and self.end_col == another_cell.end_col


# Note: currently save the relation with two cell object involved,
# can be replaced by cell_id in follow-up memory clean up
class AdjRelation:

    DIR_HORIZ = 1
    DIR_VERT = 2

    def __init__(self, fromText, toText, direction):
        # @param: fromText, toText are Cell objects （may be changed to cell-ID for further development）
        self.fromText = fromText
        self.toText = toText
        self.direction = direction

    def __str__(self):
        if self.direction == self.DIR_VERT:
            dir = "vertical"
        else:
            dir = "horizontal"
        return "ADJ_RELATION: " + str(self.fromText) + "  " + str(self.toText) + "    " + dir

    def isEqual(self, otherRelation):
        return self.fromText.cell_id == otherRelation.fromText.cell_id and self.toText.cell_id == otherRelation.toText.cell_id and self.direction == otherRelation.direction


class Table:

    def __init__(self, tableNode, is_parse_empty_table=False):
        self._root = tableNode
        self._id = id(self)
        self._table_coords = ""
        self._maxRow = 0  # PS: indexing from 0
        self._maxCol = 0
        self._cells = []  # save a table as list of <Cell>s
        self.adj_relations = []  # save the adj_relations for the table
        self.parsed = False
        self.found = False  # check if the find_adj_relations() has been called once

        if is_parse_empty_table:
            self.parse_empty_table()
        else:
            self.parse_table()

        self._bbox = self.parse_bbox()

    def __str__(self):
        return "TABLE object - {} row x {} col".format(self._maxRow + 1, self._maxCol + 1)

    @property
    def id(self):
        return self._id

    @property
    def table_coords(self):
        return self._table_coords

    @property
    def table_cells(self):
        return self._cells

    # parse input xml to cell lists
    def parse_table(self):
        # get the table bbox
        self._table_coords = str(self._root.getElementsByTagName("Coords")[0].getAttribute("points"))

        # get info for each cell
        cells = self._root.getElementsByTagName("cell")
        max_row = max_col = 0
        for cell in cells:
            sr = cell.getAttribute("start-row")
            sc = cell.getAttribute("start-col")
            b_points = str(cell.getElementsByTagName("Coords")[0].getAttribute("points"))
            er = cell.getAttribute("end-row") if cell.hasAttribute("end-row") else -1
            ec = cell.getAttribute("end-col") if cell.hasAttribute("end-col") else -1
            new_cell = Cell(table_id=str(self.id), start_row=sr, start_col=sc, cell_box=b_points, end_row=er, end_col=ec)
            max_row = max(max_row, int(sr), int(er))
            max_col = max(max_col, int(sc), int(ec))
            self._cells.append(new_cell)
        self._maxCol = max_col
        self._maxRow = max_row

        self.parsed = True

    def parse_empty_table(self):
        self._empty_cells = []  # save a table as list of <Cell>s

        MAX_CELL_NUM = 1000
        cell_flag = [[False for _ in range(MAX_CELL_NUM)] for _ in range(MAX_CELL_NUM)]
        row_min = [9999] * MAX_CELL_NUM
        row_max = [-9999] * MAX_CELL_NUM
        col_min = [9999] * MAX_CELL_NUM
        col_max = [-9999] * MAX_CELL_NUM

        # get the table bbox
        self._table_coords = str(self._root.getElementsByTagName("Coords")[0].getAttribute("points"))

        # get info for each cell
        cells = self._root.getElementsByTagName("cell")
        max_row = max_col = 0
        min_row = min_col = 9999
        for cell in cells:
            sr = cell.getAttribute("start-row")
            sc = cell.getAttribute("start-col")
            b_points = str(cell.getElementsByTagName("Coords")[0].getAttribute("points"))
            er = cell.getAttribute("end-row") if cell.hasAttribute("end-row") else -1
            ec = cell.getAttribute("end-col") if cell.hasAttribute("end-col") else -1
            new_cell = Cell(table_id=str(self.id), start_row=sr, start_col=sc, cell_box=b_points, end_row=er, end_col=ec, is_parse_cell_boox=True)

            sc, ec, sr, er = int(sc), int(ec), int(sr), int(er)
            col_min[sc] = min(col_min[sc], new_cell._cell_box[:, 0].min())
            col_max[ec] = max(col_max[ec], new_cell._cell_box[:, 0].max())
            row_min[sr] = min(row_min[sr], new_cell._cell_box[:, 1].min())
            row_max[er] = max(row_max[er], new_cell._cell_box[:, 1].max())

            for r in range(sr, er + 1):
                for c in range(sc, ec + 1):
                    cell_flag[r][c] = True

            max_row = max(max_row, sr, er)
            max_col = max(max_col, sc, ec)
            min_row = min(min_row, sr, er)
            min_col = min(min_col, sc, ec)

            self._cells.append(new_cell)

        self._maxCol = max_col
        self._maxRow = max_row

        # calc cell boundaries for empty cells
        while True:  # row-wise traverse
            changed = False
            for r in range(min_row, max_row + 1):
                if r != min_row:
                    if abs(row_min[r]) == 9999:
                        row_min[r] = row_max[r - 1]
                        if abs(row_min[r]) != 9999:
                            changed = True
                if r != max_row:
                    if abs(row_max[r]) == 9999:
                        row_max[r] = row_min[r + 1]
                        if abs(row_max[r]) != 9999:
                            changed = True
            if not changed:
                break
        while True:  # col-wise traverse
            changed = False
            for c in range(min_col, max_col + 1):
                if c != min_col:
                    if abs(col_min[c]) == 9999:
                        col_min[c] = col_max[c - 1]
                        if abs(col_min[c]) != 9999:
                            changed = True
                if c != max_col:
                    if abs(col_max[c]) == 9999:
                        col_max[c] = col_min[c + 1]
                        if abs(col_max[c]) != 9999:
                            changed = True
            if not changed:
                break

        # find empty cells
        for r in range(min_row, max_row + 1):
            for c in range(min_col, max_col + 1):
                if cell_flag[r][c]:
                    continue

                x1, y1, x2, y2 = col_min[c], row_min[r], col_max[c], row_max[r]

                if 9999 in [abs(x1), abs(x2), abs(y1), abs(y2)]:
                    # print(f"cannot make empty cell... (r={r}, c={c}), ({x1},{y1},{x2},{y2})")
                    continue

                self._empty_cells.append(
                    Cell(
                        table_id=str(self.id),
                        start_row=str(r),
                        start_col=str(c),
                        cell_box=f"{x1},{y1} {x2},{y1} {x2},{y2} {x1},{y2}",
                        end_row=str(r),
                        end_col=str(c),
                        is_parse_cell_boox=True,
                    )
                )

        self.parsed = True

    def parse_bbox(self):
        table_box = []
        for pt_str in self.table_coords.split():
            x, y = pt_str.split(",")
            table_box.append([int(x), int(y)])

        try:
            table_box = Polygon(table_box)
        except:
            table_box = None

        return table_box

    # generate a table-like structure for finding adj_relations
    def convert_2d(self):
        table = [[0 for x in range(self._maxCol + 1)] for y in range(self._maxRow + 1)]  # init blank cell with int 0
        for cell in self._cells:
            cur_row = cell.start_row
            while cur_row <= cell.end_row:
                cur_col = cell.start_col
                while cur_col <= cell.end_col:
                    temp = table[cur_row][cur_col]
                    if temp == 0:
                        table[cur_row][cur_col] = [cell]
                    else:
                        temp.append(cell)
                        table[cur_row][cur_col] = temp
                    cur_col += 1
                cur_row += 1

        return table

    def find_adj_relations(self):
        if self.found:
            return self.adj_relations
        else:
            # if len(self._cells) == 0:
            if self.parsed == False:
                # fix: cases where there's no cell in table?
                print("table is not parsed for further steps.")
                self.parse_table()
                self.find_adj_relations()
            else:
                retVal_h = []
                retVal_v = []
                tab = self.convert_2d()

                # find horizontal relations
                for r in range(self._maxRow + 1):
                    for c_from in range(self._maxCol):
                        temp_pos = tab[r][c_from]
                        if temp_pos == 0:
                            continue
                        else:
                            for cell in temp_pos:
                                c_to = c_from + 1
                                if tab[r][c_to] != 0:
                                    # find relation between two adjacent cells
                                    for cell_to in tab[r][c_to]:
                                        if cell != cell_to and (not cell.check_same(cell_to)):
                                            adj_relation = AdjRelation(cell, cell_to, AdjRelation.DIR_HORIZ)
                                            retVal_h.append(adj_relation)
                                else:
                                    # find the next non-blank cell, if exists
                                    for temp in range(c_from + 1, self._maxCol + 1):
                                        if tab[r][temp] != 0:
                                            for cell_to in tab[r][temp]:
                                                adj_relation = AdjRelation(cell, cell_to, AdjRelation.DIR_HORIZ)
                                                retVal_h.append(adj_relation)
                                            break

                # find vertical relations
                for c in range(self._maxCol + 1):
                    for r_from in range(self._maxRow):
                        temp_pos = tab[r_from][c]
                        if temp_pos == 0:
                            continue
                        else:
                            for cell in temp_pos:
                                r_to = r_from + 1
                                if tab[r_to][c] != 0:
                                    # find relation between two adjacent cells
                                    for cell_to in tab[r_to][c]:
                                        if cell != cell_to and (not cell.check_same(cell_to)):
                                            adj_relation = AdjRelation(cell, cell_to, AdjRelation.DIR_VERT)
                                            retVal_v.append(adj_relation)
                                else:
                                    # find the next non-blank cell, if exists
                                    for temp in range(r_from + 1, self._maxRow + 1):
                                        if tab[temp][c] != 0:
                                            for cell_to in tab[temp][c]:
                                                adj_relation = AdjRelation(cell, cell_to, AdjRelation.DIR_VERT)
                                                retVal_v.append(adj_relation)
                                            break

                new_retVal = []
                st_h = set()
                for val in retVal_h:
                    key = f"{val.fromText.cell_id}, {val.toText.cell_id}"
                    if key not in st_h:
                        new_retVal.append(val)
                        st_h.add(key)

                st_v = set()
                for val in retVal_v:
                    key = f"{val.fromText.cell_id}, {val.toText.cell_id}"
                    if key not in st_v:
                        new_retVal.append(val)
                        st_v.add(key)

                self.found = True
                self.adj_relations = new_retVal
            return self.adj_relations

    # compute the IOU of table, pass-in var is another Table object
    def compute_table_iou(self, another_table):
        return compute_poly_iou(self._bbox, another_table._bbox)

    def compute_table_olp(self, another_table):
        return compute_poly_max_overlap(self._bbox, another_table._bbox)

    def bind_cell_ious(self, target_table):
        self._bind_cell_ios = []
        for cell_1 in self.table_cells:
            ious = []
            for cell_2 in target_table.table_cells:
                ious.append((compute_poly_iou(cell_1._bbox, cell_2._bbox), cell_2))
            self._bind_cell_ios.append(ious)

    def find_bind_cell_mapping(self, iou_value):
        mapped_cell = []  # store the matches as tuples - (gt, result) mind the order of table when passing in
        for i, cell_1 in enumerate(self.table_cells):
            for j in range(len(self._bind_cell_ios[i])):
                cur_iou_value, cell_2 = self._bind_cell_ios[i][j]
                if cur_iou_value >= iou_value:
                    mapped_cell.append((cell_1, cell_2))
                    break
        ret = dict(mapped_cell)
        return ret

    # find the cell mapping of tables as dictionary, pass-in var is another table and the desired IOU value
    def find_cell_mapping(self, target_table, iou_value):
        mapped_cell = []  # store the matches as tuples - (gt, result) mind the order of table when passing in
        for cell_1 in self.table_cells:
            for cell_2 in target_table.table_cells:
                if cell_1.compute_cell_iou(cell_2) >= iou_value:
                    mapped_cell.append((cell_1, cell_2))
                    break
        ret = dict(mapped_cell)
        return ret


class ResultStructure:
    def __init__(self, truePos, gtTotal, resTotal):
        self.truePos = truePos
        self.gtTotal = gtTotal
        self.resTotal = resTotal

    @property
    def P(self):
        return self.truePos / (self.resTotal + 1e-6)

    @property
    def R(self):
        return self.truePos / (self.gtTotal + 1e-6)

    @property
    def F1(self):
        P = self.P
        R = self.R
        return 2 * P * R / (P + R + 1e-6)

    def __str__(self):
        return "true: {}, gt: {}, res: {}".format(self.truePos, self.gtTotal, self.resTotal)


class CellAdj:
    def __init__(self, cell):
        self._bbox = cell._bbox
        self._cell_box = cell._cell_box
        self._minx = min(self._cell_box[:, 0])
        self._miny = min(self._cell_box[:, 1])
        self._maxx = max(self._cell_box[:, 0])
        self._maxy = max(self._cell_box[:, 1])
        self._table_id = cell.table_id  # the table_id this cell belongs to
        self._cell_id = cell.cell_id
        self._rows = (cell.start_row, cell.end_row)
        self._cols = (cell.start_col, cell.end_col)

        self._num_hor_adj = 0
        self._num_ver_adj = 0

    def __str__(self):
        return "CELL Adjency object - {} hor, {} ver".format(self._num_hor_adj, self._num_ver_adj)


class AllTables:
    def __init__(self, tableList):
        self._celladj = list()
        self._empty_celladj = list()
        self._tableList = tableList

        # cell matching
        self._matched = False
        self._match_idx = list()

        self.parse_cell_adj()
        self.parse_empty_cell_adj()

    def parse_cell_adj(self):
        celladj_dict = dict()

        for table in self._tableList:
            for adj in table.find_adj_relations():
                # retrieve adjency data
                fromCell = adj.fromText

                # retrieve celladj
                cell_id = fromCell.cell_id
                celladj = celladj_dict.get(cell_id, None)
                if celladj is None:
                    celladj = CellAdj(fromCell)

                # increase adj num
                if adj.direction == AdjRelation.DIR_HORIZ:  # "horizontal":
                    celladj._num_hor_adj += 1
                elif adj.direction == AdjRelation.DIR_VERT:  # "vertical":
                    celladj._num_ver_adj += 1

                # replace
                celladj_dict[cell_id] = celladj

            # add cell without adj for matching candidate
            for cell in table._cells:
                if not cell.cell_id in celladj_dict:
                    celladj_dict[cell.cell_id] = CellAdj(cell)

        # dict() to list()
        for key, value in celladj_dict.items():
            self._celladj.append(value)

    def parse_empty_cell_adj(self):
        for table in self._tableList:
            for c in table._empty_cells:
                self._empty_celladj.append(CellAdj(c))

    def match_cells(self, A, B):
        if self._matched:
            return
        self._matched = True

        # find area recall & area precision
        matAR = self.calculate_overlap(A, B)
        matAP = self.calculate_overlap(B, A).transpose()

        # matching matrix
        match_idx = [[] for _ in range(len(A))]
        matched_flag = [False] * len(B)

        if len(A) == 0 or len(B) == 0:
            self._match_idx = match_idx
            return

        # step1) match based on AR
        max_AR = np.argmax(matAR, axis=1)
        for m in range(len(A)):
            n = max_AR[m]
            if matAR[m, n] != 0:
                match_idx[m].append(n)
                matched_flag[n] = True

        # step2) match based on AP
        indSorted = np.dstack(np.unravel_index(np.argsort(-matAP, axis=None), matAP.shape))[0]
        for m, n in indSorted:
            # no more overlap
            if matAP[m, n] == 0:
                break
            if matched_flag[n]:
                continue

            match_idx[m].append(n)
            matched_flag[n] = True

        self._match_idx = match_idx

    def match_empty_cells(self, B):
        A = self._empty_celladj
        # find area recall & area precision
        matAR = self.calculate_overlap(A, B)

        # matching matrix
        match_idx = [[] for _ in range(len(A))]

        if len(A) == 0 or len(B) == 0:
            self._match_empty_idx = match_idx
            return

        # step1) match based on AR
        max_AR = np.argmax(matAR, axis=1)
        for m in range(len(A)):
            n = max_AR[m]
            if matAR[m, n] != 0:
                match_idx[m].append(n)

        self._match_empty_idx = match_idx

    def calculate_overlap(self, A, B):
        len_A = len(A)
        len_B = len(B)
        mat_overlap = np.zeros((len_A, len_B), dtype=np.float32)

        for m in range(len_A):
            for n in range(len_B):
                a, b = A[m], B[n]
                # minimum requiremnt
                if a._minx > b._maxx or a._miny > b._maxy or a._maxx < b._minx or a._maxy < b._miny:
                    continue
                mat_overlap[m, n] = compute_poly_overlap(a._bbox, b._bbox)

        return mat_overlap

    def calculate_score(self, res_tables):
        # merge cells with the same content
        GTAdj, PredAdj = self._celladj, res_tables._celladj

        # Match cells
        self.match_cells(GTAdj, PredAdj)
        self.match_empty_cells(PredAdj)

        # 1) Total GT, Pred
        tot_gt, tot_pred = 0, 0
        for adj in GTAdj:
            tot_gt += adj._num_hor_adj + adj._num_ver_adj
        for adj in PredAdj:
            tot_pred += adj._num_hor_adj + adj._num_ver_adj

        # 2) Match GT
        tot_tp, tot_dc = 0, 0
        cor_hor_adj, cor_ver_adj = 0, 0
        match_flag = [False] * len(PredAdj)
        for m in range(len(GTAdj)):
            num_gt_hor_adj = GTAdj[m]._num_hor_adj
            num_gt_ver_adj = GTAdj[m]._num_ver_adj

            for n in self._match_idx[m]:
                match_flag[n] = True
                cur_cor_hor_adj = min(PredAdj[n]._num_hor_adj, num_gt_hor_adj)
                num_gt_hor_adj -= cur_cor_hor_adj
                PredAdj[n]._num_hor_adj -= cur_cor_hor_adj
                cor_hor_adj += cur_cor_hor_adj

                cur_cor_ver_adj = min(PredAdj[n]._num_ver_adj, num_gt_ver_adj)
                num_gt_ver_adj -= cur_cor_ver_adj
                PredAdj[n]._num_ver_adj -= cur_cor_ver_adj
                cor_ver_adj += cur_cor_ver_adj

        # 3) Match DC GT
        for m in range(len(self._empty_celladj)):
            for n in self._match_empty_idx[m]:
                if not match_flag[n]:
                    num_adj = PredAdj[n]._num_hor_adj + PredAdj[n]._num_ver_adj
                    tot_dc += num_adj
                    tot_pred -= num_adj

        tot_tp = cor_hor_adj + cor_ver_adj
        tot_fp = tot_pred - tot_tp
        tot_fn = tot_gt - tot_tp

        return {
            "gt": tot_gt,
            "pred": tot_pred,
            "tp": tot_tp,
            "fp": tot_fp,
            "fn": tot_fn,
            "dc": tot_dc,
        }


class Evaluate:
    def __init__(self, pred_structure, gt_structure):
        self.pred_structure = pred_structure
        self.gt_structure = gt_structure

    def evaluate(self, ious=[0.6, 0.7, 0.8, 0.9], table_iou_value=0.1):
        gt_dom = self.generate_table_xml(self.gt_structure)
        pred_dom = self.generate_table_xml(self.pred_structure)

        # parse the tables in input elements
        gt_tables = Evaluate.get_table_list(gt_dom)
        pred_tables = Evaluate.get_table_list(pred_dom)

        # sort
        gt_tables.sort(key=lambda table: -1 if table._bbox is None else table._bbox.area, reverse=True)
        pred_tables.sort(key=lambda table: -1 if table._bbox is None else table._bbox.area, reverse=True)

        # duplicate result table list
        pred_remaining = pred_tables.copy()
        gt_remaining = gt_tables.copy()

        # map the tables in gt and result file
        table_matches = []  # @param: table_matches - list of mapping of tables in gt and res file, in order (gt, res)
        for gtt in gt_remaining:
            for rest in pred_remaining:
                # note: for structural analysis, use 0.8 for table mapping
                if gtt.compute_table_iou(rest) >= table_iou_value:
                    # if gtt.compute_table_olp(rest) >= table_iou_value:
                    table_matches.append((gtt, rest))
                    pred_remaining.remove(rest)  # unsafe... should be ok with the break below
                    gt_remaining.remove(gtt)
                    break

        # if len(table_matches) == 0:
        #     print(len(pred_remaining), len(gt_remaining))

        total_gt_relation, total_pred_relation = 0, 0

        new_table_matches = []
        for gt_table, pred_table in table_matches:
            gt_table.bind_cell_ious(pred_table)
            # set up the adj relations, convert the one for result table to a dictionary for faster searching
            gt_AR = gt_table.find_adj_relations()
            total_gt_relation += len(gt_AR)

            res_AR = pred_table.find_adj_relations()
            total_pred_relation += len(res_AR)
            new_table_matches.append(((gt_table, gt_AR), (pred_table, res_AR)))

        # handle gt_relations in unmatched gt table
        for gtt_remain in gt_remaining:
            total_gt_relation += len(gtt_remain.find_adj_relations())

        # handle gt_relation in unmatched res table
        for pred_remain in pred_remaining:
            total_pred_relation += len(pred_remain.find_adj_relations())

        evaluate_result_list = []
        for iou in ious:
            total_correct_relation = self.evaluate_result(new_table_matches, iou)
            result = ResultStructure(truePos=total_correct_relation, gtTotal=total_gt_relation, resTotal=total_pred_relation)
            evaluate_result_list.append(result)

        return evaluate_result_list

    @staticmethod
    def get_table_list(dom, is_parse_empty_table=False):
        """
        return a list of Table objects corresponding to the table element of the DOM.
        """
        return [Table(_nd, is_parse_empty_table) for _nd in dom.documentElement.getElementsByTagName("table")]

    @staticmethod
    def evaluate_result(table_matches, iou_value):
        total_correct_relation = 0
        for (gt_table, gt_AR), (res_table, res_AR) in table_matches:
            # set up the cell mapping for matching tables
            # cell_mapping = gt_table.find_cell_mapping(res_table, iou_value)
            cell_mapping = gt_table.find_bind_cell_mapping(iou_value)

            # Now map GT adjacency relations to result
            lMappedAR = []
            for ar in gt_AR:
                try:
                    resFromCell = cell_mapping[ar.fromText]
                    resToCell = cell_mapping[ar.toText]
                    # make a mapped adjacency relation
                    lMappedAR.append(AdjRelation(resFromCell, resToCell, ar.direction))
                except:
                    # no mapping is possible
                    pass

            # compare two list of adjacency relation
            correct_dect = 0
            for ar1 in res_AR:
                for ar2 in lMappedAR:
                    if ar1.isEqual(ar2):
                        correct_dect += 1
                        break

            total_correct_relation += correct_dect

        return total_correct_relation

    @staticmethod
    def generate_table_xml(tables):
        """
        生成表格结构xml文件

        Args:
            - tables: list of dict, 每个dict包含表格的坐标和单元格信息 {"cells": [{"coords": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], "structure": [start_col, end_col, start_row, end_row]}, ...]}

        Returns:
            - xml_dom: Document, 生成的 xml Document 对象
        """

        def calc_table_area(cells):
            if len(cells) == 0:
                return [[0, 0], [0, 0], [0, 0], [0, 0]]
            xmin = min(min(coord[0] for coord in cell["coords"]) for cell in cells)
            ymin = min(min(coord[1] for coord in cell["coords"]) for cell in cells)
            xmax = max(max(coord[0] for coord in cell["coords"]) for cell in cells)
            ymax = max(max(coord[1] for coord in cell["coords"]) for cell in cells)

            return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]

        root = xml.dom.minidom.Document()
        document = root.createElement("document")

        for table in tables:
            quad = calc_table_area(table["cells"])
            table_elem = root.createElement("table")
            table_coords = root.createElement("Coords")
            table_coords.setAttribute(
                "points",
                f"{int(quad[0][0])},{int(quad[0][1])} {int(quad[1][0])},{int(quad[1][1])} {int(quad[2][0])},{int(quad[2][1])} {int(quad[3][0])},{int(quad[3][1])}",
            )
            table_elem.appendChild(table_coords)

            for cell in table["cells"]:
                cell_elem = root.createElement("cell")
                # cell logic coords
                cell_elem.setAttribute("start-col", cell["structure"][0])
                cell_elem.setAttribute("end-col", cell["structure"][1])
                cell_elem.setAttribute("start-row", cell["structure"][2])
                cell_elem.setAttribute("end-row", cell["structure"][3])
                # cell coordinates
                cell_coords = root.createElement("Coords")
                x1, y1 = cell["coords"][0]
                x2, y2 = cell["coords"][1]
                x3, y3 = cell["coords"][2]
                x4, y4 = cell["coords"][3]
                cell_coords.setAttribute("points", f"{int(x1)},{int(y1)} {int(x2)},{int(y2)} {int(x3)},{int(y3)} {int(x4)},{int(y4)}")

                cell_elem.appendChild(cell_coords)
                table_elem.appendChild(cell_elem)
            document.appendChild(table_elem)
        root.appendChild(document)

        return root

    @staticmethod
    def compute_metrics(evaluate_result_list):
        if len(evaluate_result_list) == 0:
            return []

        iou_results = [[] for _ in range(len(evaluate_result_list[0]))]
        metrics = []
        for evaluate_result in evaluate_result_list:
            for i, iou_result in enumerate(evaluate_result):
                iou_results[i].append(iou_result)

        for iou_result in iou_results:
            truePos, resTotal, gtTotal = 0, 0, 0
            for item in iou_result:
                truePos += item.truePos
                resTotal += item.resTotal
                gtTotal += item.gtTotal

            precision = truePos / resTotal if resTotal > 0 else 0
            recall = truePos / gtTotal if gtTotal > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

            metrics.append({"P": precision, "R": recall, "F1": f1})

        return metrics


class Carte:
    """
    CARTE: Cell Adjacency Relation for Table Evaluation
    URL: https://github.com/naver-ai/carte/
    """

    def __init__(self, pred_structure, gt_structure):
        self.pred_structure = pred_structure
        self.gt_structure = gt_structure

    def evaluate(self):
        gt_dom = Evaluate.generate_table_xml(self.gt_structure)
        pred_dom = Evaluate.generate_table_xml(self.pred_structure)

        # parse the tables in input elements
        gt_table_list = Evaluate.get_table_list(gt_dom, True)
        pred_table_list = Evaluate.get_table_list(pred_dom, True)

        # Combine all tables
        gt_tables = AllTables(gt_table_list)
        pred_tables = AllTables(pred_table_list)

        # Calculate scores
        dict_stat = gt_tables.calculate_score(pred_tables)

        return ResultStructure(truePos=dict_stat["tp"], gtTotal=dict_stat["gt"], resTotal=dict_stat["pred"])

    @staticmethod
    def compute_metric(evaluate_result):
        truePos, resTotal, gtTotal = 0, 0, 0
        for item in evaluate_result:
            truePos += item.truePos
            resTotal += item.resTotal
            gtTotal += item.gtTotal

        precision = truePos / resTotal if resTotal > 0 else 0
        recall = truePos / gtTotal if gtTotal > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        return {"P": precision, "R": recall, "F1": f1}
