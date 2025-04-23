#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description: 
Version: 
Autor: dreamy-xay
Date: 2024-10-22 10:34:01
LastEditors: dreamy-xay
LastEditTime: 2024-10-24 11:20:23
"""
import os
import glob
import json
import pandas as pd
from engine.base.validator import BaseValidator
from .predictor import TablePredictor
from pycocotools.coco import COCO
from utils.evaluator import PolygonEvaluator, BoxEvaluator, TableStructEvaluator, TableCoordsEvaluator, ScitsrRelationEvaluator, LogAdjRelationEvaluator
from utils.parallel import ComputateParallel
from utils.excel import format_excel


class TableValidator(BaseValidator):

    def __init__(self, args, predictor=None):
        self.args = args

        # 初始化验证器
        super().__init__(None if args.only_eval else (TablePredictor if predictor is None else predictor)(args))

        # 加载 COCO 数据集标注文件作为金标准
        self.coco = COCO(args.label)

        self.coco.loadImgs(self.coco.getImgIds())

        # 加载 COCO 图片名映射
        self.coco_map = {}
        for img in self.coco.dataset["images"]:
            self.coco_map[img["file_name"]] = img["id"]

    def run(self):
        if self.args.only_eval:
            # 读取数据
            results = self.read_results(self.args.save_results_dir)
        else:
            devices = TablePredictor._get_devices(self.args.device)  # 获取推理所需GPU
            is_parallel_infer = self.args.infer_workers * len(devices) > 1  # 是否并行推理
            setattr(self.predictor.args, "save_corners", False)  # 设置其他参数

            if is_parallel_infer and os.path.isdir(self.args.source):
                print(f"Start multi-process inference. Using GPUs {devices}, and each GPU runs {self.args.infer_workers} processes in parallel.")
                setattr(self.args, "devices", devices)  # 设置多进程设备列表
                results = self.parallel_infer(self.args)
            else:
                if is_parallel_infer:
                    print("The input source is a file rather than a directory. Switch to single-process inference.")
                else:
                    print(f"Start single process inference. Using GPU {self.args.device}.")

                results = self.infer(self.args)

            # 保存预测结果
            if self.args.save_result:
                self.predictor.save_results(results, self.args.save_results_dir)

        # 评估结果
        coords_evaluate_reuslts, teds_evaluate_reuslts, scitsr_evaluate_reuslts, icdar_ar_evaluate_reuslts, crate_ar_evaluate_reuslts = self.evalute(results, self.args.evaluate_ious)

        # 输出评估结果
        for threshold, item in coords_evaluate_reuslts.items():
            print(f"IoU {threshold}: Precision=>{item['avg']['P']}, Recall=>{item['avg']['R']}, F1=>{item['avg']['F1']}, Accuracy(LogicCoords)=>{item['avg']['L_Acc']}")
            print(
                f"> Accuracy(Start Row)=>{item['avg']['Lsr_Acc']} Accuracy(End Row)=>{item['avg']['Ler_Acc']} Accuracy(Start Col)=>{item['avg']['Lsc_Acc']} Accuracy(End Col)=>{item['avg']['Lec_Acc']}"
            )
        print(f"TEDS(only structure)=>{teds_evaluate_reuslts['avg']['TEDS']}")
        for threshold, item in icdar_ar_evaluate_reuslts.items():
            print(f"IoU {threshold}(Cell Adjacency Relation): Precision=>{item['P']}, Recall=>{item['R']}, F1=>{item['F1']}")
        for tp, item in [("Scitsr", scitsr_evaluate_reuslts["avg"]), ("Crate", crate_ar_evaluate_reuslts)]:
            print(f"{tp}(Cell Adjacency Relation): Precision=>{item['P']}, Recall=>{item['R']}, F1=>{item['F1']}")

        # 保存评估结果
        self.save_evaluate_results((coords_evaluate_reuslts, teds_evaluate_reuslts, scitsr_evaluate_reuslts, icdar_ar_evaluate_reuslts, crate_ar_evaluate_reuslts), self.args.save_dir)

    def read_results(self, results_path):
        json_file_paths = glob.glob(os.path.join(results_path, "*.json"))

        results = []

        for json_file_path in json_file_paths:
            with open(json_file_path, "r") as f:
                content = json.load(f)

                type = content["type"]

                if type != "image":
                    continue

                name = content["file_name"]
                result = []
                for cell in content["cells"]:
                    cell_coords = cell["cell"]
                    result.append(
                        [
                            cell_coords["x1"],
                            cell_coords["y1"],
                            cell_coords["x2"],
                            cell_coords["y2"],
                            cell_coords["x3"],
                            cell_coords["y3"],
                            cell_coords["x4"],
                            cell_coords["y4"],
                            cell["score"],
                            cell["start_col"],
                            cell["end_col"],
                            cell["start_row"],
                            cell["end_row"],
                        ]
                    )

                results.append({"type": type, "name": name, "result": [result]})

        return results

    def evalute(self, pred_results, iou_thresholds):
        # * 获取需要参与评估的全部参数
        # 多进程评估参数列表
        evalute_args_list = []
        # 开始遍历
        for pred_result in pred_results:
            image_name = pred_result["name"]

            if pred_result["type"] != "image" or image_name not in self.coco_map:
                print(f"Skip: {image_name}")
                continue

            # 获取预测结果的单元格列表
            pred_cells = []
            for polygon in pred_result["result"][0]:
                x1, y1, x2, y2, x3, y3, x4, y4 = [float(num) for num in polygon[:8]]
                pred_physical_coord = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                pred_logic_coord = [max(0, int(num) - 1) for num in polygon[9:13]]  # [*polygon[11:13], *polygon[9:11]]
                pred_cells.append([pred_physical_coord, pred_logic_coord])

            # 获取标准单元格列表
            gt_cells = []
            ann_ids = self.coco.getAnnIds(imgIds=[self.coco_map[image_name]])
            anns = self.coco.loadAnns(ids=ann_ids)
            for ann in anns:
                # 从标签中取出角点
                seg_mask = ann["segmentation"][0]
                x1, y1 = seg_mask[0], seg_mask[1]
                x2, y2 = seg_mask[2], seg_mask[3]
                x3, y3 = seg_mask[4], seg_mask[5]
                x4, y4 = seg_mask[6], seg_mask[7]
                # 从标签中取出逻辑坐标
                gt_physical_coord = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                gt_logic_coord = [int(item) for item in ann["logic_axis"][0][:4]]
                gt_cells.append([gt_physical_coord, gt_logic_coord, ann.get("table_id")])

            evalute_args_list.append((image_name, pred_cells, gt_cells))

        # * 多进程评估：计算准确率
        # 物理坐标PRF1评估器
        CellEvaluator = PolygonEvaluator if self.args.evaluate_poly_iou else BoxEvaluator
        # 计算IOU时使用不使用 union area，而使用 min area
        union_area = not getattr(self.args, "not_union_area", False)

        # 定义单张图片评估函数
        def parallel_evalute_table(image_name, pred_cells, gt_cells):
            gt_tables, is_multitable = LogAdjRelationEvaluator.split_tables_by_id(gt_cells)
            if is_multitable:
                pred_tables = LogAdjRelationEvaluator.split_tables(pred_cells, 5)
            else:
                pred_tables = [pred_cells]
            ar_evaluator = LogAdjRelationEvaluator(pred_tables, gt_tables)
            return (
                image_name,
                TableCoordsEvaluator(CellEvaluator, pred_cells, gt_cells, union_area).evaluate(iou_thresholds),
                TableStructEvaluator(pred_tables, gt_tables).evaluate(),
                ScitsrRelationEvaluator(pred_cells, gt_cells).evaluate(),
                ar_evaluator.evaluate(iou_thresholds),
                ar_evaluator.evaluate_carte(),
            )

        # 开始并行评估
        parallel_evalutor = ComputateParallel(parallel_evalute_table, evalute_args_list, self.args.eval_workers).set_tqdm(desc="Evaluate predict results")
        all_evaluate_reuslts = parallel_evalutor.run(False)

        # * 整合评估参数并计算平均评估结果
        # 物理坐标和逻辑坐标评估结果
        coords_evaluate_reuslts = {}
        for threshold in iou_thresholds:
            coords_evaluate_reuslts[threshold] = {
                "images": [],
                "avg": {"num_images": 0, "P": 0.0, "R": 0.0, "F1": 0.0, "L_Acc": 0.0, "Lsr_Acc": 0.0, "Ler_Acc": 0.0, "Lsc_Acc": 0.0, "Lec_Acc": 0.0},
            }

        # 逻辑坐标teds评估结果
        teds_evaluate_reuslts = {"images": [], "avg": {}}

        # 单元格邻接关系评估结果
        scitsr_evaluate_reuslts = {
            "images": [],
            "avg": {"num_images": 0, "AR_P": 0.0, "AR_R": 0.0, "AR_F1": 0.0},
        }
        icdar_ar_evaluate_reuslts = {}
        crate_ar_evaluate_reuslts = {}

        # 枚举每张图片评估结果
        for (
            image_name,
            tc_evalute_results,
            ts_evaluate_reuslts,
            sr_evaluate_reuslts,
            _,
            _,
        ) in all_evaluate_reuslts:
            # 存储评估结果
            for threshold, item in tc_evalute_results.items():
                coords_evaluate_reuslts[threshold]["images"].append({"image_name": image_name, **item})

            teds_evaluate_reuslts["images"].append({"image_name": image_name, "TEDS": ts_evaluate_reuslts})
            scitsr_evaluate_reuslts["images"].append({"image_name": image_name, **sr_evaluate_reuslts})

        # 计算物理坐标平均评估结果
        for evaluate_reuslt in coords_evaluate_reuslts.values():
            num_images = len(evaluate_reuslt["images"])

            for metric in ["P", "R", "L_Acc", "Lsr_Acc", "Ler_Acc", "Lsc_Acc", "Lec_Acc"]:
                evaluate_reuslt["avg"][metric] = sum([item[metric] for item in evaluate_reuslt["images"]]) / num_images

            evaluate_reuslt["avg"]["F1"] = CellEvaluator.calc_f1_score(evaluate_reuslt["avg"]["P"], evaluate_reuslt["avg"]["R"])

            evaluate_reuslt["avg"]["num_images"] = num_images

        # 计算逻辑坐标平均评估结果
        num_images = len(teds_evaluate_reuslts["images"])
        teds_evaluate_reuslts["avg"]["num_images"] = num_images
        teds_evaluate_reuslts["avg"]["TEDS"] = sum([item["TEDS"] for item in teds_evaluate_reuslts["images"]]) / num_images

        # 计算单元格邻接关系平均评估结果（scitsr）
        num_images = len(scitsr_evaluate_reuslts["images"])
        scitsr_evaluate_reuslts["avg"]["num_images"] = num_images
        for metric in ["P", "R"]:
            scitsr_evaluate_reuslts["avg"][metric] = sum([item[metric] for item in scitsr_evaluate_reuslts["images"]]) / num_images
        scitsr_evaluate_reuslts["avg"]["F1"] = CellEvaluator.calc_f1_score(scitsr_evaluate_reuslts["avg"]["P"], scitsr_evaluate_reuslts["avg"]["R"])

        # 计算单元格邻接关系平均评估结果（icdar2019）
        for i, ar_evaluate_reuslt in enumerate(LogAdjRelationEvaluator.average([results[4] for results in all_evaluate_reuslts])):
            icdar_ar_evaluate_reuslts[iou_thresholds[i]] = ar_evaluate_reuslt

        # 计算单元格邻接关系平均评估结果（crate）
        crate_ar_evaluate_reuslts = LogAdjRelationEvaluator.average([results[5] for results in all_evaluate_reuslts], True)

        return coords_evaluate_reuslts, teds_evaluate_reuslts, scitsr_evaluate_reuslts, icdar_ar_evaluate_reuslts, crate_ar_evaluate_reuslts

    def save_evaluate_results(self, evaluate_reuslts, save_path):
        # 解析
        coords_evaluate_reuslts, teds_evaluate_reuslts, scitsr_evaluate_reuslts, icdar_ar_evaluate_reuslts, crate_ar_evaluate_reuslts = evaluate_reuslts

        # markdown 内容
        markdown = "# 表格结构识别的评估结果"

        # 创建一个 Excel writer 对象
        excel_path = os.path.join(save_path, "evaluate_results.xlsx")
        with pd.ExcelWriter(excel_path) as writer:
            # Step 1: 合并各 IoU 的 avg 值生成一张表
            avg_data = []
            for iou, data in coords_evaluate_reuslts.items():
                avg_data.append({"IoU": iou, **data["avg"]})

            avg_df = pd.DataFrame(avg_data)
            avg_df.rename(
                columns={
                    "IoU": "交并集比（IoU）",
                    "num_images": "图像数量",
                    "P": "精确率（Precision）",
                    "R": "召回率（Recall）",
                    "F1": "调和平均（F1）",
                    "L_Acc": "逻辑坐标准确率（L_Acc）",
                    "Lsr_Acc": "开始行准确率（Lsr_Acc）",
                    "Ler_Acc": "结束行准确率（Ler_Acc）",
                    "Lsc_Acc": "开始列准确率（Lsc_Acc）",
                    "Lec_Acc": "结束列准确率（Lec_Acc）",
                },
                inplace=True,
            )
            avg_df.to_excel(writer, sheet_name="Cells Physical and Logical Coordinate Avg Results", index=False)

            # 使用 df.to_markdown 输出表格
            markdown += "\n## Cells Physical and Logical Coordinate Avg Results\n"
            markdown += avg_df.to_markdown(index=False)

            # Step 2: 生成一张逻辑坐标评估结果的表格
            avg_df = pd.DataFrame([teds_evaluate_reuslts["avg"]])
            avg_df.rename(columns={"num_images": "图像数量", "TEDS": "基于树编辑距离的相似度（TEDS）"}, inplace=True)
            avg_df.to_excel(writer, sheet_name="Table Structure Avg Results", index=False)

            # 使用 df.to_markdown 输出表格
            markdown += "\n## Table Structure Avg Results\n"
            markdown += avg_df.to_markdown(index=False)

            # Step 3: 合并各 IoU 的单元格邻接关系结果生成一张表
            avg_data = []
            for iou, data in icdar_ar_evaluate_reuslts.items():
                avg_data.append({"Type": "ICDAR2019", "IoU": iou, "P": data["P"], "R": data["R"], "F1": data["F1"]})
            for tp, data in [("SCITSR", scitsr_evaluate_reuslts["avg"]), ("CRATE", crate_ar_evaluate_reuslts)]:
                avg_data.append({"Type": tp, "IoU": "-", "P": data["P"], "R": data["R"], "F1": data["F1"]})

            avg_df = pd.DataFrame(avg_data)
            avg_df.rename(
                columns={"Type": "评估代码类型", "IoU": "交并集比（IoU）", "P": "精确率（Precision）", "R": "召回率（Recall）", "F1": "调和平均（F1）"},
                inplace=True,
            )
            avg_df.to_excel(writer, sheet_name="Cells Adjacency Relation Avg Results", index=False)

            # 使用 df.to_markdown 输出表格
            markdown += "\n## Cells Adjacency Relation Avg Results\n"
            markdown += avg_df.to_markdown(index=False)

            # Step 4: 每个 IoU 中的 images 生成单独的表格
            # markdown += "\n## Cells Physical and Logical Coordinate Results"
            for iou, data in coords_evaluate_reuslts.items():
                images_df = pd.DataFrame(data["images"])
                images_df.rename(
                    columns={
                        "IoU": "交并集比（IoU）",
                        "num_images": "图像数量",
                        "P": "精确率（Precision）",
                        "R": "召回率（Recall）",
                        "F1": "调和平均（F1）",
                        "L_Acc": "逻辑坐标准确率（L_Acc）",
                        "Lsr_Acc": "开始行准确率（Lsr_Acc）",
                        "Ler_Acc": "结束行准确率（Ler_Acc）",
                        "Lsc_Acc": "开始列准确率（Lsc_Acc）",
                        "Lec_Acc": "结束列准确率（Lec_Acc）",
                    },
                    inplace=True,
                )
                images_df.to_excel(writer, sheet_name=f"IoU_{iou} => Cells Physical and Logical Coordinate Results", index=False)

                # 使用 df.to_markdown 输出表格
                # markdown += f"\n### IoU_{iou}\n"
                # markdown += images_df.to_markdown(index=False)

            # Step 5: 逻辑坐标评估结果中的 images 生成单独的表格
            images_df = pd.DataFrame(teds_evaluate_reuslts["images"])
            images_df.rename(columns={"image_name": "图像名", "TEDS": "基于树编辑距离的相似度（TEDS）"}, inplace=True)
            images_df.to_excel(writer, sheet_name="Table Structure Results", index=False)

            # 使用 df.to_markdown 输出表格
            # markdown += "\n## Table Structure Results\n"
            # markdown += images_df.to_markdown(index=False)

        # 创建一个mrakdown文件并写入内容
        with open(os.path.join(save_path, "evaluate_results.md"), "w+") as f:
            f.write(markdown)

        # 格式化 Excel 文件
        format_excel(excel_path)
