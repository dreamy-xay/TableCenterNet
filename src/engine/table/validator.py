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

        # Initialize the validator
        super().__init__(None if args.only_eval else (TablePredictor if predictor is None else predictor)(args))

        # Load the COCO dataset annotation file as the gold standard
        self.coco = COCO(args.label)

        self.coco.loadImgs(self.coco.getImgIds())

        # Load the COCO image name mapping
        self.coco_map = {}
        for img in self.coco.dataset["images"]:
            self.coco_map[img["file_name"]] = img["id"]

    def run(self):
        if self.args.only_eval:
            # Read data
            results = self.read_results(self.args.save_results_dir)
        else:
            devices = TablePredictor._get_devices(self.args.device)  # Obtain the GPU required for inference
            is_parallel_infer = self.args.infer_workers * len(devices) > 1  # Whether parallel reasoning is used
            setattr(self.predictor.args, "save_corners", False)  # Set other parameters

            if is_parallel_infer and os.path.isdir(self.args.source):
                print(f"Start multi-process inference. Using GPUs {devices}, and each GPU runs {self.args.infer_workers} processes in parallel.")
                setattr(self.args, "devices", devices)  # Set the multi-process device list
                results = self.parallel_infer(self.args)
            else:
                if is_parallel_infer:
                    print("The input source is a file rather than a directory. Switch to single-process inference.")
                else:
                    print(f"Start single process inference. Using GPU {self.args.device}.")

                results = self.infer(self.args)

            # Save the prediction results
            if self.args.save_result:
                self.predictor.save_results(results, self.args.save_results_dir)

        # Evaluate the results
        coords_evaluate_reuslts, teds_evaluate_reuslts, scitsr_evaluate_reuslts, icdar_ar_evaluate_reuslts, crate_ar_evaluate_reuslts = self.evalute(results, self.args.evaluate_ious)

        # Output the evaluation result
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

        # Save the evaluation results
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
        # * Get all the parameters you need to participate in the evaluation
        # List of parameters for multi-process evaluation
        evalute_args_list = []
        # Start traversal
        for pred_result in pred_results:
            image_name = pred_result["name"]

            if pred_result["type"] != "image" or image_name not in self.coco_map:
                print(f"Skip: {image_name}")
                continue

            # Get a list of cells with the prediction result
            pred_cells = []
            for polygon in pred_result["result"][0]:
                x1, y1, x2, y2, x3, y3, x4, y4 = [float(num) for num in polygon[:8]]
                pred_physical_coord = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                pred_logic_coord = [max(0, int(num) - 1) for num in polygon[9:13]]  # [*polygon[11:13], *polygon[9:11]]
                pred_cells.append([pred_physical_coord, pred_logic_coord])

            # Get a list of standard cells
            gt_cells = []
            ann_ids = self.coco.getAnnIds(imgIds=[self.coco_map[image_name]])
            anns = self.coco.loadAnns(ids=ann_ids)
            for ann in anns:
                # Remove the corner points from the label
                seg_mask = ann["segmentation"][0]
                x1, y1 = seg_mask[0], seg_mask[1]
                x2, y2 = seg_mask[2], seg_mask[3]
                x3, y3 = seg_mask[4], seg_mask[5]
                x4, y4 = seg_mask[6], seg_mask[7]
                # Take out the logical coordinates from the tag
                gt_physical_coord = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                gt_logic_coord = [int(item) for item in ann["logic_axis"][0][:4]]
                gt_cells.append([gt_physical_coord, gt_logic_coord, ann.get("table_id")])

            evalute_args_list.append((image_name, pred_cells, gt_cells))

        # * Multi-process evaluation: Calculate accuracy
        # Physical coordinates PRF1 evaluator
        CellEvaluator = PolygonEvaluator if self.args.evaluate_poly_iou else BoxEvaluator
        # When calculating IOUs, use min area instead of union area
        union_area = not getattr(self.args, "not_union_area", False)

        # Define a single image evaluation function
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

        # Start parallel assessments
        parallel_evalutor = ComputateParallel(parallel_evalute_table, evalute_args_list, self.args.eval_workers).set_tqdm(desc="Evaluate predict results")
        all_evaluate_reuslts = parallel_evalutor.run(False)

        # * Consolidate evaluation parameters and calculate average evaluation results
        # Physical and logical coordinates are evaluated
        coords_evaluate_reuslts = {}
        for threshold in iou_thresholds:
            coords_evaluate_reuslts[threshold] = {
                "images": [],
                "avg": {"num_images": 0, "P": 0.0, "R": 0.0, "F1": 0.0, "L_Acc": 0.0, "Lsr_Acc": 0.0, "Ler_Acc": 0.0, "Lsc_Acc": 0.0, "Lec_Acc": 0.0},
            }

        # Logical coordinates TEDS evaluation results
        teds_evaluate_reuslts = {"images": [], "avg": {}}

        # Cell adjacency evaluation results
        scitsr_evaluate_reuslts = {
            "images": [],
            "avg": {"num_images": 0, "AR_P": 0.0, "AR_R": 0.0, "AR_F1": 0.0},
        }
        icdar_ar_evaluate_reuslts = {}
        crate_ar_evaluate_reuslts = {}

        # Enumerate the evaluation results for each image
        for (
            image_name,
            tc_evalute_results,
            ts_evaluate_reuslts,
            sr_evaluate_reuslts,
            _,
            _,
        ) in all_evaluate_reuslts:
            # Store the evaluation results
            for threshold, item in tc_evalute_results.items():
                coords_evaluate_reuslts[threshold]["images"].append({"image_name": image_name, **item})

            teds_evaluate_reuslts["images"].append({"image_name": image_name, "TEDS": ts_evaluate_reuslts})
            scitsr_evaluate_reuslts["images"].append({"image_name": image_name, **sr_evaluate_reuslts})

        # Calculate the average evaluation result of physical coordinates
        for evaluate_reuslt in coords_evaluate_reuslts.values():
            num_images = len(evaluate_reuslt["images"])

            for metric in ["P", "R", "L_Acc", "Lsr_Acc", "Ler_Acc", "Lsc_Acc", "Lec_Acc"]:
                evaluate_reuslt["avg"][metric] = sum([item[metric] for item in evaluate_reuslt["images"]]) / num_images

            evaluate_reuslt["avg"]["F1"] = CellEvaluator.calc_f1_score(evaluate_reuslt["avg"]["P"], evaluate_reuslt["avg"]["R"])

            evaluate_reuslt["avg"]["num_images"] = num_images

        # Calculate the average evaluation result of logical coordinates
        num_images = len(teds_evaluate_reuslts["images"])
        teds_evaluate_reuslts["avg"]["num_images"] = num_images
        teds_evaluate_reuslts["avg"]["TEDS"] = sum([item["TEDS"] for item in teds_evaluate_reuslts["images"]]) / num_images

        # Calculating Cell Adjacency Average Evaluation Results (SCITSR)
        num_images = len(scitsr_evaluate_reuslts["images"])
        scitsr_evaluate_reuslts["avg"]["num_images"] = num_images
        for metric in ["P", "R"]:
            scitsr_evaluate_reuslts["avg"][metric] = sum([item[metric] for item in scitsr_evaluate_reuslts["images"]]) / num_images
        scitsr_evaluate_reuslts["avg"]["F1"] = CellEvaluator.calc_f1_score(scitsr_evaluate_reuslts["avg"]["P"], scitsr_evaluate_reuslts["avg"]["R"])

        # Calculating the Average Evaluation Results of Cell Adjacency Relationships (ICDAR2019)
        for i, ar_evaluate_reuslt in enumerate(LogAdjRelationEvaluator.average([results[4] for results in all_evaluate_reuslts])):
            icdar_ar_evaluate_reuslts[iou_thresholds[i]] = ar_evaluate_reuslt

        # Calculate Cell Adjacency Average Evaluation Result (Crate)
        crate_ar_evaluate_reuslts = LogAdjRelationEvaluator.average([results[5] for results in all_evaluate_reuslts], True)

        return coords_evaluate_reuslts, teds_evaluate_reuslts, scitsr_evaluate_reuslts, icdar_ar_evaluate_reuslts, crate_ar_evaluate_reuslts

    def save_evaluate_results(self, evaluate_reuslts, save_path):
        # Parsing
        coords_evaluate_reuslts, teds_evaluate_reuslts, scitsr_evaluate_reuslts, icdar_ar_evaluate_reuslts, crate_ar_evaluate_reuslts = evaluate_reuslts

        # markdown content
        markdown = "# The evaluation results of the table structure recognition"

        # Create an Excel writer object
        excel_path = os.path.join(save_path, "evaluate_results.xlsx")
        with pd.ExcelWriter(excel_path) as writer:
            # Step 1: Merge the avg values of each IoU to generate a table
            avg_data = []
            for iou, data in coords_evaluate_reuslts.items():
                avg_data.append({"IoU": iou, **data["avg"]})

            avg_df = pd.DataFrame(avg_data)
            avg_df.rename(
                columns={
                    "IoU": "IoU",
                    "num_images": "Number of images",
                    "P": "Precision",
                    "R": "Recall",
                    "F1": "F1",
                    "L_Acc": "Logical Coordinate Accuracy (L_Acc)",
                    "Lsr_Acc": "Start Row accuracy (Lsr_Acc)",
                    "Ler_Acc": "End Row Accuracy (Ler_Acc)",
                    "Lsc_Acc": "Start Column Accuracy (Lsc_Acc)",
                    "Lec_Acc": "End Column Accuracy (Lec_Acc)",
                },
                inplace=True,
            )
            avg_df.to_excel(writer, sheet_name="Cells Physical and Logical Coordinate Avg Results", index=False)

            # Use df.to_markdown to output tables
            markdown += "\n## Cells Physical and Logical Coordinate Avg Results\n"
            markdown += avg_df.to_markdown(index=False)

            # Step 2: Generate a table of the results of the evaluation of the logical coordinates
            avg_df = pd.DataFrame([teds_evaluate_reuslts["avg"]])
            avg_df.rename(columns={"num_images": "Number of images", "TEDS": "TEDS"}, inplace=True)
            avg_df.to_excel(writer, sheet_name="Table Structure Avg Results", index=False)

            # Use df.to_markdown to output tables
            markdown += "\n## Table Structure Avg Results\n"
            markdown += avg_df.to_markdown(index=False)

            # Step 3: Merge the cell adjacency results of each IoU to generate a table
            avg_data = []
            for iou, data in icdar_ar_evaluate_reuslts.items():
                avg_data.append({"Type": "ICDAR2019", "IoU": iou, "P": data["P"], "R": data["R"], "F1": data["F1"]})
            for tp, data in [("SCITSR", scitsr_evaluate_reuslts["avg"]), ("CRATE", crate_ar_evaluate_reuslts)]:
                avg_data.append({"Type": tp, "IoU": "-", "P": data["P"], "R": data["R"], "F1": data["F1"]})

            avg_df = pd.DataFrame(avg_data)
            avg_df.rename(
                columns={"Type": "Type", "IoU": "IoU", "P": "Precision", "R": "Recall", "F1": "F1"},
                inplace=True,
            )
            avg_df.to_excel(writer, sheet_name="Cells Adjacency Relation Avg Results", index=False)

            # Use df.to_markdown to output tables
            markdown += "\n## Cells Adjacency Relation Avg Results\n"
            markdown += avg_df.to_markdown(index=False)

            # Step 4: Generate a separate table for the images in each IoU
            # markdown  = "\n## Cells Physical and Logical Coordinate Results"
            for iou, data in coords_evaluate_reuslts.items():
                images_df = pd.DataFrame(data["images"])
                images_df.rename(
                    columns={
                        "IoU": "IoU",
                        "num_images": "Number of images",
                        "P": "Precision",
                        "R": "Recall",
                        "F1": "F1",
                        "L_Acc": "Logical Coordinate Accuracy (L_Acc)",
                        "Lsr_Acc": "Start Row accuracy (Lsr_Acc)",
                        "Ler_Acc": "End Row Accuracy (Ler_Acc)",
                        "Lsc_Acc": "Start Column Accuracy (Lsc_Acc)",
                        "Lec_Acc": "End Column Accuracy (Lec_Acc)",
                    },
                    inplace=True,
                )
                images_df.to_excel(writer, sheet_name=f"IoU_{iou} => Cells Physical and Logical Coordinate Results", index=False)

                # Use df.to_markdown to output tables
                # markdown  = f"\n### IoU_{iou}\n"
                # markdown  = images_df.to_markdown(index=False)

            # Step 5: Generate a separate table by the images in the logical coordinate evaluation results
            images_df = pd.DataFrame(teds_evaluate_reuslts["images"])
            images_df.rename(columns={"image_name": "Image Name", "TEDS": "TEDS"}, inplace=True)
            images_df.to_excel(writer, sheet_name="Table Structure Results", index=False)

            # Use df.to_markdown to output tables
            # markdown  = "\n## Table Structure Results\n"
            # markdown  = images_df.to_markdown(index=False)

        # Create a mrakdown file and write the contents
        with open(os.path.join(save_path, "evaluate_results.md"), "w+") as f:
            f.write(markdown)

        # Format the Excel file
        format_excel(excel_path)
