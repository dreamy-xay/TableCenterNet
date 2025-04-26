#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description: 
Version: 
Autor: dreamy-xay
Date: 2024-10-22 10:34:01
LastEditors: dreamy-xay
LastEditTime: 2024-10-29 13:55:01
"""
import torch
from engine.table.predictor import TablePredictor
from .decode import cells_decode


class STablePredictor(TablePredictor):
    def __init__(self, args):
        super().__init__(args)

    def process(self, input, meta, *args, **kwargs):
        with torch.no_grad():
            # Model inference
            outputs = self.model(input)
            output = outputs[-1]

            # Obtain the model inference output layer
            hm = output["hm"].sigmoid_()
            reg = output["reg"]
            ct2cn = output["ct2cn"]
            cn2ct = output["cn2ct"]
            lc = output["lc"]
            sp = output["sp"]

            # Output the inference result graph
            # np.save(os.path.join(self.args.save_dir, meta["image_name"]), lc.detach().cpu()[0].numpy())

            # Decoding of cell physical coordinates
            cells, cells_scores, cells_corner_count, logic_coords, *rets = cells_decode(
                hm, reg, ct2cn, cn2ct, lc, sp, self.args.center_k, self.args.corner_k, self.args.center_thresh, self.args.corner_thresh, self.args.save_corners
            )

            # Reduce the score of a cell based on the number of times it is optimized for corners
            is_modify = False
            for i in range(cells.size(1)):
                if cells_scores[0, i, 0] < self.args.center_thresh:
                    break

                if cells_corner_count[0, i, :].sum() <= self.cell_min_optimize_count:
                    cells_scores[0, i, 0] *= self.cell_decay_thresh
                    is_modify = True

            # Merge outputs
            detections = torch.cat([cells, cells_scores, logic_coords], dim=2)

            # If the score is modified, the order will be reordered
            if is_modify:
                _, sorted_inds = torch.sort(cells_scores, descending=True, dim=1)
                detections = detections.gather(1, sorted_inds.expand_as(detections))

            # Returns the test result
            return detections, rets[0] if self.args.save_corners else None, meta
