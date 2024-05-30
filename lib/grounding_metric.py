# Copyright (c) OpenRobotLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Union, Any
# 这份代码并不设计在这个项目中运行。
from terminaltables import AsciiTable
import logging
import numpy as np
import torch
from lib.euler_utils import euler_iou3d_split

def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (np.ndarray): Recalls with shape of (num_scales, num_dets)
            or (num_dets, ).
        precisions (np.ndarray): Precisions with shape of
            (num_scales, num_dets) or (num_dets, ).
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or np.ndarray: Calculated average precision.
    """
    if recalls.ndim == 1:
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]

    assert recalls.shape == precisions.shape
    assert recalls.ndim == 2

    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    return ap
    
mapping = {
    'single_attribute_unique': 'sin_attr_uni',
    'single_attribute_common': 'sin_attr_com',
    'single_space': 'sin_space',
    'single_eq': 'sin_eq',
    'inter_or': 'int_or',
    'inter_space_oo': 'int_space',
    'inter_attribute_oo': 'int_attr',
    'other': 'other',
    'overall': 'overall'
}

def ground_eval_subset(gt_anno_list, det_anno_list, logger=None, prefix=''):
    """
        det_anno_list: list of dictionaries with keys:
            'bboxes_3d': (N, 9) or a (list, tuple) (center, size, rotmat): (N, 3), (N, 3), (N, 3, 3)
            'target_scores_3d': (N, )
        gt_anno_list: list of dictionaries with keys:
            'gt_bboxes_3d': (M, 9) or a (list, tuple) (center, size, rotmat): (M, 3), (M, 3), (M, 3, 3)
            'sub_class': str
    """
    assert len(det_anno_list) == len(gt_anno_list)
    iou_thr = [0.25, 0.5]
    num_samples = len(gt_anno_list) # each sample contains multiple pred boxes
    total_gt_boxes = 0
    # these lists records for each sample, whether a gt box is matched or not
    gt_matched_records = [[] for _ in iou_thr]
    # these lists records for each pred box, NOT for each sample        
    sample_indices = [] # each pred box belongs to which sample
    confidences = [] # each pred box has a confidence score
    ious = [] # each pred box has a ious, shape (num_gt) in the corresponding sample
    # record the indices of each reference type

    for sample_idx in range(num_samples):
        det_anno = det_anno_list[sample_idx]
        gt_anno = gt_anno_list[sample_idx]

        target_scores = det_anno['score']  # (num_query, )
        top_idxs =  torch.argsort(target_scores, descending=True)[:20] #HACK: hard coded
        target_scores = target_scores[top_idxs]
        pred_center = det_anno['center'][top_idxs]
        pred_size = det_anno['size'][top_idxs]
        pred_rot = det_anno['rot'][top_idxs]
        
        gt_center = gt_anno['center']
        gt_size = gt_anno['size']
        gt_rot = gt_anno['rot']

        num_preds = pred_center.shape[0]
        num_gts = gt_center.shape[0]
        total_gt_boxes += num_gts
        for iou_idx in range(len(iou_thr)):
            gt_matched_records[iou_idx].append(np.zeros(num_gts, dtype=bool))

        iou_mat = euler_iou3d_split(pred_center, pred_size, pred_rot, gt_center, gt_size, gt_rot)
        for i, score in enumerate(target_scores):
            sample_indices.append(sample_idx)
            confidences.append(score)
            ious.append(iou_mat[i])


    confidences = np.array(confidences)
    sorted_inds = np.argsort(-confidences)
    sample_indices = [sample_indices[i] for i in sorted_inds]
    ious = [ious[i] for i in sorted_inds]

    tp_thr = {}
    fp_thr = {}
    for thr in iou_thr:
        tp_thr[f'{prefix}@{thr}'] = np.zeros(len(sample_indices))
        fp_thr[f'{prefix}@{thr}'] = np.zeros(len(sample_indices))

    for d, sample_idx in enumerate(sample_indices):
        iou_max = -np.inf
        num_gts = gt_anno_list[sample_idx]['center'].shape[0]
        cur_iou = ious[d]
        if num_gts > 0:
            for j in range(num_gts):
                iou = cur_iou[j]
                if iou > iou_max:
                    iou_max = iou
                    jmax = j

        for iou_idx, thr in enumerate(iou_thr):
            if iou_max >= thr:
                if not gt_matched_records[iou_idx][sample_idx][jmax]:
                    gt_matched_records[iou_idx][sample_idx][jmax] = True
                    tp_thr[f'{prefix}@{thr}'][d] = 1.0
                else:
                    fp_thr[f'{prefix}@{thr}'][d] = 1.0
            else:
                fp_thr[f'{prefix}@{thr}'][d] = 1.0

    ret = {}
    for t in iou_thr:
        metric = prefix + '@' + str(t)
        fp = np.cumsum(fp_thr[metric])
        tp = np.cumsum(tp_thr[metric])
        recall = tp / float(total_gt_boxes)
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = average_precision(recall, precision)
        ret[metric] = float(ap)
        best_recall = recall[-1] if len(recall) > 0 else 0
        ret[metric + '_rec'] = float(best_recall)
    return ret

def ground_eval(gt_anno_list, det_anno_list, logger=None):
    """
        det_anno_list: list of dictionaries with keys:
            'bboxes_3d': (N, 9) or a (list, tuple) (center, size, rotmat): (N, 3), (N, 3), (N, 3, 3)
            'target_scores_3d': (N, )
        gt_anno_list: list of dictionaries with keys:
            'gt_bboxes_3d': (M, 9) or a (list, tuple) (center, size, rotmat): (M, 3), (M, 3), (M, 3, 3)
            'sub_class': str
    """
    iou_thr = [0.25, 0.5]
    reference_options = [v for k, v in mapping.items()]
    assert len(det_anno_list) == len(gt_anno_list)
    results = {}
    for ref in reference_options:
        indices = [i for i, gt_anno in enumerate(gt_anno_list) if gt_anno.get('sub_class', 'other').strip('vg_') == ref]
        sub_gt_annos = [gt_anno_list[i] for i in indices ]
        sub_det_annos = [det_anno_list[i] for i in indices ]
        ret = ground_eval_subset(sub_gt_annos, sub_det_annos, logger=logger, prefix=ref)
        for k, v in ret.items():
            results[k] = v
    overall_ret = ground_eval_subset(gt_anno_list, det_anno_list, logger=logger, prefix='overall')
    for k, v in overall_ret.items():
        results[k] = v

    header = ['Type']
    header.extend(reference_options)
    table_columns = [[] for _ in range(len(header))]
    ret = {}
    for t in iou_thr:
        table_columns[0].append('AP  '+str(t))
        table_columns[0].append('Rec '+str(t))            
        for i, ref in enumerate(reference_options):
            metric = ref + '@' + str(t)
            ap = results[metric]
            best_recall = results[metric + '_rec']
            table_columns[i+1].append(f'{float(ap):.4f}')
            table_columns[i+1].append(f'{float(best_recall):.4f}')

    table_data = [header]
    table_rows = list(zip(*table_columns))
    table_data += table_rows
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    # print('\n' + table.table)
    if logger is not None:
        logger.write('\n' + table.table + '\n')
        logger.flush()
    print('\n' + table.table)

    return ret