# Copyright (c) OpenRobotLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Union, Any
from terminaltables import AsciiTable
import logging
import numpy as np
import torch
from tqdm import tqdm
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

def abbr(sub_class):
    sub_class = sub_class.lower()
    sub_class = sub_class.replace('single', 'sngl')
    sub_class = sub_class.replace('inter', 'int')
    sub_class = sub_class.replace('unique', 'uniq')
    sub_class = sub_class.replace('common', 'cmn')
    sub_class = sub_class.replace('attribute', 'attr')
    if 'sngl' in sub_class and ('attr' in sub_class or 'eq' in sub_class):
        sub_class = 'vg_sngl_attr'
    return sub_class

def ground_eval_single_query(gt_anno, det_anno, logger=None, prefix=''):
    iou_thr = [0.25, 0.5]
    target_scores = det_anno['score']  # (num_query, )
    top_idxs =  torch.argsort(target_scores, descending=True)
    target_scores = target_scores[top_idxs]
    pred_center = det_anno['center'][top_idxs]
    pred_size = det_anno['size'][top_idxs]
    pred_rot = det_anno['rot'][top_idxs]
    
    gt_center = gt_anno['center']
    gt_size = gt_anno['size']
    gt_rot = gt_anno['rot']

    num_preds = pred_center.shape[0]
    num_gts = gt_center.shape[0]
    
    if num_gts == 0:
        ret = {}
        for t in iou_thr:
            ret[f'{prefix}@{t}'] = np.nan
            ret[f'{prefix}@{t}_rec'] = np.nan
        ret[prefix + '_num_gt'] = num_gts
        return ret

    ious = euler_iou3d_split(pred_center, pred_size, pred_rot, gt_center, gt_size, gt_rot)
    # num_pred 

    confidences = np.array(target_scores)
    sorted_inds = np.argsort(-confidences)
    gt_matched_records = [np.zeros((num_gts), dtype=bool) for _ in iou_thr]
    tp_thr = {}
    fp_thr = {}
    for thr in iou_thr:
        tp_thr[f'{prefix}@{thr}'] = np.zeros(num_preds)
        fp_thr[f'{prefix}@{thr}'] = np.zeros(num_preds)

    for d, pred_idx in enumerate(range(num_preds)):
        iou_max = -np.inf
        cur_iou = ious[d]
        num_gts = cur_iou.shape[0]

        if num_gts > 0:
            for j in range(num_gts):
                iou = cur_iou[j]
                if iou > iou_max:
                    iou_max = iou
                    jmax = j
        
        for iou_idx, thr in enumerate(iou_thr):
            if iou_max >= thr:
                if not gt_matched_records[iou_idx][jmax]:
                    gt_matched_records[iou_idx][jmax] = True
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
        recall = tp / float(num_gts)
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = average_precision(recall, precision)
        ret[metric] = float(ap)
        best_recall = recall[-1] 
        ret[metric + '_rec'] = float(best_recall)
    ret[prefix + '_num_gt'] = num_gts
    return ret




def ground_eval(gt_annos, det_annos, logger=None):
    """
        det_anno_list: list of dictionaries with keys:
            'bboxes_3d': (N, 9) or a (list, tuple) (center, size, rotmat): (N, 3), (N, 3), (N, 3, 3)
            'target_scores_3d': (N, )
        gt_anno_list: list of dictionaries with keys:
            'gt_bboxes_3d': (M, 9) or a (list, tuple) (center, size, rotmat): (M, 3), (M, 3), (M, 3, 3)
            'sub_class': str
    """
    iou_thr = [0.25, 0.5]
    reference_options = [abbr(gt_anno.get('sub_class', 'other')) for gt_anno in gt_annos]
    reference_options = list(set(reference_options))
    reference_options.sort()
    reference_options.append('overall')
    assert len(det_annos) == len(gt_annos)
    metric_results = {}
    for i, (gt_anno, det_anno) in tqdm(enumerate(zip(gt_annos, det_annos))):
        partial_metric = ground_eval_single_query(gt_anno, det_anno, logger=logger, prefix=abbr(gt_anno.get('sub_class', 'other')))
        for k, v in partial_metric.items():
            if k not in metric_results:
                metric_results[k] = []
            metric_results[k].append(v)
    for thr in iou_thr:
        metric_results['overall@' + str(thr)] = []
        metric_results['overall@' + str(thr) + '_rec'] = []
    metric_results['overall_num_gt'] = 0
    for ref in reference_options:
        for thr in iou_thr:
            metric = ref + '@' + str(thr)
            if ref != 'overall':
                metric_results['overall@' + str(thr)] += metric_results[metric]
                metric_results['overall@' + str(thr) + '_rec'] += metric_results[metric + '_rec']
            ap = np.nanmean(metric_results[metric])
            rec = np.nanmean(metric_results[metric + '_rec'])
            metric_results[metric] = ap
            metric_results[metric + '_rec'] = rec       
        metric_results[ref + '_num_gt'] = np.sum(metric_results[ref + '_num_gt'])
        if ref != 'overall':
            metric_results['overall_num_gt'] += np.sum(metric_results[ref + '_num_gt'])
    # Print the precision and recall for each iou threshold
    header = ['Type']
    header.extend(reference_options)
    table_columns = [[] for _ in range(len(header))]
    for t in iou_thr:
        table_columns[0].append('AP  '+str(t))
        table_columns[0].append('Rec '+str(t))            
        for i, ref in enumerate(reference_options):
            metric = ref + '@' + str(t)
            assert isinstance(metric, str)
            ap = metric_results[metric]
            best_recall = metric_results[metric + '_rec']
            table_columns[i+1].append(f'{float(ap):.4f}')
            table_columns[i+1].append(f'{float(best_recall):.4f}')
    table_columns[0].append('Num GT')            
    for i, ref in enumerate(reference_options):
        # add num_gt
        table_columns[i+1].append(f'{int(metric_results[ref + "_num_gt"])}')

    table_data = [header]
    table_rows = list(zip(*table_columns))
    table_data += table_rows
    table_data = [list(row) for row in zip(*table_data)] # transpose the table
    table = AsciiTable(table_data)
    if logger is not None:
        logger.write('\n' + table.table + '\n')
        logger.flush()
    print('\n' + table.table)
    return metric_results
