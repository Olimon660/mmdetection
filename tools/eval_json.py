from argparse import ArgumentParser

import mmcv
import numpy as np

from mmdet import datasets
from mmdet.core import eval_map

import json
import glob

id_to_class = {1: 'Candida',
               2: 'ASC-US',
               3: 'LSIL',
               4: 'HSIL',
               5: 'ASC-H',
               6: 'Trichomonas'}
class_to_id = {'Candida': 1,
               'ASC-US': 2,
               'LSIL': 3,
               'HSIL': 4,
               'ASC-H': 5,
               'Trichomonas': 6}


def voc_eval_json(result_dir, gt_dir, iou_thr=0.5):
    result_files = sorted(glob.glob(f'{result_dir}/*.json'))
    det_results_raw = [json.load(open(fp, 'r')) for fp in result_files]
    det_results = []
    for i in range(len(det_results_raw)):
        det_f = []
        det_current_file = det_results_raw[i]  # det_current_file = list of dicts
        for c in range(1, len(class_to_id) + 1):
            det_c = np.array([np.array([int(pred['x']), int(pred['y']), (int(pred['x']) + int(pred['w'])),
                                        (int(pred['y']) + int(pred['h'])), float(pred['p'])]) for pred in
                              det_current_file if
                              class_to_id[pred['class']] == c])
            if det_c.size == 0:
                det_f.append(np.array([]).reshape(0,5))
            else:
                det_f.append(det_c)

        det_results.append(det_f)

    gt_files = sorted(glob.glob(f'{gt_dir}/*.json'))
    gt_raw = [json.load(open(fp, 'r')) for fp in gt_files]
    gt_bboxes = []
    gt_labels = []
    for i in range(len(gt_raw)):
        bboxes = np.array([[int(gt['x']), int(gt['y']), (int(gt['x']) + int(gt['w'])),
                            (int(gt['y']) + int(gt['h']))] for gt in gt_raw[i] if gt['class'] != 'roi'])
        labels = np.array([class_to_id[gt['class']] for gt in gt_raw[i] if gt['class'] != 'roi'])

        gt_bboxes.append(bboxes)
        gt_labels.append(labels)

    gt_ignore = None
    dataset_name = list(class_to_id.keys())
    eval_map(
        det_results,
        gt_bboxes,
        gt_labels,
        gt_ignore=gt_ignore,
        scale_ranges=None,
        iou_thr=iou_thr,
        dataset=dataset_name,
        print_summary=True)


def main():
    parser = ArgumentParser(description='VOC Evaluation')
    parser.add_argument('result_dir', help='result file path')
    parser.add_argument('gt_dir', help='config file path')
    parser.add_argument(
        '--iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold for evaluation')
    args = parser.parse_args()
    voc_eval_json(args.result_dir.strip('/'), args.gt_dir.strip('/'), args.iou_thr)


if __name__ == '__main__':
    main()
