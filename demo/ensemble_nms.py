# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


import cv2
import numpy as np
from ensemble_boxes import *
import os


if __name__ == '__main__':
    draw_image = True


    model_cfgs = ['feet_low', 'cascade', 'pisa', 'deform_low', 'htc']
    print(model_cfgs)
    save_dir = './work_dirs'
    img_w = 640
    img_h = 480
    results = []
    confidences = []
    bboxes = []
    labels = []
    weights = [1] * len(model_cfgs)
    iou_thr = 0.5

    names = []

    for model in model_cfgs:
        bbox_path = os.path.join(save_dir, model, 'inference.csv')
        data = np.genfromtxt(bbox_path, dtype=str, delimiter=',', skip_header=1)
        filenames = data[:, 0]
        bbox = data[:, 1:].astype(np.float32)
        bbox[:, 0] = bbox[:, 0] / 640.0
        bbox[:, 2] = bbox[:, 2] / 640.0
        bbox[:, 1] = bbox[:, 1] / 480.0
        bbox[:, 3] = bbox[:, 3] / 480.0
        result = {}
        for i in range(bbox.shape[0]):
            if filenames[i] in result:
                result[filenames[i]] = np.append(result[filenames[i]], bbox[np.newaxis, i, :], axis=0)
            else:
                result[filenames[i]] = bbox[np.newaxis, i, :]
                names.append(filenames[i])
        results.append(result)

    names = list(set(names))
    names.sort()
    output_file = []
    output_file.append('filename,xmin,ymin,xmax,ymax,score')
    for name in names:
        bboxes = []
        confs = []
        labels = []
        for i in range(len(results)):
            if not name in results[i]:
                bb = np.array([[0.0, 0.0, 1.0, 1.0]])
                c = np.array([0.0001])
            else:
                bb = results[i][name][:, :-1]
                c = results[i][name][:, -1]

            bboxes.append(bb.tolist())
            confs.append(c.tolist())
            labels.append([0] * len(c))

        boxes, scores, labels = weighted_boxes_fusion(bboxes, confs, labels, weights=weights, iou_thr=iou_thr, skip_box_thr=0.3)

        for j in range(boxes.shape[0]):
            if scores[j] > 0.7:
                output_bbox = '%s,%d,%d,%d,%d,%.4f'% (name, boxes[j][0] * 640, boxes[j][1] * 480, boxes[j][2] * 640, boxes[j][3] * 480, scores[j]) 
                output_file.append(output_bbox)

    np.savetxt(os.path.join('./inference.csv'), np.asarray(output_file), fmt='%s')

