from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot, save_result
import os
import mmcv
import numpy as np
import cv2
def main():
    parser = ArgumentParser()
    #parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.7, help='bbox score threshold')
    parser.add_argument('--out-dir', type=str, default='./')
    parser.add_argument('--save-img', action='store_true')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image

    root_dir = '/home/ryo/ssd1/ryo/DFU/validation/'

    files = os.listdir(root_dir)
    files.sort()
    output_dir = args.out_dir
    output_file = []
    output_file.append('filename,xmin,ymin,xmax,ymax,score')
    for file in files:
        result = inference_detector(model, os.path.join(root_dir, file))
        if len(result[0]) == 0:
            continue
        result = np.vstack(result[0])
        #print(result[:, -1])
        bboxes = result[result[:, -1] > args.score_thr]
        if args.save_img:
            img = cv2.imread(os.path.join(root_dir, file))
        for j in range(bboxes.shape[0]):
            output_bbox = '%s,%d,%d,%d,%d,%.4f'% (file, bboxes[j][0], bboxes[j, 1], bboxes[j, 2], bboxes[j, 3], bboxes[j, 4]) 
            output_file.append(output_bbox)
            left_top = (bboxes[j, 0], bboxes[j, 1])
            right_bottom = (bboxes[j, 2], bboxes[j, 3])
            if args.save_img:
                cv2.rectangle(img, left_top, right_bottom, (0, 0 ,255), thickness=1)
        if args.save_img:
            cv2.imwrite(os.path.join(output_dir, file), img)
    np.savetxt(os.path.join(output_dir, 'inference.csv'), np.asarray(output_file), fmt='%s')

if __name__ == '__main__':
    main()
