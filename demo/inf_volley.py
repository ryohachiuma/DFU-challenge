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
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    parser.add_argument('--path', type=str, default='/home/ryo/W2020V1_142_NECvsAGE_F8_200119.MP4')
    parser.add_argument('--out-dir', type=str, default='./')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    cap_file = cv2.VideoCapture(args.path)
    out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1280,720))
    count = 0
    while (cap_file.isOpened()):
        print(count)
        if count > 10000:
            break
        count+=1
        ret, frame = cap_file.read()
        if ret:
            result = inference_detector(model, frame)
            result = np.vstack(result[0])
            bboxes = result[result[:, -1] > args.score_thr]
            for j in range(bboxes.shape[0]):
                left_top = (bboxes[j, 0], bboxes[j, 1])
                right_bottom = (bboxes[j, 2], bboxes[j, 3])
                cv2.rectangle(frame, left_top, right_bottom, (0, 0 ,255), thickness=1)
            out.write(frame)
        else:
            break
    cap_file.release()
    out.release()


if __name__ == '__main__':
    main()
