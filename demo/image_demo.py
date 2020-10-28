from argparse import ArgumentParser
from mmdet.apis import inference_detector, init_detector
import numpy as np
import cv2

def main():
    parser = ArgumentParser()
    parser.add_argument('img_path', help='Path to the Image file')
    parser.add_argument('config', help='path to the Config file')
    parser.add_argument('checkpoint', help='path to the Checkpoint file')
    parser.add_argument('--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()

    model = init_detector(args.config, args.checkpoint, device='cuda:0')
    result = inference_detector(model, args.img_path)[0]
    img = cv2.imread(args.img_path)
    for k in range(result.shape[0]):
        left_top = (result[k, 0], result[k, 1])
        right_bottom = (result[k, 2], result[k, 3])
        conf = result[k, 4]
        if conf > args.score_thr:
            cv2.rectangle(img, left_top, right_bottom, (255, 0 ,0), thickness=3)

    cv2.imshow("result", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
