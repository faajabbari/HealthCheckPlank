import argparse
import os
import glob

import openpifpaf
import cv2

from plank import PlankCorrectness


parser = argparse.ArgumentParser()
parser.add_argument('--path', default='realWorld/fat2.MOV', help='path to photo')
parser.add_argument('--save_path', default='res_test', help='path to save photo')
parser.add_argument('--frame_rate', default=10, help='video frame rate')

args = parser.parse_args()


def main():
    os.makedirs(args.save_path, exist_ok=True)
    plank_check = PlankCorrectness()
    predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k30')
    cap = cv2.VideoCapture(args.path)
    counter = -1
    while cap.isOpened():
        counter += 1
        ret, frame = cap.read()
        if ret:
            if counter % args.frame_rate == 0:
                pil_im, dim = plank_check.resize(frame)
                predictions, _, _ = predictor.pil_image(pil_im)
                plank_check.is_correct(predictions, counter)

        else:
            cap.release()
            cv2.destroyAllWindows()
            img_array = []
            for filename in sorted(glob.glob(os.path.join(args.save_path, '*.png'))):
                img = cv2.imread(filename)
                height, width, layers = img.shape
                size = (width, height)
                img_array.append(img)

            out = cv2.VideoWriter('results.avi', cv2.VideoWriter_fourcc(*'DIVX'), 3, size)

            for i in range(len(img_array)):
                out.write(img_array[i])
            out.release()





if __name__ == '__main__':
    main()
    # overall_risk, n_risks = main()
    # print(overall_risk)
    # print(n_risks)
