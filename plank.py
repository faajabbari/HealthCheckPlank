import math
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


class PlankCorrectness:
    def __init__(self):
        self.nose = 0
        self.lEye = 1
        self.rEye = 2
        self.lEar = 3
        self.rEar = 4
        self.lShoulder = 5
        self.rShoulder = 6
        self.lElbow = 7
        self.rElbow = 8
        self.lWrist = 9
        self.rWirst = 10
        self.lHip = 11
        self.rHip = 12
        self.lKnee = 13
        self.rKnee = 14
        self.lAnkle = 15
        self.rAnkle = 16

    def resize(self, image):
        r = 400.0 / image.shape[0]
        dim = (int(image.shape[1] * r), 400)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        self.image = Image.fromarray(image, 'RGB')
        return self.image, dim

    def calc_display_joints(self, prediction, counter, is_plot=True):
        xx = []
        yy = []
        zz = []
        for p in range(len(prediction)):
            for i in range(17):
                x = prediction[p].data[i][0]
                y = prediction[p].data[i][1]
                z = prediction[p].data[i][1]
                if x > 0 and y > 0:
                    xx.append(x)
                    yy.append(y)
                    zz.append(z)
            if is_plot:
                # self.image = self.image.copy()
                img = np.array(self.image)[:, :, ::-1]
                plt.imshow(img)
                plt.scatter(xx, yy)
                plt.xticks([])
                plt.yticks([])
                plt.savefig(f'res_test/{counter}.png')
                # plt.title('Joints')

    def calc_keypoints(self, predictions):
        pred = []
        for i in predictions:
            pred.append(i.json_data())
        for i in range(0, len(pred)):
            keypoints = [pred[i]['keypoints'][s:s + 3:] for s in range(0, len(pred[i]['keypoints']), 3)]
            self.keypoints = np.array(keypoints)

    @staticmethod
    def get_angle_between_degs(v1, v2):
        len_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        len_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
        result = math.acos(round(np.dot(v1, v2) / (len_v1 * len_v2), 5)) * 180 / math.pi

        return result

    @staticmethod
    def plot_vector(p1, p2, p3, p4, im, theta, i, txt='', x_text=0.1, y_text=0.9):
        x1 = p1[0]
        x2 = p2[0]
        x3 = p3[0]
        x4 = p4[0]
        y1 = p1[1]
        y2 = p2[1]
        y3 = p3[1]
        y4 = p4[1]
        xx = [x1, x2, x3, x4]
        yy = [y1, y2, y3, y4]
        # plt.sca(axs.flatten()[i])
        im = np.array(im)[:, :, ::-1]
        plt.imshow(im)
        plt.scatter(xx, yy)
        plt.plot([x1, x2], [y1, y2])
        plt.plot([x3, x4], [y3, y4])
        main_text = str(str(txt) + str('%.2f' % theta))
        # plt.text(x_text, y_text, main_text, color='purple', fontsize=7)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f'res_test/{i}.png')

    def compute_theta(self, p3, p4):
        v2 = np.array(p3) - np.array(p4)
        p1 = [p3[0], p3[1]]
        p2 = [p3[0], p3[1] - 200]
        v1 = np.array(p1) - np.array(p2)

        theta = self.get_angle_between_degs(v1, v2)
        # plot_vector(p1, p2, p3, p4, im, theta)
        print(theta)

        return theta, p1, p2, p3, p4

    def find_arms_angle(self, counter):
        thetaLHipShoulserElbow = None
        thetaLElboeShoulderWrist = None
        thetaRHipShoulderElbow = None
        thetaRShoulderElbowWrist = None

        if (self.keypoints[self.lHip][2] > 0 and self.keypoints[self.lShoulder][2] > 0 and self.keypoints[self.lElbow][
            2] > 0) or \
                (self.keypoints[self.rHip][2] > 0 and self.keypoints[self.rShoulder][2] > 0 and
                 self.keypoints[self.rElbow][2] > 0):

            if self.keypoints[self.lHip][2] > 0 and self.keypoints[self.lShoulder][2] > 0 and \
                    self.keypoints[self.lElbow][2] > 0:
                v2 = np.array(self.keypoints[self.lHip][0:2]) - np.array(self.keypoints[self.lShoulder][0:2])
                v1 = np.array(np.array(self.keypoints[self.lElbow][0:2] - self.keypoints[self.lShoulder][0:2]))
                thetaLHipShoulserElbow = self.get_angle_between_degs(v1, v2)
                print('left hand angle is: ', thetaLHipShoulserElbow)
                self.plot_vector(self.keypoints[self.lHip][0:2], self.keypoints[self.lShoulder][0:2],
                                 self.keypoints[self.lElbow][0:2],
                                 self.keypoints[self.lShoulder][0:2],
                                 self.image,
                                 thetaLHipShoulserElbow, counter, txt='left hand angle', x_text=5, y_text=20)
                if self.keypoints[self.lWrist][2] > 0:
                    v2 = np.array(self.keypoints[self.lShoulder][0:2]) - np.array(self.keypoints[self.lElbow][0:2])
                    v1 = np.array(self.keypoints[self.lWrist][0:2]) - np.array(self.keypoints[self.lElbow][0:2])
                    thetaLElboeShoulderWrist = self.get_angle_between_degs(v1, v2)

                    self.plot_vector(self.keypoints[self.lShoulder][0:2], self.keypoints[self.lElbow][0:2],
                                     self.keypoints[self.lWrist][0:2], self.keypoints[self.lElbow][0:2],
                                     self.image, thetaLElboeShoulderWrist, counter, txt='left hand elbow', x_text=5,
                                     y_text=40)
                    print('left hand side elbow flexion angle: ', thetaLElboeShoulderWrist)

            if self.keypoints[self.rHip][2] > 0 and self.keypoints[self.rShoulder][2] > 0 and \
                    self.keypoints[self.rElbow][2] > 0:
                v2 = np.array(self.keypoints[self.rHip][0:2]) - np.array(self.keypoints[self.rShoulder][0:2])
                v1 = np.array(np.array(self.keypoints[self.rElbow][0:2] - self.keypoints[self.rShoulder][0:2]))
                thetaRHipShoulderElbow = self.get_angle_between_degs(v1, v2)
                self.plot_vector(self.keypoints[self.rHip][0:2], self.keypoints[self.rShoulder][0:2],
                                 self.keypoints[self.rElbow][0:2],
                                 self.keypoints[self.rShoulder][0:2],
                                 self.image, thetaRHipShoulderElbow, counter, txt='right hand angle', x_text=5,
                                 y_text=60)
                print('right hand angle is: ', thetaRHipShoulderElbow)
                if self.keypoints[self.rWirst][2] > 0:
                    v2 = np.array(self.keypoints[self.rShoulder][0:2]) - np.array(self.keypoints[self.rElbow][0:2])
                    v1 = np.array(self.keypoints[self.rWirst][0:2]) - np.array(self.keypoints[self.rElbow][0:2])
                    thetaRShoulderElbowWrist = self.get_angle_between_degs(v1, v2)

                    self.plot_vector(self.keypoints[self.rShoulder][0:2], self.keypoints[self.rElbow][0:2],
                                     self.keypoints[self.rWirst][0:2], self.keypoints[self.rElbow][0:2],
                                     self.image, thetaRShoulderElbowWrist, counter, txt='right hand elbow', x_text=5,
                                     y_text=80)
                    print('right hand side elbow flexion angle: ', thetaRShoulderElbowWrist)

        return thetaLHipShoulserElbow, thetaLElboeShoulderWrist, thetaRHipShoulderElbow, thetaRShoulderElbowWrist

    def find_back_angle(self, counter):
        thetaLKneeHipShoulser = None
        thetaRKneeHipShoulder = None
        if (self.keypoints[self.lHip][2] > 0.0 and self.keypoints[self.lShoulder][2] > 0.0 and
            self.keypoints[self.lKnee][2]) or (
                self.keypoints[self.rHip][2] > 0.0 and self.keypoints[self.rShoulder][2] > 0.0 and
                self.keypoints[self.rKnee][2]):

            if self.keypoints[self.lHip][2] > 0 and self.keypoints[self.lShoulder][2] > 0 and \
                    self.keypoints[self.lKnee][2] > 0:
                v2 = np.array(self.keypoints[self.lShoulder][0:2]) - np.array(self.keypoints[self.lHip][0:2])
                v1 = np.array(np.array(self.keypoints[self.lKnee][0:2] - self.keypoints[self.lHip][0:2]))
                thetaLKneeHipShoulser = self.get_angle_between_degs(v1, v2)
                print('left back angle is: ', thetaLKneeHipShoulser)
                self.plot_vector(self.keypoints[self.lHip][0:2], self.keypoints[self.lKnee][0:2],
                                 self.keypoints[self.lHip][0:2],
                                 self.keypoints[self.lShoulder][0:2],
                                 self.image,
                                 thetaLKneeHipShoulser, counter, txt='left hand angle', x_text=5, y_text=100)

            if self.keypoints[self.rHip][2] > 0 and self.keypoints[self.rShoulder][2] > 0 and \
                    self.keypoints[self.rKnee][2] > 0:
                v2 = np.array(self.keypoints[self.rShoulder][0:2]) - np.array(self.keypoints[self.rHip][0:2])
                v1 = np.array(np.array(self.keypoints[self.rKnee][0:2] - self.keypoints[self.rHip][0:2]))
                thetaRKneeHipShoulder = self.get_angle_between_degs(v1, v2)
                print('left back angle is: ', thetaRKneeHipShoulder)
                self.plot_vector(self.keypoints[self.rHip][0:2], self.keypoints[self.rKnee][0:2],
                                 self.keypoints[self.rHip][0:2],
                                 self.keypoints[self.rShoulder][0:2],
                                 self.image,
                                 thetaRKneeHipShoulder, counter, txt='left hand angle', x_text=5, y_text=120)

        return thetaLKneeHipShoulser, thetaRKneeHipShoulder

    def calculate_all_angles(self, predictions, counter):
        self.calc_display_joints(predictions, counter)
        self.calc_keypoints(predictions)
        thetaLHipShoulserElbow, thetaLElboeShoulderWrist, thetaRHipShoulderElbow, thetaRShoulderElbowWrist = \
            self.find_arms_angle(counter)
        thetaLKneeHipShoulser, thetaRKneeHipShoulder = self.find_back_angle(counter)
        return thetaLHipShoulserElbow, thetaLElboeShoulderWrist, thetaRHipShoulderElbow, thetaRShoulderElbowWrist,\
            thetaLKneeHipShoulser, thetaRKneeHipShoulder


    def is_correct(self, predictions, counter):
        hl, _, hr, _, fl, fr = self.calculate_all_angles(predictions, counter)
        if hl and hr:
            if 65 < hl< 90 and 65 < hr< 85:
                plt.text(10, 30, 'Hand: Correct', color='green', fontsize=20,)
                plt.savefig(f'res_test/{counter}.png')
            else:
                plt.text(10, 30, 'Hand: Incorrect', color='red', fontsize=20)
                plt.savefig(f'res_test/{counter}.png')
        elif hl:
            if 65 < hl < 90:
                plt.text(10, 40, 'Hand: Correct', color='green', fontsize=20, )
                plt.savefig(f'res_test/{counter}.png')
            else:
                plt.text(10, 40, 'Hand: Incorrect', color='red', fontsize=20)
                plt.savefig(f'res_test/{counter}.png')
        elif hr:
            if 65 < hr < 90:
                plt.text(10, 40, 'Hand: Correct', color='green', fontsize=20, )
                plt.savefig(f'res_test/{counter}.png')
            else:
                plt.text(10, 40, 'Hand: Incorrect', color='red', fontsize=20)
                plt.savefig(f'res_test/{counter}.png')
        else:
            pass

        if fl and fr:
            if 160 < fl < 185 and 160 < hr< 185:
                plt.text(10, 80, 'Back: Correct', color='green', fontsize=20,)
                plt.savefig(f'res_test/{counter}.png')
            else:
                plt.text(10, 80, 'Back: Incorrect', color='red', fontsize=20)
                plt.savefig(f'res_test/{counter}.png')
        elif fl:
            if 160 < fl < 180:
                plt.text(10, 80, 'Back: Correct', color='green', fontsize=20, )
                plt.savefig(f'res_test/{counter}.png')
            else:
                plt.text(10, 80, 'Back: Incorrect', color='red', fontsize=20)
                plt.savefig(f'res_test/{counter}.png')
        elif fr:
            if 160 < fr < 185:
                plt.text(10, 80, 'Back: Correct', color='green', fontsize=20, )
                plt.savefig(f'res_test/{counter}.png')
            else:
                plt.text(10, 80, 'Back: Incorrect', color='red', fontsize=20)
                plt.savefig(f'res_test/{counter}.png')
        else:
            pass

        plt.close()




    def creat_result_video(self):
        pass
