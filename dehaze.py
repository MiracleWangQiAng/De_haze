# -*- coding: utf-8 -*-
import cv2
import math
import numpy as np
import argparse
import os
from skimage.metrics import structural_similarity as sk_cpt_ssim


def PSNR(target, ref):
    target = np.uint8(target)
    #必须归一化
    target = target/255.0
    ref = ref/255.0
    MSE = np.mean((target-ref)**2)
    if MSE<1e-10:
        return 100
    MAXI=1
    PSNR = 20*math.log10(MAXI/math.sqrt(MSE))
    return PSNR


def dark_channel(img, size = 15):
    r, g, b = cv2.split(img)
    min_img = cv2.min(r, cv2.min(g, b))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dc_img = cv2.erode(min_img, kernel)
    return dc_img


def get_A(img, percent = 0.001):
    mean_perpix = np.mean(img, axis = 2).reshape(-1)
    mean_topper = mean_perpix[:int(img.shape[0] * img.shape[1] * percent)]
    return np.mean(mean_topper)


def get_t(img, atom, w = 0.95):
    x = img / atom
    t = 1 - w * dark_channel(x, 15)
    return t


def guided_filter(p, i, r, e):
    """
    :param p: input image
    :param i: guidance image
    :param r: radius
    :param e: regularization
    :return: filtering output q
    """
    #1 用拉普拉斯算子对输入图和引导图进行锐化，使其细节更明显
    mean_I = cv2.boxFilter(i, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    corr_I = cv2.boxFilter(i * i, cv2.CV_64F, (r, r))
    corr_Ip = cv2.boxFilter(i * p, cv2.CV_64F, (r, r))
    #2
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    #3
    a = cov_Ip / (var_I + e)
    b = mean_p - a * mean_I
    #4
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    #5
    q = mean_a * i + mean_b
    return q


def dehaze(path, filename, output):
    Path = os.path.join(path, filename)
    im = cv2.imread(Path)
    img = im.astype('float64') / 255
    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype('float64') / 255
    img_dc = dark_channel(img, 15)

    atom = get_A(img)
    trans = get_t(img, atom)
    trans_guided = guided_filter(trans, img_gray, 20, 0.0001)
    trans_guided = cv2.max(trans_guided, 0.25)

    result = np.empty_like(img)
    for i in range(3):
        result[:, :, i] = (img[:, :, i] - atom) / trans_guided + atom

    #cv2.imshow("source", img)
    #cv2.imshow("dark_channel", img_dc)
    #cv2.imshow("result", result)

    imgs = np.hstack([img, result])
    cv2.imshow("compare", imgs)
    cv2.waitKey()
    if output is not None:
        print(output)
        dc_Save = os.path.join(output, 'canon3dc.png')
        resulet_Save = os.path.join(output, filename)
        cv2.imwrite(dc_Save, img_dc * 255)
        cv2.imwrite(resulet_Save, imgs * 255)

    print("PSRN：", PSNR(result*255, img*255,), "\n")
    print("SSIM：", sk_cpt_ssim(result*255, img*255, win_size=11, data_range=255, channel_axis=2))


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input')
parser.add_argument('-o', '--output')
args = parser.parse_args()


if __name__ == '__main__':
    if args.input is None:
        dehaze('image', 'canon3.bmp', 'image/result')
    else:
        dehaze(args.input, args.output)
