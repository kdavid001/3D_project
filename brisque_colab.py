import cv2
import numpy as np
import math as m
import sys
import os
from scipy.special import gamma as tgamma

# Import svm functions (we expect them in the same folder now)
try:
    import svm
    import svmutil
    from svmutil import svm_load_model, gen_svm_nodearray, svm_predict
    from svm import PRECOMPUTED, ONE_CLASS, EPSILON_SVR, NU_SVC
    import ctypes
except ImportError:
    print("Error: libsvm python files not found.")
    sys.exit(1)


# AGGD fit model
def AGGDfit(structdis):
    poscount = len(structdis[structdis > 0])
    negcount = len(structdis[structdis < 0])

    possqsum = np.sum(np.power(structdis[structdis > 0], 2))
    negsqsum = np.sum(np.power(structdis[structdis < 0], 2))

    abssum = np.sum(structdis[structdis > 0]) + np.sum(-1 * structdis[structdis < 0])

    lsigma_best = np.sqrt((negsqsum / negcount)) if negcount > 0 else 0
    rsigma_best = np.sqrt((possqsum / poscount)) if poscount > 0 else 0

    gammahat = lsigma_best / rsigma_best if rsigma_best > 0 else 0

    totalcount = structdis.shape[1] * structdis.shape[0]

    rhat = m.pow(abssum / totalcount, 2) / ((negsqsum + possqsum) / totalcount)
    rhatnorm = rhat * (m.pow(gammahat, 3) + 1) * (gammahat + 1) / (m.pow(m.pow(gammahat, 2) + 1, 2))

    prevgamma = 0
    prevdiff = 1e10
    sampling = 0.001
    gam = 0.2

    vectfunc = np.vectorize(func, otypes=[np.float64], cache=False)
    gamma_best = vectfunc(gam, prevgamma, prevdiff, sampling, rhatnorm)

    return [lsigma_best, rsigma_best, gamma_best]


def func(gam, prevgamma, prevdiff, sampling, rhatnorm):
    while (gam < 10):
        r_gam = tgamma(2 / gam) * tgamma(2 / gam) / (tgamma(1 / gam) * tgamma(3 / gam))
        diff = abs(r_gam - rhatnorm)
        if (diff > prevdiff): break
        prevdiff = diff
        prevgamma = gam
        gam += sampling
    gamma_best = prevgamma
    return gamma_best


def compute_features(img):
    scalenum = 2
    feat = []
    im_original = img.copy().astype(np.float32)

    for itr_scale in range(scalenum):
        im = im_original.copy()
        im = im / 255.0

        mu = cv2.GaussianBlur(im, (7, 7), 1.166)
        mu_sq = mu * mu
        sigma = cv2.GaussianBlur(im * im, (7, 7), 1.166)
        sigma = (sigma - mu_sq) ** 0.5

        structdis = im - mu
        structdis /= (sigma + 1.0 / 255)

        best_fit_params = AGGDfit(structdis)
        lsigma_best = best_fit_params[0]
        rsigma_best = best_fit_params[1]
        gamma_best = best_fit_params[2]

        feat.append(gamma_best)
        feat.append((lsigma_best * lsigma_best + rsigma_best * rsigma_best) / 2)

        shifts = [[0, 1], [1, 0], [1, 1], [-1, 1]]

        for itr_shift in range(1, len(shifts) + 1):
            OrigArr = structdis
            reqshift = shifts[itr_shift - 1]
            M = np.float32([[1, 0, reqshift[1]], [0, 1, reqshift[0]]])
            ShiftArr = cv2.warpAffine(OrigArr, M, (structdis.shape[1], structdis.shape[0]))

            Shifted_new_structdis = ShiftArr * structdis
            best_fit_params = AGGDfit(Shifted_new_structdis)
            lsigma_best = best_fit_params[0]
            rsigma_best = best_fit_params[1]
            gamma_best = best_fit_params[2]

            constant = m.pow(tgamma(1 / gamma_best), 0.5) / m.pow(tgamma(3 / gamma_best), 0.5)
            meanparam = (rsigma_best - lsigma_best) * (tgamma(2 / gamma_best) / tgamma(1 / gamma_best)) * constant

            feat.append(gamma_best)
            feat.append(meanparam)
            feat.append(m.pow(lsigma_best, 2))
            feat.append(m.pow(rsigma_best, 2))

        im_original = cv2.resize(im_original, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    return feat


def test_measure_BRISQUE(imgPath):
    dis = cv2.imread(imgPath, 1)
    if (dis is None):
        print(f"Error: Could not read image at {imgPath}")
        sys.exit(0)

    dis = cv2.cvtColor(dis, cv2.COLOR_BGR2GRAY)
    features = compute_features(dis)

    x = [0]
    min_ = [0.336999, 0.019667, 0.230000, -0.125959, 0.000167, 0.000616, 0.231000, -0.125873, 0.000165, 0.000600,
            0.241000, -0.128814, 0.000179, 0.000386, 0.243000, -0.133080, 0.000182, 0.000421, 0.436998, 0.016929,
            0.247000, -0.200231, 0.000104, 0.000834, 0.257000, -0.200017, 0.000112, 0.000876, 0.257000, -0.155072,
            0.000112, 0.000356, 0.258000, -0.154374, 0.000117, 0.000351]
    max_ = [9.999411, 0.807472, 1.644021, 0.202917, 0.712384, 0.468672, 1.644021, 0.169548, 0.713132, 0.467896,
            1.553016, 0.101368, 0.687324, 0.533087, 1.554016, 0.101000, 0.689177, 0.533133, 3.639918, 0.800955,
            1.096995, 0.175286, 0.755547, 0.399270, 1.095995, 0.155928, 0.751488, 0.402398, 1.041992, 0.093209,
            0.623516, 0.532925, 1.042992, 0.093714, 0.621958, 0.534484]

    for i in range(0, 36):
        min_val = min_[i]
        max_val = max_[i]
        x.append(-1 + (2.0 / (max_val - min_val) * (features[i] - min_val)))

    model = svmutil.svm_load_model("allmodel")
    x_node, idx = gen_svm_nodearray(x[1:], isKernel=(model.param.kernel_type == PRECOMPUTED))

    nr_classifier = 1
    dec_values = (ctypes.c_double * nr_classifier)()

    svmutil.libsvm.svm_predict_probability(model, x_node, dec_values)
    return dec_values[0]


if len(sys.argv) != 2:
    print("Usage: python brisque_colab.py <image_path>")
    sys.exit(0)

score = test_measure_BRISQUE(sys.argv[1])
print(f"Score of the given image: {score}")