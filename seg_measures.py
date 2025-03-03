from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import directed_hausdorff
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

# Functions are used for segmentation
# performance evaluation of user’s inputs (truth masks and segmentation results)

# input: Two binary masks {0, 1} in 2D, same size



# Segmentation Confusion Matrix
def SegConfusionMatrix(truth_mask, seg):
    # tn, fp, fn, tp = SegConfusionMatrix(truth_mask, seg)
    # input two binary 2D np.array
    # Flatten a 2d numpy array into 1d array
    truth_mask = np.array(truth_mask).flatten()
    seg = np.array(seg).flatten()
    # ref: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    return confusion_matrix(y_true=truth_mask, y_pred=seg).ravel()


# Dice Coefficient
def DICE(truth_mask, seg):
    # input two binary 2D np.array
    tn, fp, fn, tp = SegConfusionMatrix(truth_mask, seg)
    return 2*tp/(2*tp+fp+fn)


# Jaccard Coefficient
def JAC(truth_mask, seg):
    # input two binary 2D np.array
    tn, fp, fn, tp = SegConfusionMatrix(truth_mask, seg)
    return tp/(tp+fp+fn)

# Hausdorff Distance
def HD(truth_mask, seg):
    # input two 2D np.array
    try:
        truth_mask = np.array(truth_mask)
        seg = np.array(seg)

        A = np.column_stack(np.where(truth_mask != 0))
        B = np.column_stack(np.where(seg != 0))

        if A.shape[1] != B.shape[1]:
            print('WARNING: dimensionality must be the same')
            return None

    # Compute the directed Hausdorff distance between two 2-D arrays
    # ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.directed_hausdorff.html
        return max(directed_hausdorff(A, B)[0], directed_hausdorff(B, A)[0])

    except Exception as e:
        print(e)
        return np.nan

def SegCompare_1vN(truth_file, seg_folder, metric_func):
    # Compare one truth_mask vs many seg_masks by using the selected metric
    # Input: the file of truth_mask, the folder includes seg_masks, the comparing metric
    truth_mask = cv2.imread(truth_file, cv2.IMREAD_GRAYSCALE)
    output = []

    for seg_file in tqdm(os.listdir(seg_folder)):
        seg_path = os.path.join(seg_folder, seg_file)
        seg_mask = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)

        output.append(metric_func(truth_mask, seg_mask))

    return output

def Plot_res(res,xlabel):
    num = len(res)
    if num//10 > 8:
        n_bins = 4*((num//10)//4)  # n_bins is the times of 4
    else:
        n_bins = 8

    _, axs = plt.subplots(2, 1, tight_layout=True)

    axs[0].hist(res, bins=n_bins)
    axs[0].set_title('Histogram')
    axs[1].boxplot(res,notch=1, vert=0)  # notched plot, horizontal boxes

    plt.xlabel(xlabel)
    plt.show()





def BLD_idx(ref, test):
    dtR, R_idx = d_min(test, ref)  # index in ref
    drT, T_idx = d_min(ref, test)  # index in test

    # the length of idx of T's pairing points in Ref is # of points in test
    BLD_R_idx = np.zeros(len(test), dtype=int)

    for i in range(len(test)):
        FMinD = dtR[i]
        FMinD_idx = R_idx[i]  # index in ref

        BMaxD_idx = np.where(T_idx == i)[0]  # index in ref
        BMaxD = drT[T_idx == i]

        BLD_Candidate = np.array([FMinD, *BMaxD])
        BLD_idx_Candidate = np.array([FMinD_idx, *BMaxD_idx])  # index in ref

        BLD_idx_Candidate_idx = np.argmax(BLD_Candidate)

        # BLD_R_idx[i] is the max index in ref; i is the index in test
        BLD_R_idx[i] = BLD_idx_Candidate[BLD_idx_Candidate_idx]

    return BLD_R_idx


def d_min(T, R):
    N_t = T.shape[0]  # # of points in T

    d = np.zeros(N_t)
    d_idx = np.zeros(N_t, dtype=int)

    for k in range(N_t):
        dist = np.sqrt(np.sum((R - T[k, :]) ** 2, axis=1))
        d[k], d_idx[k] = np.min(dist), np.argmin(dist)  # idx in R

    return d, d_idx


def MSI(ref_img, test_img, il=1, ol=1):
    # try:
    b_ref, mask_ref = img2boundary(test_img)
    b_test, _ = img2boundary(ref_img)

    COM_ref = np.round(np.mean(b_ref, axis=0))
    COM_test = np.round(np.mean(b_test, axis=0))

    mv_r = COM_ref[0] - COM_test[0]
    mv_c = COM_ref[1] - COM_test[1]

    new_b_test = np.zeros_like(b_test)
    new_b_test[:, 0] = b_test[:, 0] + mv_r
    new_b_test[:, 1] = b_test[:, 1] + mv_c



    BLD_R_idx = BLD_idx(b_ref, new_b_test)

    N = len(BLD_R_idx)
    LDP = np.zeros(N)
    MCF = np.zeros(N)

    for i in range(len(BLD_R_idx)):
        d = np.sqrt((b_ref[BLD_R_idx[i], 0] - b_test[i, 0]) ** 2 + (b_ref[BLD_R_idx[i], 1] - b_test[i, 1]) ** 2)
        if mask_ref[b_test[i, 0], b_test[i, 1]] == 0:  # outside
            LDP[i] = d
            MCF[i] = WF(d, ol)
        else:  # inside
            LDP[i] = -d
            MCF[i] = WF(d, il)

    MSI_value = np.mean(MCF)
    # except Exception as e:
    #     print(e)
    #     MSI_value = np.nan

    return MSI_value


def img2boundary(img):
    from IPfunctions import find_close_indeces, imbinarize

    mask = imbinarize(img)  # binarize area image

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on a blank image
    contour_image = np.zeros_like(mask)
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)
    b = find_close_indeces(contour_image)

    return np.array(b), mask


def WF(x, l):
    return np.exp(-(l * x) ** 2 / 200)
