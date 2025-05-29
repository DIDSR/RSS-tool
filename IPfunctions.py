import cv2, os, shutil
import numpy as np
from scipy.spatial.distance import pdist, squareform
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage import measure
from skimage.morphology import square, dilation
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm

def FD_change(Fcode, l, h, sigma):
    Len = Fcode.shape[0]
    # deal with even and odd nums
    if Len % 2:
        n = (Len - 1) // 2
    else:
        n = Len // 2

    # high-freq:h; low-freq:l
    # h >= l, in [0, n]
    # if l = 0: include n (DC)
    # if l > 0: keep no change on n (DC)

    # l = 2; h = 10;

    s_i = list(range(n - h, n - l)) + list(range(n + l, n + h))
    idx = [i + 1 for i in s_i]

    # sigma = 1;

    for i in idx:
        Fcode[i, 0] = Fcode[i, 0] * (1 + np.random.normal(0, sigma))
        Fcode[i, 1] = Fcode[i, 1] * (1 + np.random.normal(0, sigma))

    return Fcode

def find_close_indeces(im_in):
    r, c = np.nonzero(im_in)
    p = pdist(np.column_stack((r, c)))
    psqr = squareform(p)

    nl = len(r)
    new_indices = [(r[0], c[0])]
    ind_ind_data = [0]
    newind = 0

    for ind in range(1, nl):
        mcur_dist = psqr[newind, :]
        dist_min_ind = np.argsort(mcur_dist)
        dist_min_ind = [i for i in dist_min_ind if i not in ind_ind_data]
        newind = dist_min_ind[0]
        new_indices.append((r[newind], c[newind]))
        ind_ind_data.append(newind)

    return new_indices


def bd2Fdesc(mask):
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour based on the area
    largest_contour = max(contours, key=cv2.contourArea)

    # Draw contours on a blank image
    contour_image = np.zeros_like(mask)
    cv2.drawContours(contour_image, [largest_contour], -1, (255, 255, 255), 1)

    new_indeces = find_close_indeces(contour_image)
    idx_arr = np.array(new_indeces)

    ii = idx_arr[:, 0]
    jj = idx_arr[:, 1]
    border_cmplx = ii + 1j * jj
    border_fft = np.fft.fftshift(np.fft.fft(border_cmplx))

    out = np.zeros((len(new_indeces), 2))
    out[:, 0] = np.real(border_fft)
    out[:, 1] = np.imag(border_fft)

    return out


def Fdesc2bd(Fdesc, SIZE):
    len_Fdesc = Fdesc.shape[0]
    ii = Fdesc[:, 0]
    jj = Fdesc[:, 1]
    cmplx = ii + 1j * jj

    border_ifft = np.fft.ifft(np.fft.ifftshift(cmplx))
    border_restored = np.zeros(SIZE, dtype=bool)
    xx = np.round(np.real(border_ifft)).astype(int)
    yy = np.round(np.imag(border_ifft)).astype(int)

    try:
        for i in range(len_Fdesc):
            border_restored[xx[i], yy[i]] = True
    except IndexError:
        border_restored = np.zeros(SIZE, dtype=bool)  # if error, return empty img -> sum(img(:))==0

    return border_restored


def imbinarize(img):
    # binarize area image
    if img.ndim != 2:
        img = rgb2gray(img)

    if not np.issubdtype(img.dtype, np.bool_):
        thresh = threshold_otsu(img)
        mask = img > thresh
    else:
        mask = img

    return np.asarray(mask, dtype="uint8")  # for cv2 use



def contour2mask(contour):
    # Define the structuring element: square(2)
    # Perform dilation
    eligible_flag = False  # flag for mask is closed and has one connected component
    pN_contour = np.count_nonzero(contour)
    # print("pN_contour: ", pN_contour)
    if pN_contour==0: # empty, no pixel
        return contour, eligible_flag

    for k in range(1, 4):  # apply k = 1, 2, or 3
        dilated = dilation(contour, square(k))  # k=1 is no dilation
        pN_dilated =  np.count_nonzero(dilated)
        # print("pN_dilation - contour: ", pN_dilated - pN_contour)

        # Fill holes
        filled = binary_fill_holes(dilated)
        pN_filled = np.count_nonzero(filled)

        # print("k= ", k, " is closed?", (pN_filled - pN_dilated)>pN_dilated)
        if (pN_filled - pN_dilated)>pN_dilated:  # is closed
            _, num_components = measure.label(filled, connectivity=2, return_num=True)
            if num_components == 1:  # has only one connected component
                eligible_flag = True
                break


    return filled, eligible_flag


def mask_augmentation(mask_folder, output_folder, times, l, h, sigma, rndSeed):

    if rndSeed!=-1:
        np.random.seed(rndSeed)
    else:
        np.random.seed(None)

    for mask_file in tqdm(os.listdir(mask_folder)):

        name, ext = os.path.splitext(mask_file)
        folder_name = f"{name}_syn"
        folder_path = os.path.join(output_folder, folder_name)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        else:
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)

        mask_path = os.path.join(mask_folder, mask_file)
        one_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        one_mask = imbinarize(one_mask)  # binarize area image
        
        inner_sigma = sigma

        for i in range(1,times+1):  # gen num SynSegs
            try_times = 0
            
            while True:
                fd_code = bd2Fdesc(one_mask)
                CFcode = FD_change(fd_code, l, h, inner_sigma)

                re_img = Fdesc2bd(CFcode, one_mask.shape)
                syn_img, eligible = contour2mask(re_img)



                if eligible:
                    new_name = f"{name}_syn_{i}{ext}"
                    new_aug_path = os.path.join(output_folder, folder_name, new_name)
                    cv2.imwrite(new_aug_path, syn_img * 255)

                    break

                else:
                    try_times +=1
                    if try_times>99:  # try 100 times

                        if inner_sigma < 0.0001:
                            print("Task unsuccessful: cannot create an eligible mask for " + str(try_times)
                                  + " times trying in a least one round, even using the sigma: " + str(inner_sigma))
                            print("Input mask filename: " + mask_file)
                            return

                        inner_sigma = inner_sigma / 2
                        print("Cannot create an eligible mask for " + str(try_times)
                              + " times trying in a least one round, trying to set a smaller sigma: " + str(inner_sigma) + "\t")
                        try_times = 0


                    print("Cannot create an eligible mask, tried " + str(try_times) + "/100 times.", end="\r", flush=True)
