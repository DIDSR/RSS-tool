from scipy.ndimage import distance_transform_edt
import numpy as np
import os
from PIL import Image
import SimpleITK as sitk

def img2bi(img_loc):
    img = Image.open(img_loc).convert('1')
    img = img.resize((360,360))
    img = np.array(img)
    img = np.int64(img)

    return np.where(img > 0, 1, 0)

# Function to create a signed distance map
def signed_distance_transform(binary_region):
    # Compute the distance transform for inside and outside regions
    inside_dist = distance_transform_edt(binary_region)
    outside_dist = distance_transform_edt(1 - binary_region)

    # Create signed distance map
    signed_dist = inside_dist - outside_dist
    return signed_dist


# Function to create a weighted label map
def create_weighted_label_map(signed_dist):
    inner_core = (signed_dist > 2).astype(int)
    inside_border = ((signed_dist > 0) & (signed_dist <= 2)).astype(int)
    outside_border = ((signed_dist < 0) & (signed_dist >= -2)).astype(int)
    outer_space = (signed_dist < -2).astype(int)

    weighted_map = 4 * inner_core + 3 * inside_border + 2 * outside_border + 1 * outer_space
    return weighted_map


# Function to combine label maps using center of gravity method
def combine_label_maps(label_maps):
    combined_map = np.mean(label_maps, axis=0)
    final_map = (combined_map >= 2.5).astype(int)  # Threshold to get final binary estimate
    return final_map


# Example usage
# reader_markings = [np.random.randint(0, 2, (100, 100)) for _ in range(4)]

def Fusion_TESD(MaskFolder, output_folder=""):

    reader_markings = []
    for mask_file in os.listdir(MaskFolder):
        mask_path = os.path.join(MaskFolder, mask_file)
        reader_markings.append(img2bi(mask_path))

    distance_maps = [signed_distance_transform(marking) for marking in reader_markings]
    weighted_maps = [create_weighted_label_map(dist_map) for dist_map in distance_maps]

    # Combine weighted label maps to get the final binary estimate
    final_estimate = combine_label_maps(weighted_maps)

    # Output the final binary estimate
    # print(final_estimate.shape)

    #


    im = Image.fromarray((final_estimate * 255).astype(np.uint8), 'L')
    im.save(os.path.join(output_folder,'FusedMask_TESD.png'))
    # print("The fused mask saved.")

def Fusion_MV(MaskFolder, output_folder=""):
    mask_stack = []
    for mask_file in os.listdir(MaskFolder):
        mask_path = os.path.join(MaskFolder, mask_file)
        mask_stack.append(img2bi(mask_path))


    mv_mask = ((sum(mask_stack)/len(mask_stack)) > 0.5).astype(int)
    im = Image.fromarray((mv_mask*255).astype(np.uint8), 'L')
    im.save(os.path.join(output_folder, 'FusedMask_MV.png'))
    # print("The fused mask saved.")

def Fusion_MEAN(MaskFolder, output_folder=""):
    mask_stack = []
    for mask_file in os.listdir(MaskFolder):
        mask_path = os.path.join(MaskFolder, mask_file)
        mask_stack.append(img2bi(mask_path))


    Mean_mask = np.round((sum(mask_stack)/len(mask_stack))*255)

    im = Image.fromarray(Mean_mask.astype(np.uint8), 'L')
    im.save(os.path.join(output_folder, 'FusedMask_MEAN.png'))
    # print("The fused mask saved.")


def Fusion_STAPLE(MaskFolder, output_folder=""):
    mask_stack = []
    for mask_file in os.listdir(MaskFolder):
        mask_path = os.path.join(MaskFolder, mask_file)
        mask_stack.append(sitk.GetImageFromArray(img2bi(mask_path)))


    STAPLE_mask_stack = sitk.STAPLE(mask_stack, 1.0)
    STAPLE_mask_stack = STAPLE_mask_stack > 0.5  # Threshold
    STAPLE_mask = sitk.GetArrayFromImage(STAPLE_mask_stack)
    im = Image.fromarray(STAPLE_mask * 255, 'L')
    im.save(os.path.join(output_folder, 'FusedMask_STAPLE.png'))
    # print("The fused mask saved.")



# from LabelFusion.wrapper import fuse_images
# MaskFolder = "LabelFusion/"
# images_to_fuse = []
#
# for mask_file in os.listdir(MaskFolder):
#     mask_path = os.path.join(MaskFolder, mask_file)
#     images_to_fuse.append(sitk.ReadImage(mask_path, sitk.sitkUInt8))
# #
# STAPLE_mask = fuse_images(images_to_fuse, 'simple', class_list=[0,1])
# # STAPLE_mask = fuse_images(images_to_fuse, 'staple')
#
# # sitk.WriteImage(STAPLE_mask, 'simple_fused.nii')
# fused_simple = sitk.GetArrayFromImage(STAPLE_mask)
# im = Image.fromarray((fused_simple * 255).astype(np.uint8), 'L')
# im.save('simple_fused.png')