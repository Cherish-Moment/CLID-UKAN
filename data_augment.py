import os
import cv2
import numpy as np
from tqdm import tqdm
import random
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

img_w = 224
img_h = 224

def elastic_transform(image, label, alpha=10, sigma=2, alpha_affine=2, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape[:2]

    # Random affine
    center_square = np.float32(shape) // 2
    square_size = min(shape) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    imageB = cv2.warpAffine(image, M, shape[::-1], borderMode=cv2.BORDER_REFLECT_101)
    imageA = cv2.warpAffine(label, M, shape[::-1], borderMode=cv2.BORDER_REFLECT_101)

    # generate random displacement fields
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    # Apply elastic transform to each channel separately
    xb = np.zeros_like(image)
    yb = np.zeros_like(label)
    for i in range(image.shape[2]):  # Assuming 3 channels (RGB)
        xb[:, :, i] = map_coordinates(imageB[:, :, i], indices, order=1, mode='constant').reshape(shape)
        yb[:, :, i] = map_coordinates(imageA[:, :, i], indices, order=1, mode='constant').reshape(shape)

    return xb, yb


def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def rotate(xb, yb, angle):
    M_rotate = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb, yb


def blur(img):
    return cv2.blur(img, (3, 3))


def add_noise(xb):
    sigma = random.uniform(5, 10)
    gauss = np.random.normal(0, sigma, xb.shape)
    noisy = xb + gauss
    return np.clip(noisy, 0, 255).astype('uint8')


def data_augment(xb, yb):
    if np.random.random() < 0.20:
        xb, yb = rotate(xb, yb, random.uniform(0, 20))
    if np.random.random() < 0.20:
        xb, yb = rotate(xb, yb, random.uniform(-20, -1))
    if np.random.random() < 0.30:
        xb = cv2.flip(xb, 1)
        yb = cv2.flip(yb, 1)
    if np.random.random() < 0.30:
        xb = random_gamma_transform(xb, 1.0)
    if np.random.random() < 0.30:
        xb = blur(xb)
    if np.random.random() < 0.25:
        xb = add_noise(xb)
    if np.random.random() < 0.35:
        xb, yb = elastic_transform(xb, yb)
    return xb, yb


def create_augmented_dataset(original_folder, augmented_folder, num_augmentations=20):
    image_files = [f for f in os.listdir(original_folder) if f.endswith('.png') and '_mask' not in f]

    os.makedirs(os.path.join(augmented_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(augmented_folder, 'masks'), exist_ok=True)

    for image_file in tqdm(image_files):
        base_name = image_file.split('.png')[0]
        image_path = os.path.join(original_folder, image_file)
        mask_path = os.path.join(original_folder, f'{base_name}_mask.png')

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)

        for i in range(num_augmentations):
            augmented_image, augmented_mask = data_augment(image, mask)

            save_image_path = os.path.join(augmented_folder, 'images', f'{base_name}_aug_{i}.png')
            save_mask_path = os.path.join(augmented_folder, 'masks', f'{base_name}_mask_aug_{i}.png')

            cv2.imwrite(save_image_path, augmented_image)
            cv2.imwrite(save_mask_path, augmented_mask)

            # Also save original image and mask if needed
            if i == 0:
                original_image_save_path = os.path.join(augmented_folder, 'images', f'{base_name}.png')
                original_mask_save_path = os.path.join(augmented_folder, 'masks', f'{base_name}_mask.png')
                cv2.imwrite(original_image_save_path, image)
                cv2.imwrite(original_mask_save_path, mask)


# Example usage
original_folder = 'path to your original folder'  # 原始文件夹路径
augmented_folder = 'path to your new folder'  # 增强后文件夹路径

create_augmented_dataset(original_folder, augmented_folder, num_augmentations=1)
