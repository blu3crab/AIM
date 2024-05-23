###############################################################################
# AIM image transformations
#
import cv2
import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import aim_util

###############################################################################
# base transforms
#
# applying grayscale conversion
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# applying thresholding
def threshold(img, thresh_val=127):
    ret, thresh = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY)
    return thresh

# applying noise reduction using Gaussian Blur
def noise_reduction(img, kernel_size=5):
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return blurred

# applying sharpening using Laplacian operator
def sharpen(img, kernel_size=3):
    sharpened = cv2.Laplacian(img, cv2.CV_8U, ksize=kernel_size)
    sharpened = cv2.add(img, sharpened, 1.0)
    return sharpened

# adjusting contrast and brightness
def contrast_and_brightness(img, contrast_factor=1.0, brightness_factor=0):
    contrast_adjusted = img * contrast_factor
    brightness_adjusted = contrast_adjusted + brightness_factor
    return np.clip(brightness_adjusted, 0, 255)
###############################################################################
def trial_xform(image_path):
    # read image
    img = mpimg.imread(image_path)

    # Perform the image transformations
    gray_img = grayscale(img)
    thresh_img = threshold(img)
    noise_reduced_img = noise_reduction(img)
    sharpened_img = sharpen(img)
    adjusted_img = contrast_and_brightness(img, 1, 0)

    # save the transformed images
    gdrive_path = '/content/gdrive/MyDrive/AIM/'
    cv2.imwrite(os.path.join(gdrive_path, 'citizen_xform1.jpg'), gray_img)
    cv2.imwrite(os.path.join(gdrive_path, 'citizen_xform2.jpg'), thresh_img)
    cv2.imwrite(os.path.join(gdrive_path, 'citizen_xform3.jpg'), noise_reduced_img)
    cv2.imwrite(os.path.join(gdrive_path, 'citizen_xform4.jpg'), sharpened_img)
    cv2.imwrite(os.path.join(gdrive_path, 'citizen_xform5.jpg'), adjusted_img)

    # Create a list to store the transformed images
    transformed_images = []

    # 1. Grayscale Conversion
    transformed_images.append(gray_img)
    transformed_images.append(thresh_img)
    transformed_images.append(noise_reduced_img)
    transformed_images.append(sharpened_img)
    transformed_images.append(adjusted_img)

    # Display the images side by side
    for i in range(0, len(transformed_images)):
        plt.subplot(1, len(transformed_images), i + 1)
        plt.title("Transform " + str(i + 1))
        plt.imshow(transformed_images[i], cmap='gray')
        plt.axis('off')
        # save_image_name = '/content/gdrive/MyDrive/AIM/citizen_xform_' + str(i) + '.jpg'
        # plt.savefig(save_image_name)
    plt.show()
###############################################################################
# apply grayscale multiple times with varying intensity
# image_path = '/content/gdrive/MyDrive/AIM/citizen_1864_rescan.jpg'

def apply_grayscale(image_path, intensity_a=0.7, intensity_b=1.3):
    # read image
    img = mpimg.imread(image_path)

    # Load an image
    image = img
    # image = cv2.imread('your_image.jpg')  # Replace 'your_image.jpg' with your image file

    # Apply grayscale multiple times with varying intensity
    result1 = grayscale(image)
    result2 = cv2.multiply(grayscale(image), intensity_a)  # Slightly darker
    result3 = cv2.multiply(grayscale(image), intensity_b)  # Slightly brighter

    # Display images side-by-side
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))  # Adjust figsize if needed

    axes[0].imshow(result1, cmap='gray')
    axes[0].set_title('Grayscale 1')

    axes[1].imshow(result2, cmap='gray')
    axes[1].set_title('Grayscale 2 (Darker)')

    axes[2].imshow(result3, cmap='gray')
    axes[2].set_title('Grayscale 3 (Brighter)')

    plt.show()
###############################################################################
# apply contrast and brightness variants
# image_path = '/content/gdrive/MyDrive/AIM/citizen_1864_rescan.jpg'
# adj1_img = contrast_and_brightness(img, 2, 8)

def apply_contrast_and_brightness(image_path, contrast_factor=1.0, brightness_factor=0):
    # read image
    img = mpimg.imread(image_path)

    adj_img = contrast_and_brightness(img, contrast_factor, brightness_factor)
    plt.imshow(adj_img)

    return adj_img
###############################################################################
