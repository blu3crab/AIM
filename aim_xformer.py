###############################################################################
# AIM image transformations
# aim_xformer.py
#
import cv2
import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import aim_util
#import aim_tuner

#gdrive_path = '/content/gdrive/MyDrive/AIM/'

###############################################################################
# base transforms
#
# xform1 - applying grayscale conversion
def grayscale(img, intensity=0.0):
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if intensity > 0.0:
        img1 = cv2.multiply(img1, intensity)
    return img1

# xform2 - applying thresholding
def threshold(img, thresh_val=127, thresh_type=cv2.THRESH_BINARY):
    ret, thresh = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY)
    return thresh

# xform3 - applying noise reduction using Gaussian Blur
def noise_reduction(img, kernel_size=5):
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return blurred

# xform4 - applying sharpening using Laplacian operator
def sharpen(img, kernel_size=3):
    sharpened = cv2.Laplacian(img, cv2.CV_8U, ksize=kernel_size)
    sharpened = cv2.add(img, sharpened, 1.0)
    return sharpened

# xform5 -adjusting contrast and brightness
def contrast_and_brightness(img, contrast_factor=1.0, brightness_factor=0):
    contrast_adjusted = img * contrast_factor
    brightness_adjusted = contrast_adjusted + brightness_factor
    return np.clip(brightness_adjusted, 0, 255)
###############################################################################
def apply_xform(gdrive_path, image_path, source_base, xform_prefix_list):
    # read image
    img = mpimg.imread(image_path)

    # Perform the image transformations
    gray_img = grayscale(img)
    thresh_img = threshold(img)
    noise_reduced_img = noise_reduction(img)
    sharpened_img = sharpen(img)
    adjusted_img = contrast_and_brightness(img, 1, 0)

    # save the transformed images
    #gdrive_path = '/content/gdrive/MyDrive/AIM/'
    # cv2.imwrite(os.path.join(gdrive_path, 'citizen_xform1.jpg'), gray_img)
    # cv2.imwrite(os.path.join(gdrive_path, 'citizen_xform2.jpg'), thresh_img)
    # cv2.imwrite(os.path.join(gdrive_path, 'citizen_xform3.jpg'), noise_reduced_img)
    # cv2.imwrite(os.path.join(gdrive_path, 'citizen_xform4.jpg'), sharpened_img)
    # cv2.imwrite(os.path.join(gdrive_path, 'citizen_xform5.jpg'), adjusted_img)
    print(f"base->{source_base}, prefix->{xform_prefix_list}")

    cv2.imwrite(os.path.join(gdrive_path, source_base + xform_prefix_list[1] + '.jpg'), gray_img)
    cv2.imwrite(os.path.join(gdrive_path, source_base + xform_prefix_list[2] + '.jpg'), thresh_img)
    cv2.imwrite(os.path.join(gdrive_path, source_base + xform_prefix_list[3] + '.jpg'), noise_reduced_img)
    cv2.imwrite(os.path.join(gdrive_path, source_base + xform_prefix_list[4] + '.jpg'), sharpened_img)
    cv2.imwrite(os.path.join(gdrive_path, source_base + xform_prefix_list[5] + '.jpg'), adjusted_img)

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
    plt.show()
###############################################################################
# apply grayscale multiple times with varying intensity
# image_path = '/content/gdrive/MyDrive/AIM/citizen_1864_rescan.jpg'

def apply_grayscale(gdrive_path, image_path, source_base, xform_prefix, intensity_a=0.7, intensity_b=1.3, intensity_c=1.8):
    # read image
    img = mpimg.imread(image_path)

    # Load an image
    image = img
    # image = cv2.imread('your_image.jpg')  # Replace 'your_image.jpg' with your image file

    # Apply grayscale multiple times with varying intensity
    result1 = grayscale(image)
    result2 = grayscale(image, intensity_a)
    result3 = grayscale(image, intensity_b)
    result4 = grayscale(image, intensity_c)

    # result2 = cv2.multiply(grayscale(image), intensity_a)  # Slightly darker
    # result3 = cv2.multiply(grayscale(image), intensity_b)  # Slightly brighter

    # Display images side-by-side
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))  # Adjust figsize if needed

    axes[0].imshow(result1, cmap='gray')
    axes[0].set_title('Grayscale 0')
    # cv2.imwrite(os.path.join(gdrive_path, 'citizen_xform1a.jpg'), result1)
    cv2.imwrite(os.path.join(gdrive_path, source_base + xform_prefix + 'a.jpg'), result1)

    axes[1].imshow(result2, cmap='gray')
    axes[1].set_title('Grayscale 1 (Darker)')
    # cv2.imwrite(os.path.join(gdrive_path, 'citizen_xform1b.jpg'), result2)
    cv2.imwrite(os.path.join(gdrive_path, source_base + xform_prefix + 'b.jpg'), result2)

    axes[2].imshow(result3, cmap='gray')
    axes[2].set_title('Grayscale 2 (Brighter)')
    # cv2.imwrite(os.path.join(gdrive_path, 'citizen_xform1c.jpg'), result3)
    cv2.imwrite(os.path.join(gdrive_path, source_base + xform_prefix + 'c.jpg'), result3)

    axes[3].imshow(result4, cmap='gray')
    axes[3].set_title('Grayscale 3 (Brighter)')
    # cv2.imwrite(os.path.join(gdrive_path, 'citizen_xform1d.jpg'), result3)
    cv2.imwrite(os.path.join(gdrive_path, source_base + xform_prefix + 'd.jpg'), result3)

    plt.show()
###############################################################################
def apply_xform_threshold(gdrive_path, image_path, source_base, xform_prefix):
    # read image
    img = mpimg.imread(image_path)

    transformed_images = []
    # try varying key input Laplacian parameters: threshold value, type
    thresh_img = threshold(img)
    transformed_images.append(thresh_img)
    # cv2.imwrite(os.path.join(gdrive_path, 'citizen_xform2a.jpg'), thresh_img)
    cv2.imwrite(os.path.join(gdrive_path, source_base + xform_prefix + 'a.jpg'), thresh_img)

    thresh_img = threshold(img, thresh_val=50)
    transformed_images.append(thresh_img)
    # cv2.imwrite(os.path.join(gdrive_path, 'citizen_xform2b.jpg'), thresh_img)
    cv2.imwrite(os.path.join(gdrive_path, source_base + xform_prefix + 'b.jpg'), thresh_img)

    # thresh_img = threshold(img, thresh_val=200, thresh_type=cv2.THRESH_BINARY_INV)
    # transformed_images.append(thresh_img)
    # # cv2.imwrite(os.path.join(gdrive_path, 'citizen_xform2c.jpg'), thresh_img)
    # cv2.imwrite(os.path.join(gdrive_path, source_base + xform_prefix + 'c.jpg'), thresh_img)

    thresh_img = threshold(img, thresh_val=127, thresh_type=cv2.THRESH_TRUNC)
    transformed_images.append(thresh_img)
    # cv2.imwrite(os.path.join(gdrive_path, 'citizen_xform2d.jpg'), thresh_img)
    cv2.imwrite(os.path.join(gdrive_path, source_base + xform_prefix + 'c.jpg'), thresh_img)

    thresh_img = threshold(img, thresh_val=75, thresh_type=cv2.THRESH_TOZERO_INV)
    transformed_images.append(thresh_img)
    # cv2.imwrite(os.path.join(gdrive_path, 'citizen_xform2e.jpg'), thresh_img)
    cv2.imwrite(os.path.join(gdrive_path, source_base + xform_prefix + 'd.jpg'), thresh_img)

    # thresh_img = threshold(img, thresh_val=170, thresh_type=cv2.THRESH_OTSU)
    # transformed_images.append(thresh_img)
    # # cv2.imwrite(os.path.join(gdrive_path, 'citizen_xform2f.jpg'), thresh_img)
    # cv2.imwrite(os.path.join(gdrive_path, source_base + xform_prefix + 'f.jpg'), thresh_img)

    # Display the images side by side
    for i in range(0, len(transformed_images)):
        plt.subplot(1, len(transformed_images), i + 1)
        plt.title("xform" + str(i))
        plt.imshow(transformed_images[i], cmap='gray')
        plt.axis('off')
    plt.show()
###############################################################################
def apply_xform_noise(gdrive_path, image_path, source_base, xform_prefix):
    # read image
    img = mpimg.imread(image_path)

    transformed_images = []
    # vary noise kernal size
    noise_img = noise_reduction(img, 3) # default
    transformed_images.append(noise_img)
    cv2.imwrite(os.path.join(gdrive_path, source_base + xform_prefix + 'a.jpg'), noise_img)

    noise_img = noise_reduction(img, 7)
    transformed_images.append(noise_img)
    cv2.imwrite(os.path.join(gdrive_path, source_base + xform_prefix + 'b.jpg'), noise_img)

    noise_img = noise_reduction(img, 21)
    transformed_images.append(noise_img)
    cv2.imwrite(os.path.join(gdrive_path, source_base + xform_prefix + 'c.jpg'), noise_img)

    noise_img = noise_reduction(img, 31)
    transformed_images.append(noise_img)
    cv2.imwrite(os.path.join(gdrive_path, source_base + xform_prefix + 'd.jpg'), noise_img)


    # Display the images side by side
    for i in range(0, len(transformed_images)):
        plt.subplot(1, len(transformed_images), i + 1)
        plt.title("xform " + str(i + 1))
        plt.imshow(transformed_images[i], cmap='gray')
        plt.axis('off')
    plt.show()
###############################################################################
def apply_xform_sharpen(gdrive_path, image_path, source_base, xform_prefix):
    # read image
    img = mpimg.imread(image_path)

    transformed_images = []
    # vary sharpen kernal size
    sharpened_img = sharpen(img, 3) # default
    transformed_images.append(sharpened_img)
    # cv2.imwrite(os.path.join(gdrive_path, 'citizen_xform4a.jpg'), sharpened_img)
    cv2.imwrite(os.path.join(gdrive_path, source_base + xform_prefix + 'a.jpg'), sharpened_img)

    sharpened_img = sharpen(img, 1)
    transformed_images.append(sharpened_img)
    # cv2.imwrite(os.path.join(gdrive_path, 'citizen_xform4b.jpg'), sharpened_img)
    cv2.imwrite(os.path.join(gdrive_path, source_base + xform_prefix + 'b.jpg'), sharpened_img)

    sharpened_img = sharpen(img, 7)
    transformed_images.append(sharpened_img)
    # cv2.imwrite(os.path.join(gdrive_path, 'citizen_xform4b.jpg'), sharpened_img)
    cv2.imwrite(os.path.join(gdrive_path, source_base + xform_prefix + 'c.jpg'), sharpened_img)

    sharpened_img = sharpen(img, 17)
    transformed_images.append(sharpened_img)
    # cv2.imwrite(os.path.join(gdrive_path, 'citizen_xform4b.jpg'), sharpened_img)
    cv2.imwrite(os.path.join(gdrive_path, source_base + xform_prefix + 'd.jpg'), sharpened_img)

    # Display the images side by side
    for i in range(0, len(transformed_images)):
        plt.subplot(1, len(transformed_images), i + 1)
        plt.title("xform " + str(i + 1))
        plt.imshow(transformed_images[i], cmap='gray')
        plt.axis('off')
    plt.show()
###############################################################################
# apply contrast and brightness variants
def apply_xform_contrast_and_brightness(gdrive_path, image_path, source_base, xform_prefix):
    # read image
    img = mpimg.imread(image_path)

    transformed_images = []
    # vary contrast_and_brightness inputs
    adj_img = contrast_and_brightness(img, contrast_factor=1, brightness_factor=0)  # default
    transformed_images.append(adj_img)
    # cv2.imwrite(os.path.join(gdrive_path, 'citizen_xform5a.jpg'), adj_img)
    cv2.imwrite(os.path.join(gdrive_path, source_base + xform_prefix + 'a.jpg'), adj_img)

    adj_img = contrast_and_brightness(img, contrast_factor=1, brightness_factor=32)
    transformed_images.append(adj_img)
    # cv2.imwrite(os.path.join(gdrive_path, 'citizen_xform5b.jpg'), adj_img)
    cv2.imwrite(os.path.join(gdrive_path, source_base + xform_prefix + 'b.jpg'), adj_img)

    adj_img = contrast_and_brightness(img, contrast_factor=2, brightness_factor=32)
    transformed_images.append(adj_img)
    # cv2.imwrite(os.path.join(gdrive_path, 'citizen_xform5c.jpg'), adj_img)
    cv2.imwrite(os.path.join(gdrive_path, source_base + xform_prefix + 'c.jpg'), adj_img)

    adj_img = contrast_and_brightness(img, contrast_factor=2, brightness_factor=64)
    transformed_images.append(adj_img)
    # cv2.imwrite(os.path.join(gdrive_path, 'citizen_xform5c.jpg'), adj_img)
    cv2.imwrite(os.path.join(gdrive_path, source_base + xform_prefix + 'd.jpg'), adj_img)

    # Display the images side by side
    for i in range(0, len(transformed_images)):
        plt.subplot(1, len(transformed_images), i + 1)
        plt.title("xform " + str(i))
        plt.imshow(transformed_images[i], cmap='gray')
        plt.axis('off')
    plt.show()
###############################################################################
