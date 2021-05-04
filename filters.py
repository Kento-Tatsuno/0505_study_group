import cv2
import numpy as np

def main():
    orig_img = cv2.imread('images/bugdroid.png')

    kernel_size = 15

    rand_kernel = np.random.random(kernel_size)
    filtered_img = cv2.filter2D(orig_img, -1, rand_kernel) / kernel_size ** 2

    average_kernel = np.ones([kernel_size, kernel_size], dtype=np.float32) / kernel_size ** 2
    moving_avarage_img = cv2.filter2D(orig_img, -1, average_kernel)

    gaussian_blur_img = cv2.GaussianBlur(orig_img, (kernel_size, kernel_size), 0)

    highpass_filter = np.full([kernel_size, kernel_size], -1, dtype=np.float32)
    highpass_filter[(kernel_size - 1) // 2, (kernel_size - 1) // 2] = kernel_size ** 2 - 1
    highpass_filter = highpass_filter / kernel_size ** 2

    highpass_img = cv2.filter2D(orig_img, -1, highpass_filter)

    cv2.imwrite('output/rand_convolution.png', filtered_img)
    cv2.imwrite('output/moving_avarage.png', moving_avarage_img)
    cv2.imwrite('output/gaussian_blur.png', gaussian_blur_img)
    cv2.imwrite('output/highpass.png', highpass_img)


if __name__ == '__main__':
    main()