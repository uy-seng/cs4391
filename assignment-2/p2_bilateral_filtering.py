"""
CS 4391 Homework 2 Programming: Part 3 - bilateral filter
Implement the bilateral_filtering() function in this python script
"""
 
import cv2
import numpy as np
import math

def bilateral_filtering(
    img: np.uint8,
    spatial_variance: float,
    intensity_variance: float,
    kernel_size: int,
) -> np.uint8:
    """
    Homework 2 Part 3
    Compute the bilaterally filtered image given an input image, kernel size, spatial variance, and intensity range variance
    """

    img = img / 255.0  # Normalize image to range [0, 1]
    img = img.astype("float32")
    img_filtered = np.zeros(img.shape)  # Placeholder for the filtered image
    offset = kernel_size // 2
    
    # Apply zero-padding to handle border pixels
    img = np.pad(img, ((offset, offset), (offset, offset)), mode='constant')

    height, width = img.shape
    
    # Iterate over each pixel in the image
    for m in range(offset, height - offset):
        for n in range(offset, width - offset):
            W_mn = 0  # Normalization factor
            intensity_sum = 0  # Weighted intensity sum
            
            f_current = img[m, n]  # Current pixel intensity
            
            # Loop through the local neighborhood
            for k in range(-offset, offset + 1):
                for l in range(-offset, offset + 1):
                    f_neighbour = img[m + k, n + l]  # Neighboring pixel intensity
                    
                    # Calculate the intensity difference
                    intensity_diff = f_current - f_neighbour
                    
                    # Calculate intensity range weight (based on intensity difference)
                    intensity_range_weight = np.exp(- (intensity_diff ** 2) / (2 * intensity_variance))
                    
                    # Calculate spatial weight (based on distance from the center)
                    spatial_weight = np.exp(- (k ** 2 + l ** 2) / (2 * spatial_variance))
                    
                    # Calculate the final weight (combined range and spatial weight)
                    weight = intensity_range_weight * spatial_weight
                    
                    # Accumulate the weighted intensity sum
                    intensity_sum += weight * f_neighbour
                    
                    # Accumulate the normalization factor
                    W_mn += weight
            
            # Normalize the filtered pixel value
            img_filtered[m - offset, n - offset] = intensity_sum / W_mn
    
    # Scale the output image back to the range [0, 255] and convert to uint8
    img_filtered = img_filtered * 255
    img_filtered = np.clip(img_filtered, 0, 255).astype("uint8")
    
    return img_filtered
 
if __name__ == "__main__":
    img = cv2.imread("data/img/butterfly.jpeg", 0) # read gray image
    img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA) # reduce image size for saving your computation time
    cv2.imwrite('results/im_original.png', img) # save image 
    
    # Generate Gaussian noise
    noise = np.random.normal(0,0.6,img.size)
    noise = noise.reshape(img.shape[0],img.shape[1]).astype('uint8')
   
    # Add the generated Gaussian noise to the image
    img_noise = cv2.add(img, noise)
    cv2.imwrite('results/im_noisy.png', img_noise)
    
    # Bilateral filtering
    spatial_variance = 30 # signma_s^2
    intensity_variance = 0.5 # sigma_r^2
    kernel_size = 7
    img_bi = bilateral_filtering(img_noise, spatial_variance, intensity_variance, kernel_size)
    cv2.imwrite('results/im_bilateral.png', img_bi)