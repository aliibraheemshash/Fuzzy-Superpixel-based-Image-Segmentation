import cv2
import numpy as np

def LocalContourFeature(img):
    # Read the input image


    if img is None:
        print("Error: Could not open or find the image.")
        return

    # Convert the image to LAB color space and use the L channel
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img = img_lab[:, :, 0]
    #print(img[:5,:5])

    # Define the vertical and horizontal gradient filters as numpy arrays
    kernel_vertical_gradient_filter = np.array([[0, -1, 0],
                                                [0,  0, 0],
                                                [0,  1, 0]])

    kernel_horizontal_gradient_filter = np.array([[ 0, 0, 0],
                                                  [-1, 0, 1],
                                                  [ 0, 0, 0]])

    # Apply the vertical gradient filter on the image
    filtered_image_vertical = cv2.filter2D(img, -1, kernel_vertical_gradient_filter,borderType = cv2.BORDER_CONSTANT)
    #print(filtered_image_vertical.shape)
    # Apply the horizontal gradient filter on the image
    filtered_image_horizontal = cv2.filter2D(img, -1, kernel_horizontal_gradient_filter,borderType = cv2.BORDER_CONSTANT)
    #print(filtered_image_horizontal[:5, :5])
    # Subtract the horizontal filter result from the vertical filter result
    difference_image = cv2.subtract(filtered_image_vertical, filtered_image_horizontal)

    # Apply square root to the result
    sqrt_image = np.sqrt(np.abs(difference_image)).astype(np.uint8)

    # Define threshold values
    T1 = 5
    T2 = 15

    # Apply thresholding
    sqrt_image = np.where(sqrt_image < T1, T1, sqrt_image)
    sqrt_image = np.where(sqrt_image > T2, T2, sqrt_image)
    # print(sqrt_image[:200,:200])
    # Save and display the resulting image
    # cv2.imwrite('sqrt_image.jpg', sqrt_image)
    # cv2.imshow('Square Root Image', sqrt_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #gradient_dif_Pq_Pr =np.linalg.norm(sqrt_image[:, np.newaxis] - sqrt_image[np.newaxis, :], axis=2)
    #print(gradient_dif_Pq_Pr[24:27,24:27])
    #print(gradient_dif_Pq_Pr.shape)

    return sqrt_image

def main():
    img = cv2.imread("sun.jpg")
    print(img.shape)
    result = LocalContourFeature(img)
    print(result.shape)
if __name__ == "__main__":
    main()
