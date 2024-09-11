import cv2
import numpy as np
def localTextureeatures(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    # Compute differential excitation
    excitation = np.zeros_like(gray, dtype=np.float64)
    for i in range(1, gray.shape[0] - 1):
        for j in range(1, gray.shape[1] - 1):
            neighbors = gray[i - 1:i + 2, j - 1:j + 2].flatten()
            center = gray[i, j]
            excitation[i, j] = np.sum((neighbors - center) / center)
        # Compute orientation
        orientation = np.arctan2(grad_y, grad_x)
        #print("local Texture feature for each value with zeroPadding: ",orientation.shape)


        textureDistanceBetweenPq_Pr = np.linalg.norm(orientation[:, np.newaxis] - orientation[np.newaxis, :], axis=2)
        # Quantize orientation
        orientation_bins = np.digitize(orientation, bins=np.linspace(-np.pi, np.pi, num=8))

        # Construct WLD histogram
        histogram = np.histogram2d(excitation.flatten(), orientation_bins.flatten(), bins=(20, 8))[0]

        #the next return is for finding the texture value between every two pixels and applying L2
        #return textureDistanceBetweenPq_Pr

        # the orginal texture feature for each pixel
        return orientation



def main():
    # Load image

    img = cv2.imread("sun.jpg")
    print(img.shape)
    #print("orginal image shape",img.shape)
    # Compute WLD
    texturedistance = localTextureeatures(img)

    # Print WLD histogram
    print("",texturedistance.shape)


if __name__ == "__main__":
    main()