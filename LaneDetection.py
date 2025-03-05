import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from numpy.ma.extras import average

# Camera top filterout
topFilterOut = 0.5

# mask filter variables
bottomLeftCol = 0
bottomRightCol = 1
topLeftCol = 0.2
topRightCol = 0.8
bottomRow = 1
topRow = 0.4

# Canny edge variables
cannyLowThreshold = 240
cannyHighThreshold = 255
# Hough transform variables
rho = 1  # 1
theta = np.pi / 180
threshold = 20  # 20
minLineLength = 50  # 20
maxLineGap = 100  # 500
# Hough line outlier filter variables
percentile = 0.25
outlierMultiplier = 1.5

def regionSelect(image, show=False):
    mask = np.zeros_like(image)
    mask_color = 255
    rows, cols = image.shape[:2]



    bottom_left = [cols * bottomLeftCol, rows * bottomRow]
    top_left = [cols * topLeftCol, rows * topRow]
    bottom_right = [cols * bottomRightCol, rows * bottomRow]
    top_right = [cols * topRightCol, rows * topRow]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, mask_color)
    masked_image = cv2.bitwise_and(image, mask)

    if show:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        overlay_image = image_rgb.copy()
        cv2.polylines(overlay_image, [vertices], isClosed=True, color=(255, 0, 0), thickness=3)
        plt.imshow(overlay_image)
        plt.title("Road Detection Region")
        plt.axis("off")
        plt.show()

    return masked_image




def hough_transform(image):

    # Distance resolution of the accumulator in pixels.
    rho = 1 # 1
    # Angle resolution of the accumulator in radians.
    theta = np.pi / 180
    # Only lines that are greater than threshold will be returned.
    threshold = 20 # 20
    # Line segments shorter than that are rejected.
    minLineLength = 100 # 20
    # Maximum allowed gap between points on the same line to link them
    maxLineGap = 500 #500
    lines = cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold,
                           minLineLength=minLineLength, maxLineGap=maxLineGap)
    return lines if lines is not None else []

def lineToPixel(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))

def filter_outliers(df, col):
    if df.empty:
        return df  # Return empty DataFrame instead of failing
    lowPercentile = df[col].quantile(percentile)
    highPercentile = df[col].quantile(1-percentile)
    IQR = highPercentile - lowPercentile
    lower_bound = lowPercentile - outlierMultiplier * IQR
    upper_bound = highPercentile + outlierMultiplier * IQR
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

def assignLane(lines, imageHeight):
    if not lines.any():
        return None, None
    dtype_dict = {
        "x1": float, "y1": float, "x2": float, "y2": float,
        "length": float, "slope": float, "intercept": float
    }

    left_lines = pd.DataFrame(columns=dtype_dict.keys()).astype(dtype_dict)
    right_lines = pd.DataFrame(columns=dtype_dict.keys()).astype(dtype_dict)

    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(x1 - x2) < 50:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))

            new_row = pd.DataFrame([[x1, y1, x2, y2, length, slope, intercept]], columns=dtype_dict.keys())

            if slope < 0:
                left_lines = pd.concat([left_lines, new_row], ignore_index=True)
            else:
                right_lines = pd.concat([right_lines, new_row], ignore_index=True)

    f_l = filter_outliers(left_lines, 'intercept')
    f_r = filter_outliers(right_lines, 'intercept')
    filtered_left = filter_outliers(f_l, 'slope')
    filtered_right = filter_outliers(f_r, 'slope')

    left_slope = np.average(filtered_left["slope"], weights=filtered_left["length"])
    left_intercept = np.average(filtered_left["intercept"], weights=filtered_left["length"])
    right_slope = np.average(filtered_right["slope"], weights=filtered_right["length"])
    right_intercept = np.average(filtered_right["intercept"], weights=filtered_right["length"])

    y1 = int(imageHeight)
    y2 = int(y1*0.5)
    left_x1 = int((y1 - left_intercept)/left_slope)
    left_x2 = int((y2 - left_intercept)/left_slope)

    right_x1 = int((y1 - right_intercept)/right_slope)
    right_x2 = int((y2 - right_intercept)/right_slope)

    leftLane = (left_x1, left_x2, y1, y2, left_slope, left_intercept)
    rightLane = (right_x1, right_x2, y1, y2, right_slope, right_intercept)
    return leftLane, rightLane


def frameProcess(image, show=False):
    h = image.shape[0]

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blurred = cv2.GaussianBlur(image_gray, (7, 7), 0)
    image_canny = cv2.Canny(image_blurred, cannyLowThreshold, cannyHighThreshold)
    image_processed = regionSelect(image_canny, show=show)
    lines = hough_transform(image_processed)

    leftLane, rightLane = assignLane(lines, h)
    if leftLane is None or rightLane is None:
        return image_canny

    left_x1, left_x2, hy1, hy2, left_slope, left_intercept = leftLane
    right_x1, right_x2, hy1, hy2, right_slope, right_intercept = rightLane

    if show:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        line_image = np.copy(image_rgb)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
        # if filteredLeft is not None:
        #     for line in filteredLeft:
        #         cv2.line(filtered_line_image, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 3)
        # if filteredRight is not None:
        #     for line in filteredRight:
        #         cv2.line(filtered_line_image, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 3)

        fig, axes = plt.subplots(2, 3, figsize=(30,20))
        axes[0,0].imshow(image_rgb)
        axes[0,0].set_title("Original Image")
        axes[0,0].axis("off")

        axes[0,1].imshow(image_blurred, cmap='gray')
        axes[0,1].set_title("Greyscale, Gaussian Blur")
        axes[0,1].axis("off")

        axes[0,2].imshow(image_canny, cmap='gray')
        axes[0,2].set_title("Canny Edge Detection")
        axes[0,2].axis("off")

        axes[1,0].imshow(line_image, cmap='gray')
        axes[1,0].set_title("Hough Transform")
        axes[1,0].axis("off")

        # axes[1,1].imshow(filtered_line_image, cmap='gray')
        # axes[1,1].set_title("Filtered lines")
        # axes[1,1].axis("off")

        hough_line_image = np.copy(image_rgb)
        cv2.line(hough_line_image, (left_x1, hy1), (left_x2, hy2), (255, 0, 0), 10)
        cv2.line(hough_line_image, (right_x1, hy1), (right_x2, hy2), (255, 0, 0), 10)

        axes[1,1].imshow(hough_line_image, cmap='gray')
        axes[1,1].set_title("Hough Transform")
        axes[1,1].axis("off")

        plt.show()

    return image_canny


testingRoad1 = cv2.imread('C:\\Users\\brifr\\OneDrive\\CMPE\\Y3 Winter\\390\\RoadDetection\\TestingVideos\\TestingRoad1.jpg')

frameProcess(testingRoad1, True)