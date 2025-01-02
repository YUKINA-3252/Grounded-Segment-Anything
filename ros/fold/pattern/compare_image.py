import cv2
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

imageA = cv2.imread('mask_top_paper_1.png')
imageB = cv2.imread('mask_top_paper_2.png')

imageA_copy = imageA.copy()
imageB_copy = imageB.copy()

gray = cv2.cvtColor(imageA_copy, cv2.COLOR_BGR2GRAY)
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
max_contour_paper = max(contours, key=cv2.contourArea)
epsilon = 0.02 * cv2.arcLength(max_contour_paper, True)
approx = cv2.approxPolyDP(max_contour_paper, epsilon, True)
if len(approx) != 4:
    raise ValueError("Not a rectangular: num of points: {}".format(len(approx)))
vertices_paper = approx[:, 0, :]
vertices_paper = np.array(vertices_paper)
A_left_top = vertices_paper[np.argmin(vertices_paper[:, 0] + vertices_paper[:, 1])]
A_left_bottom = vertices_paper[np.argmin(vertices_paper[:, 0] - vertices_paper[:, 1])]
A_right_top = vertices_paper[np.argmax(vertices_paper[:, 0] - vertices_paper[:, 1])]
A_right_bottom = vertices_paper[np.argmax(vertices_paper[:, 0] + vertices_paper[:, 1])]


grayB = cv2.cvtColor(imageB_copy, cv2.COLOR_BGR2GRAY)
contours, _ = cv2.findContours(grayB, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
max_contour_paper = max(contours, key=cv2.contourArea)
hull = cv2.convexHull(max_contour_paper)
hull_mask = np.zeros_like(gray)
cv2.drawContours(hull_mask, [hull], 0, 255, -1)
convex_defects = cv2.subtract(hull_mask, grayB)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(convex_defects)
sizes = stats[:, cv2.CC_STAT_AREA]
largest_label = 1 + np.argmax(sizes[1:])
largest_convex_defects = np.zeros_like(convex_defects)
largest_convex_defects[labels == largest_label] = 255
visualization = cv2.cvtColor(largest_convex_defects, cv2.COLOR_GRAY2BGR)
cv2.imwrite("non.jpg", visualization)
coordinates = np.column_stack(np.where(largest_convex_defects > 0))
max_x_idx = np.argmax(coordinates[:, 1])
max_x_point = coordinates[max_x_idx]
# extract paper mask
extracted_imageB_mask = np.zeros_like(grayB)
coordinates = np.column_stack(np.where(grayB > 0))
for y, x in coordinates:
    if x >= max_x_point[1]:
        extracted_imageB_mask[y, x] = 255
cv2.imwrite("extracted.jpg", extracted_imageB_mask)
contours, _ = cv2.findContours(extracted_imageB_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
max_contour_paper = max(contours, key=cv2.contourArea)
epsilon = 0.02 * cv2.arcLength(max_contour_paper, True)
approx = cv2.approxPolyDP(max_contour_paper, epsilon, True)
if len(approx) != 4:
        raise ValueError("Not a rectangular: num of points: {}".format(len(approx)))
vertices_paper = approx[:, 0, :]
vertices_paper = np.array(vertices_paper)
B_left_top = vertices_paper[np.argmin(vertices_paper[:, 0] + vertices_paper[:, 1])]
B_left_bottom = vertices_paper[np.argmin(vertices_paper[:, 0] - vertices_paper[:, 1])]
B_right_top = vertices_paper[np.argmax(vertices_paper[:, 0] - vertices_paper[:, 1])]
B_right_bottom = vertices_paper[np.argmax(vertices_paper[:, 0] + vertices_paper[:, 1])]

left_top_x = A_left_top[0] if A_left_top[0] > B_left_top[0] else B_left_top[0]
left_top_y = A_left_top[1] if A_left_top[1] > B_left_top[1] else B_left_top[1]
left_top = np.array([left_top_x, left_top_y])
left_bottom_x = A_left_bottom[0] if A_left_bottom[0] > B_left_bottom[0] else B_left_bottom[0]
left_bottom_y = A_left_bottom[1] if A_left_bottom[1] < B_left_bottom[1] else B_left_bottom[1]
left_bottom = np.array([left_bottom_x, left_bottom_y])
right_top_x = A_right_top[0] if A_right_top[0] < B_right_top[0] else B_right_top[0]
right_top_y = A_right_top[1] if A_right_top[1] > B_right_top[1] else B_right_top[1]
right_top = np.array([right_top_x, right_top_y])
right_bottom_x = A_right_bottom[0] if A_right_bottom[0] < B_right_bottom[0] else B_right_bottom[0]
right_bottom_y = A_right_bottom[1] if A_right_bottom[1] < B_right_bottom[1] else B_right_bottom[1]
right_bottom = np.array([right_bottom_x, right_bottom_y])
mask_points = np.array([left_top, left_bottom, right_bottom, right_top])
mask = np.zeros(imageA.shape[:2], dtype=np.uint8)
cv2.fillPoly(mask, [mask_points], 255)
cv2.imwrite("mask.jpg", mask)
imageA = cv2.imread('top_paper_1.jpg')
imageB = cv2.imread('top_paper_2.jpg')
imageA_crop = cv2.bitwise_and(imageA, imageA, mask=mask)
imageB_crop = cv2.bitwise_and(imageB, imageB, mask=mask)
cv2.imwrite("crop_top_paper_1.jpg", imageA_crop)
cv2.imwrite("crop_top_paper_2.jpg", imageB_crop)




imgA = cv2.cvtColor(imageA_crop, cv2.COLOR_BGR2RGB)
imgB = cv2.cvtColor(imageB_crop, cv2.COLOR_BGR2RGB)

hA, wA, cA = imgA.shape[:3]
hB, wB, cA = imgB.shape [:3]

akaze = cv2.AKAZE_create()

kpA, desA = akaze.detectAndCompute(imgA,None)
kpB, desB = akaze.detectAndCompute(imgB,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(desA,desB)

matches = sorted(matches, key = lambda x:x.distance)

good = matches[:int(len(matches) * 0.15)]

src_pts = np.float32([kpA[m.queryIdx].pt for m in good]).reshape(-1,1,2)
dst_pts = np.float32([kpB[m.trainIdx].pt for m in good]).reshape(-1,1,2)

M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)
imgB_transform = cv2.warpPerspective(imgB, M, (wA, hA))

result = cv2.absdiff(imgA, imgB_transform)
result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

_, result_bin = cv2.threshold(result_gray, 50, 255, cv2.THRESH_BINARY)

kernel = np.ones((2,2),np.uint8)

result_bin = cv2.morphologyEx(result_bin, cv2.MORPH_OPEN, kernel)

result_bin_rgb = cv2.cvtColor(result_bin, cv2.COLOR_GRAY2RGB)
result_add = cv2.addWeighted(imgA, 0.3, result_bin_rgb, 0.7, 2.2)
cv2.imwrite("result.jpg", result_bin)

# detect pattern change
valid_column = []
for i in range(result_bin.shape[1]):
    valid_row = 0
    for j in range(result_bin.shape[0]):
        if result_bin[j, i] == 255:
            valid_row += 1
    if valid_row > 50:
        valid_column.append(i)
for v_valid_column in valid_column:
    cv2.line(result_add, (v_valid_column,0), (v_valid_column,479), (0, 0,255))
cv2.imwrite("line.jpg", result_add)
