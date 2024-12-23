import cv2
import json
import math
import numpy as np
import open3d as o3d
import os
import yaml

def get_average_depth(depth_img, x, y, window_size=10):
    half_size = window_size // 2
    x_min = max(x - half_size, 0)
    x_max = min(x + half_size + 1, depth_img.shape[0])
    y_min = max(y - half_size, 0)
    y_max = min(y + half_size + 1, depth_img.shape[1])

    region = depth_img[y_min:y_max, x_min:x_max]
    valid_region = region[np.isfinite(region) & (region > 0)]

    return np.mean(valid_region)

def coords_to_depth(depth_img, img_x, img_y):
    x = (img_x - cx) / fx
    y = (img_y - cy) / fy
    z = get_average_depth(depth_img, int(img_x.round()), int(img_y.round()))
    # z = depth_img[int(img_x.round()),  int(img_y.round())]
    # z = depth_img.reshape(-1)[int(track_y.round()) * width + int(track_x.round())]
    x *= z
    y *= z
    return (x, y, z)

def transformation(point):
    base_to_camera_transformation_translation = np.asarray([0.099087, 0.020357, 0.56758])
    base_to_camera_transformation_rotation = np.asarray([[0.017396, -0.896664, 0.442385], [-0.999853, 0.014582, -0.009762], [0.002302, -0.442487, -0.89678]])
    base_to_camera_transformation = np.eye(4)
    base_to_camera_transformation[:3, :3] = base_to_camera_transformation_rotation
    base_to_camera_transformation[:3, 3] = base_to_camera_transformation_translation

    point_ones = np.ones((point.shape[0], 1))
    point_homogeneous = np.hstack([point, point_ones])
    point_base_coords_homogeneous = (base_to_camera_transformation @ point_homogeneous.T).T
    point_base_coords = point_base_coords_homogeneous[:, :3]
    return point_base_coords

def normalize(v):
    return v / np.linalg.norm(v)


image_path = "ros/rope/outputs/box_mask.png"
mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

kernel = np.ones((13, 13), np.uint8)
filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

largest_contour = max(contours, key=cv2.contourArea)

output_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
cv2.drawContours(output_image, [largest_contour], -1, (0, 255, 0), 2)

output_path = "ros/rope/cross/contour_result.jpg"
cv2.imwrite(output_path, output_image)

# extract
contour_mask = np.zeros_like(mask)
cv2.drawContours(contour_mask, [largest_contour], -1, (255), thickness=-1)
inverted_region = cv2.bitwise_not(mask, mask=contour_mask)
output_image = np.zeros_like(mask)
output_image[contour_mask == 255] = inverted_region[contour_mask == 255]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
output_image = cv2.morphologyEx(output_image, cv2.MORPH_CLOSE, kernel)
output_path = "ros/rope/cross/cross.png"
cv2.imwrite(output_path, output_image)

sum_ribbon_mask = cv2.imread("ros/rope/cross/cross.png", cv2.IMREAD_GRAYSCALE)
contours, _ = cv2.findContours(sum_ribbon_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour = max(contours, key=cv2.contourArea)

# contour_points = contour.reshape(-1, 2)
contour_points = np.array(contour, dtype=np.float64).reshape((contour.shape[0], contour.shape[2]))
mean, eigenvectors, eigenvalues = cv2.PCACompute2(contour_points, mean=None)
if abs(eigenvectors[0][0]) > abs(eigenvectors[0][1]):
    major_axis = eigenvectors[0] # horizontal
    minor_axis = eigenvectors[1] # vertical
else:
    major_axis = eigenvectors[1] # horizontal
    minor_axis = eigenvectors[0] # vertical
cx, cy = mean[0][0], mean[0][1]

vertical_points = []
horizontal_points = []
for contour_point in contour_points:
    x, y = contour_point
    projection_major = (x - cx) * major_axis[0] + (y - cy) * major_axis[1]
    projection_minor = (x - cx) * minor_axis[0] + (y - cy) * minor_axis[1]
    if abs(projection_major) > abs(projection_minor):
        vertical_points.append(contour_point)
    else:
        horizontal_points.append(contour_point)

regions = [[] for _ in range(4)]
for point in vertical_points:
    x, y = point
    if x > cx:
        regions[0].append(point)
    else:
        regions[2].append(point)
for point in horizontal_points:
    x, y = point
    if y > cy:
        regions[1].append(point)
    else:
        regions[3].append(point)

image = cv2.imread("ros/rope/cross/cross.png")
output = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
lines = []
arrow_point = []
for i, region in enumerate(regions):
    # region_contour = np.array(region).reshape(-1, 1, 2)
    region_contour = np.array((region)).reshape((-1, 1, 2)).astype(np.int32)
    cv2.drawContours(image, [region_contour], -1, colors[i], 2)
    region_contour = np.array(region_contour, dtype=np.float64).reshape((region_contour.shape[0], region_contour.shape[2]))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(region_contour, mean=None)
    scale = 0.02
    end_point = [int(mean[0][0] + scale * eigenvectors[0][0] * eigenvalues[0][0]), int(mean[0][1] + scale * eigenvectors[0][1] * eigenvalues[0][0])]
    lines.append({"p": np.array(mean[0]),"d": np.array(eigenvectors[0])})
    arrow_point.append([int(mean[0][0]), int(mean[0][1]), end_point[0], end_point[1]])
    cv2.arrowedLine(image, (int(mean[0][0]), int(mean[0][1])), end_point, (255, 0, 255), thickness=3)

A = []
b = []
for line in lines:
    d = line["d"] / np.linalg.norm(line["d"])
    A.append(np.eye(2) - np.outer(d, d))
    b.append((np.eye(2) - np.outer(d, d)) @ line["p"])
A = np.sum(A, axis=0)
b = np.sum(b, axis=0)
q = np.linalg.solve(A, b)
cv2.circle(image, [int(q[0]), int(q[1])], 10, (0, 255, 255), -1)

cv2.imwrite("ros/rope/cross/Resions.jpg", image)

# compute depth of knot
depth_image_directory_path = "ros/standard_box/depth_image"
depth_image_files = [f for f in os.listdir(depth_image_directory_path) if f.endswith(".png")]
files_with_timestamp = [(f, os.path.getmtime(os.path.join(depth_image_directory_path, f))) for f in depth_image_files]
latest_depth_image_file = max(files_with_timestamp, key=lambda x: x[1])
depth_image = o3d.io.read_image(os.path.join(depth_image_directory_path, latest_depth_image_file[0]))
depth_image = np.asarray(depth_image, dtype=np.float32)

# camera info
camera_info_file_path = os.path.join("ros/standard_box", "camera_info.txt")
with open(camera_info_file_path, "r") as file:
    camera_info_lines = file.readlines()
camera_K_values = []
for line in camera_info_lines:
    if line.strip().startswith("K:"):
        camera_K_values = eval(line.split("K:")[1].strip())
        break
fx = camera_K_values[0]
fy = camera_K_values[4]
cx = camera_K_values[2]
cy = camera_K_values[5]

knot_point = np.array(coords_to_depth(depth_image, q[0], q[1])).reshape(1, 3)
knot_point_base_coords = transformation(knot_point)

with open(os.path.join("ros/standard_box", "output.yaml"), "r") as file:
    yaml_data = yaml.safe_load(file)

# distance between standard box's centroid and knot
standard_box_centroid = np.array(yaml_data.get("centroid"))
print(knot_point_base_coords)
# distance = math.sqrt((standard_box_centroid[0] - knot_point_base_coords[0][0]) ** 2 + (standard_box_centroid[1] - knot_point_base_coords[0][1]) ** 2 + (standard_box_centroid[2] - knot_point_base_coords[0][2]) ** 2)
distance = math.sqrt((standard_box_centroid[0] - knot_point_base_coords[0][0]) ** 2 + (standard_box_centroid[1] - knot_point_base_coords[0][1]) ** 2)

# compare standard box and arrow direction
arrow_point_3d = []
for i in range(4):
    start_3d_arrow_point = np.array(coords_to_depth(depth_image, np.float32(arrow_point[i][0]), np.float32(arrow_point[i][1])))
    end_3d_arrow_point = np.array(coords_to_depth(depth_image, np.float32(arrow_point[i][2]), np.float32(arrow_point[i][3])))
    arrow_point_3d.append([start_3d_arrow_point, end_3d_arrow_point])
arrow_directions = [pair[1] - pair[0] for pair in arrow_point_3d]
normalized_arrow_directions = [v[:2] / np.linalg.norm(v[:2]) for v in arrow_directions]
print(normalized_arrow_directions)
# normalize and caluculate angle with "tate" of standard box
tate = normalize(np.array(yaml_data.get("tate")))[:2]
angles_deg = []
for arrow in normalized_arrow_directions:
    # arrow_normalized = normalize(arrow)
    cos_theta = np.dot(arrow, tate)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_deg = np.degrees(angle)
    angles_deg.append(angle_deg.tolist())

# evaluation of string slack
cosine_yoko = np.dot(normalized_arrow_directions[0], normalized_arrow_directions[2])
angle_yoko = np.degrees(np.arccos(np.clip(cosine_yoko, -1.0, 1.0)))
cosine_tate = np.dot(normalized_arrow_directions[1], normalized_arrow_directions[3])
angle_tate = np.degrees(np.arccos(np.clip(cosine_tate, -1.0, 1.0)))
# write results in yaml file
data = {"distance": distance,
        "angles_deg": angles_deg,
        "angle_yoko": angle_yoko.tolist(),
        "angle_tate": angle_tate.tolist()
}
with open(os.path.join("ros/rope/cross", "results.yaml"), "w") as yaml_file:
    yaml.dump(data, yaml_file, default_flow_style=False)
with open("ros/rope/cross/output.txt", "w") as file:
    json.dump(data, file)
