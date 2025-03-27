import argparse
import os
import sys

import json
import numpy as np
import math
import json
import torch
from PIL import Image

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))


# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

import open3d as o3d

# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)


def get_average_depth(depth_img, x, y, window_size=10):
    half_size = window_size // 2
    x_min = max(x - half_size, 0)
    x_max = min(x + half_size + 1, depth_img.shape[0])
    y_min = max(y - half_size, 0)
    y_max = min(y + half_size + 1, depth_img.shape[1])

    region = depth_img[y_min:y_max, x_min:x_max]
    # valid_region = region[np.isfinite(region) & (region > 0)]
    valid_region = region[np.isfinite(region)]

    return np.mean(valid_region)


def coords_to_depth(depth_img, img_x, img_y):
    x = (img_x - cx) / fx
    y = (img_y - cy) / fy
    z = depth_img[int(img_y.round()), int(img_x.round())]
    # z = get_average_depth(depth_img, int(img_x.round()), int(img_y.round()))
    # z = depth_img[int(img_x.round()),  int(img_y.round())]
    # z = depth_img.reshape(-1)[int(track_y.round()) * width + int(track_x.round())]
    x *= z
    y *= z
    return (x, y, z)

def normalize(v):
    return v / np.linalg.norm(v)


if __name__ == "__main__":
    directory_path = "ros/fold/rgb_image"
    files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    latest_file = max(files, key=os.path.getmtime)
    latest_file_name = os.path.basename(latest_file)
    latest_file_path = os.path.join(directory_path, f'{latest_file_name}')

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=False, help="path to config file", default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=False, help="path to checkpoint file", default="groundingdino_swint_ogc.pth"
    )
    parser.add_argument(
        "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file", default="sam_vit_h_4b8939.pth"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--input_image", type=str, required=False, help="path to image file", default=latest_file_path)
    parser.add_argument("--text_prompt", type=str, required=False, help="text prompt", default="floral pattern box. paper.")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="ros/fold/outputs", required=False, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold",)
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold",)

    parser.add_argument("--device", type=str, default="cuda", help="running on cpu only!, default=False")
    parser.add_argument("--bert_base_uncased_path", type=str, required=False, help="bert_base_uncased model path, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    image_path = args.input_image
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device
    bert_base_uncased_path = args.bert_base_uncased_path

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil, image = load_image(image_path)
    # load model
    model = load_model(config_file, grounded_checkpoint, bert_base_uncased_path, device=device)

    # visualize raw image
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, device=device
    )


    # initialize SAM
    if use_sam_hq:
        predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )

    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)

    plt.axis('off')
    plt.savefig(
        os.path.join(output_dir, "grounded_sam_output.jpg"),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )

    save_mask_data(output_dir, masks, boxes_filt, pred_phrases)

    # extract box and paper mask
    boxes_filt_np = boxes_filt.cpu().numpy()
    box_indices = [index for index, item in enumerate(pred_phrases) if 'box' in item]
    box_area = abs(boxes_filt_np[box_indices[0]][2] - boxes_filt_np[box_indices[0]][0]) * np.abs(boxes_filt_np[box_indices[0]][3] - boxes_filt_np[box_indices[0]][1])
    mask_box = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    mask_np_box = masks[box_indices[0]].cpu().numpy()[0]
    mask_box[mask_np_box] = 255
    x1 = int(min(boxes_filt_np[box_indices[0]][0], boxes_filt_np[box_indices[0]][2]))
    x2 = int(max(boxes_filt_np[box_indices[0]][0], boxes_filt_np[box_indices[0]][2]))
    y1 = int(min(boxes_filt_np[box_indices[0]][1], boxes_filt_np[box_indices[0]][3]))
    y2 = int(max(boxes_filt_np[box_indices[0]][1], boxes_filt_np[box_indices[0]][3]))
    cv2.imwrite(os.path.join(output_dir, 'box_mask.png'), mask_box)

    # paper mask
    paper_indices = [index for index, item in enumerate(pred_phrases) if 'paper' in item]
    paper_masks = []
    for i in(paper_indices):
        paper_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        paper_mask_np = masks[i].cpu().numpy()[0]
        paper_mask[paper_mask_np] = 255
        # fix_paper_mask[white_mask] = 0
        paper_masks.append(paper_mask)
    paper_masks.append(mask_box)

    sum_paper_mask = np.maximum.reduce(paper_masks)
    only_paper_mask = np.copy(sum_paper_mask)
    only_paper_mask[(sum_paper_mask == 255) & (mask_box == 255)] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    sum_paper_mask = cv2.morphologyEx(sum_paper_mask, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(os.path.join(output_dir, 'paper_mask.png'), sum_paper_mask)
    cv2.imwrite(os.path.join(output_dir, 'only_paper_mask.png'), only_paper_mask)

    # get 3d info of box
    # camera info
    camera_info_file_path = os.path.join("ros/fold", "camera_info.txt")
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

    box_points = []
    depth_image_directory_path = "ros/fold/depth_image"
    depth_image_files = [f for f in os.listdir(depth_image_directory_path) if f.endswith(".png")]
    files_with_timestamp = [(f, os.path.getmtime(os.path.join(depth_image_directory_path, f))) for f in depth_image_files]
    latest_depth_image_file = max(files_with_timestamp, key=lambda x: x[1])
    depth_image = o3d.io.read_image(os.path.join(depth_image_directory_path, latest_depth_image_file[0]))
    # depth_image = cv2.imread(os.path.join(depth_image_directory_path, latest_depth_image_file[0]), cv2.IMREAD_UNCHANGED)
    depth_image = np.asarray(depth_image, dtype=np.float32)
    box_points_indices = np.where(mask_box == 255)
    for i in range(len(box_points_indices[0])):
        box_points.append(coords_to_depth(depth_image, box_points_indices[1][i], box_points_indices[0][i]))
    box_points = np.array(box_points)

    # transformation
    base_to_camera_transformation_translation = np.asarray([0.099087, 0.020357, 0.56758])
    base_to_camera_transformation_rotation = np.asarray([[0.017396, -0.896664, 0.442385], [-0.999853, 0.014582, -0.009762], [0.002302, -0.442487, -0.89678]])
    base_to_camera_transformation = np.eye(4)
    base_to_camera_transformation[:3, :3] = base_to_camera_transformation_rotation
    base_to_camera_transformation[:3, 3] = base_to_camera_transformation_translation

    box_points_ones = np.ones((box_points.shape[0], 1))
    box_points_homogeneous = np.hstack([box_points, box_points_ones])
    box_points_base_coords_homogeneous = (base_to_camera_transformation @ box_points_homogeneous.T).T
    box_points_base_coords = box_points_base_coords_homogeneous[:, :3]
    box_points_base_coords = box_points_base_coords[~np.isnan(box_points_base_coords).any(axis=1)]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(box_points_base_coords)
    plane_model, inliers = point_cloud.segment_plane(distance_threshold=10, ransac_n=3, num_iterations=1000)
    plane_points = point_cloud.select_by_index(inliers)
    plane_points_np = np.asarray(plane_points.points)
    box_points_base_coords = np.asarray(plane_points.points)

    # center point
    normal = np.array(plane_model[:3])
    normal = normal / np.linalg.norm(normal)
    centroid = np.mean(plane_points_np, axis=0)
    shifted_points = plane_points_np - centroid
    u, s, vh = np.linalg.svd(shifted_points)
    basis_x = vh[0]
    basis_y = vh[1]
    projected_2d = np.dot(shifted_points, np.vstack((basis_x, basis_y)).T)
    min_2d = np.min(projected_2d, axis=0)
    max_2d = np.max(projected_2d, axis=0)
    center_2d = (min_2d + max_2d) / 2
    center_point = centroid + center_2d[0] * basis_x + center_2d[1] * basis_y

    # plane length
    min_bound = projected_2d.min(axis=0)
    max_bound = projected_2d.max(axis=0)
    width = max_bound[0] - min_bound[0]
    height = max_bound[1] - min_bound[1]

    # paper
    contours_paper, _ = cv2.findContours(sum_paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour_paper = max(contours_paper, key=cv2.contourArea)
    contour_image = sum_paper_mask.copy()
    contour_image = cv2.cvtColor(contour_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, contours_paper, -1, (0, 255, 0), 10)
    cv2.imwrite("ros/fold/outputs/contour_image.png", contour_image)
    epsilon = 0.02 * cv2.arcLength(max_contour_paper, True)
    approx = cv2.approxPolyDP(max_contour_paper, epsilon, True)
    linear_approx_image = sum_paper_mask.copy()
    linear_approx_image = cv2.cvtColor(linear_approx_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(linear_approx_image, approx, -1, (0, 255, 0), 20)
    cv2.imwrite("ros/fold/outputs/linear_approx_image.png", linear_approx_image)
    if len(approx) != 4:
        raise ValueError("Not a rectangular: num of points: {}".format(len(approx)))
    vertices_paper = approx[:, 0, :]
    paper_points = []
    for i in range(4):
        paper_points.append(coords_to_depth(depth_image, vertices_paper[i][0], vertices_paper[i][1]))
    paper_points = np.array(paper_points)

    # transformation
    paper_points_ones = np.ones((paper_points.shape[0], 1))
    paper_points_homogeneous = np.hstack([paper_points, paper_points_ones])
    paper_points_base_coords_homogeneous = (base_to_camera_transformation @ paper_points_homogeneous.T).T
    paper_points_base_coords = paper_points_base_coords_homogeneous[:, :3]
    paper_points_base_coords = paper_points_base_coords[~np.isnan(paper_points_base_coords).any(axis=1)]

    # graph
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    x = box_points_base_coords[:, 0]
    y = box_points_base_coords[:, 1]
    z = box_points_base_coords[:, 2]
    ax.scatter(x, y, z, c='b', marker='o', alpha=0.5, s=0.1)
    x_paper = paper_points_base_coords[:, 0]
    y_paper = paper_points_base_coords[:, 1]
    z_paper = paper_points_base_coords[:, 2]
    ax.scatter(x_paper, y_paper, z_paper, c='orange', marker='o', alpha=1.0, s=50)
    num_points = 100
    for i in range(4):
        x_values = np.linspace(paper_points_base_coords[i%4][0], paper_points_base_coords[(i+1)%4][0], num_points)
        y_values = np.linspace(paper_points_base_coords[i%4][1], paper_points_base_coords[(i+1)%4][1], num_points)
        z_values = np.linspace(paper_points_base_coords[i%4][2], paper_points_base_coords[(i+1)%4][2], num_points)
        ax.scatter(x_values, y_values, z_values, c='orange', s=10)
    # draw 3d bbox
    box_cx = center_point[0]
    box_cy = center_point[1]
    # calculate box depth
    box_top_face_average_z = np.mean(box_points_base_coords, axis=0)[2]
    paper_average_z = np.mean(paper_points_base_coords, axis=0)[2]
    depth = box_top_face_average_z - paper_average_z
    box_cz = paper_average_z + depth / 2

    # box_cz = box_top_face_average_z
    ax.scatter(box_cx, box_cy, box_cz, c='r', marker='o', alpha=1.0, s=200)
    x = [box_cx - width / 2, box_cx + width / 2]
    y = [box_cy - height / 2, box_cy + height / 2]
    z = [box_cz - depth / 2, box_cz + depth / 2]
    vertices = np.array([
        [x[0], y[0], z[0]],
        [x[1], y[0], z[0]],
        [x[1], y[1], z[0]],
        [x[0], y[1], z[0]],
        [x[0], y[0], z[1]],
        [x[1], y[0], z[1]],
        [x[1], y[1], z[1]],
        [x[0], y[1], z[1]],
    ])
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
    ]
    for edge in edges:
        start, end = edge
        ax.plot(
            [vertices[start][0], vertices[end][0]],
            [vertices[start][1], vertices[end][1]],
            [vertices[start][2], vertices[end][2]],
            color="b"
        )
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color="b")

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.axis('equal')
    plt.show()

    # info
    box_center = np.array([box_cx, box_cy, box_cz])
    box_length = np.array([width, height, depth])
    paper_center = np.mean(paper_points_base_coords, axis=0)
    distances = []
    for i in range(len(paper_points_base_coords)):
        for j in range(i + 1, len(paper_points_base_coords)):
            dist = np.linalg.norm(paper_points_base_coords[i] - paper_points_base_coords[j])
            distances.append(dist)
    distances = np.sort(distances)
    print(distances)
    paper_length = np.array([distances[0], distances[2]])
    data = {"box_center": box_center.tolist(),
            "paper_center": paper_center.tolist(),
            "box_point": vertices.tolist(),
            "paper_point": paper_points_base_coords.tolist(),
            "box_length": box_length.tolist(),
            "paper_length": paper_length.tolist(),
    }
    with open(os.path.join("ros/fold", "target_object_info.yaml"), "w") as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)
