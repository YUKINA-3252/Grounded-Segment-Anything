import argparse
import os
import sys

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

import yaml


# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt

import open3d as o3d
from sklearn.decomposition import PCA


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
    return [x, y, z]

def point_to_plane_distance(point, normal, d):
    return abs(np.dot(normal, point) + d) / np.linalg.norm(normal)

if __name__ == "__main__":
    directory_path = "ros/standard_box/rgb_image"
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
    parser.add_argument("--text_prompt", type=str, required=False, help="text prompt", default="box with stripe pattern")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="ros/standard_box/outputs", required=False, help="output directory"
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

    standard_box_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    standard_box_mask_np = masks[0].cpu().numpy()[0]
    standard_box_mask[standard_box_mask_np] = 255
    cv2.imwrite(os.path.join(output_dir, 'standard_box_mask.png'), standard_box_mask)

    # fill
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    output_image = cv2.morphologyEx(standard_box_mask, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("ros/standard_box/outputs/standard_box_mask_fill.png", output_image)
    standard_box_mask = output_image

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

    standard_box_points = []
    depth_image_directory_path = "ros/standard_box/depth_image"
    depth_image_files = [f for f in os.listdir(depth_image_directory_path) if f.endswith(".png")]
    files_with_timestamp = [(f, os.path.getmtime(os.path.join(depth_image_directory_path, f))) for f in depth_image_files]
    latest_depth_image_file = max(files_with_timestamp, key=lambda x: x[1])
    depth_image = o3d.io.read_image(os.path.join(depth_image_directory_path, latest_depth_image_file[0]))
    # depth_image = cv2.imread(os.path.join(depth_image_directory_path, latest_depth_image_file[0]), cv2.IMREAD_UNCHANGED)
    depth_image = np.asarray(depth_image, dtype=np.float32)
    standard_box_points_indices = np.where(standard_box_mask == 255)
    for i in range(len(standard_box_points_indices[0])):
        standard_box_points.append(coords_to_depth(depth_image, standard_box_points_indices[1][i], standard_box_points_indices[0][i]))
    standard_box_points = np.array(standard_box_points)

    # transformation
    base_to_camera_transformation_translation = np.asarray([0.099087, 0.020357, 0.56758])
    base_to_camera_transformation_rotation = np.asarray([[0.017396, -0.896664, 0.442385], [-0.999853, 0.014582, -0.009762], [0.002302, -0.442487, -0.89678]])
    base_to_camera_transformation = np.eye(4)
    base_to_camera_transformation[:3, :3] = base_to_camera_transformation_rotation
    base_to_camera_transformation[:3, 3] = base_to_camera_transformation_translation

    standard_box_points_ones = np.ones((standard_box_points.shape[0], 1))
    standard_box_points_homogeneous = np.hstack([standard_box_points, standard_box_points_ones])
    standard_box_points_base_coords_homogeneous = (base_to_camera_transformation @ standard_box_points_homogeneous.T).T
    standard_box_points_base_coords = standard_box_points_base_coords_homogeneous[:, :3]

    standard_box_points_base_coords = standard_box_points_base_coords[~np.isnan(standard_box_points_base_coords).any(axis=1)]
    centroid = np.mean(standard_box_points_base_coords, axis=0)
    shifted_standard_box_points_base_coords = standard_box_points_base_coords - centroid
    # cov_matrix = np.dot(shifted_standard_box_points_base_coords.T, shifted_standard_box_points_base_coords)
    # eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # normal_vector = eigenvectors[:, 0]
    # _, _, vh = np.linalg.svd(shifted_standard_box_points_base_coords)
    # normal_vector = vh[-1]
    # d = -np.dot(normal_vector, centroid)

    # threshold = 15

    # inliers = []
    # for point in standard_box_points_base_coords:
    #     if point_to_plane_distance(point, normal_vector, d) <= threshold:
    #         inliers.append(point)

    # standard_box_points_base_coords = np.array(inliers)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(standard_box_points_base_coords)
    plane_model, inliers = point_cloud.segment_plane(distance_threshold=10,
                                                 ransac_n=3,
                                                 num_iterations=1000)
    plane_points = point_cloud.select_by_index(inliers)
    plane_points_np = np.asarray(plane_points.points)
    standard_box_points_base_coords = np.asarray(plane_points.points)

    # center point
    normal = np.array(plane_model[:3])
    normal = normal / np.linalg.norm(normal)
    centroid = np.mean(plane_points_np, axis=0)
    shifted_points = plane_points_np - centroid
    u, s, vh = np.linalg.svd(shifted_points)

    basis_x = vh[0]
    basis_y = vh[1]
    tate = basis_x / np.linalg.norm(vh[0])
    yoko = basis_y / np.linalg.norm(vh[1])

    projected_2d = np.dot(shifted_points, np.vstack((basis_x, basis_y)).T)

    min_2d = np.min(projected_2d, axis=0)
    max_2d = np.max(projected_2d, axis=0)

    center_2d = (min_2d + max_2d) / 2

    center_point = centroid + center_2d[0] * basis_x + center_2d[1] * basis_y

    data = {"centroid": center_point.tolist(),
            "tate": tate.tolist(),
            "yoko": yoko.tolist()
            }
    with open(os.path.join("ros/standard_box", "output.yaml"), "w") as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)

    # graph
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    x = standard_box_points_base_coords[:, 0]
    y = standard_box_points_base_coords[:, 1]
    z = standard_box_points_base_coords[:, 2]
    ax.scatter(x, y, z, c='b', marker='o', alpha=0.5, s=0.1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.axis('equal')

    ax.scatter(center_point[0], center_point[1], center_point[2], c='r', marker='o', s=100)
    # knot_point =[324.48811174, 48.43470071, -546.95052783]
    # ax.scatter(knot_point[0], knot_point[1], knot_point[2], c='g', marker='o', s=800)
    # plt.show()
    print(center_point)
