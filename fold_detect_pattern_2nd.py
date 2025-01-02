import argparse
import os
import sys

import json
import numpy as np
import math
import json
import torch
from PIL import Image
from skimage.feature.texture import graycomatrix, graycoprops

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

from sklearn.preprocessing import StandardScaler
from PIL import Image

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

def detect_discontinuity_in_region(image, mask, patch_size=20, threshold=0.3):
    h, w, _ = image.shape
    discontinuity_map = np.zeros((h, w), dtype=np.uint8)

    binary_mask = (mask > 0).astype(np.uint8)

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch_mask = binary_mask[y:y + patch_size, x:x + patch_size]
            if np.sum(patch_mask) == 0:
                continue

            patch = lab_image[y:y + patch_size, x:x + patch_size]
            patch_mean = np.mean(patch, axis=(0, 1))

            if x + patch_size < w:
                right_patch = lab_image[y:y + patch_size, x + patch_size:x + 2 * patch_size]
                right_patch_mask = binary_mask[y:y + patch_size, x + patch_size:x + 2 * patch_size]
                if np.sum(right_patch_mask) > 0:
                    right_mean = np.mean(right_patch, axis=(0, 1))
                    diff_right = np.linalg.norm(patch_mean - right_mean)
                    if diff_right > threshold * 100:
                        discontinuity_map[y:y + patch_size, x + patch_size - patch_size // 2:x + patch_size] = 255

            if y + patch_size < h:
                bottom_patch = lab_image[y + patch_size:y + 2 * patch_size, x:x + patch_size]
                bottom_patch_mask = binary_mask[y + patch_size:y + 2 * patch_size, x:x + patch_size]
                if np.sum(bottom_patch_mask) > 0:
                    bottom_mean = np.mean(bottom_patch, axis=(0, 1))
                    diff_bottom = np.linalg.norm(patch_mean - bottom_mean)
                    if diff_bottom > threshold * 100:
                        discontinuity_map[y + patch_size - patch_size // 2:y + patch_size, x:x + patch_size] = 255

    result_image = image.copy()
    result_image[discontinuity_map == 255] = [0, 0, 255]

    return result_image


def detect_edges_in_region(image, mask, low_threshold=50, high_threshold=150):
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    masked_image = cv2.bitwise_and(gray_image, gray_image, mask=mask)

    edges = cv2.Canny(masked_image, low_threshold, high_threshold)

    return edges

def normalize_features(arr):
    normalized_arr = np.zeros_like(arr)
    for i in range(3):
        min_val = arr[..., i].min()
        max_val = arr[..., i].max()
        normalized_arr[..., i] = (arr[..., i] - min_val) / (max_val - min_val)
    return normalized_arr

def detect_horizontal_changes(image, mask, num_slices=30, threshold=30):
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    h, w = masked_image.shape[:2]
    print(h, w)
    slice_width = w // num_slices

    features = []
    for i in range(num_slices):
        x_start = i * slice_width
        x_end = (i + 1) * slice_width if i < num_slices - 1 else w
        cluster = masked_image[:, x_start:x_end]
        features.append(calculate_features(cluster))
    distances = []
    change_regions = []
    for i in range(len(features) - 1):
        distance = np.linalg.norm(features[i] - features[i + 1])
        distances.append(distance)
        if distance > threshold:
            x_start = i * slice_width
            x_end = (i + 1) * slice_width
            change_regions.append((x_start, x_end))
    return distances, change_regions


if __name__ == "__main__":
    directory_path = "ros/fold/pattern/rgb"
    latest_file_path = os.path.join(directory_path, 'top_paper_2.jpg')

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
    parser.add_argument("--text_prompt", type=str, required=False, help="text prompt", default="floral patterned-paper")
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

    # paper mask
    paper_indices = [index for index, item in enumerate(pred_phrases) if 'paper' and 'pattern' in item]
    paper_masks = []
    for i in(paper_indices):
        paper_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        paper_mask_np = masks[i].cpu().numpy()[0]
        paper_mask[paper_mask_np] = 255
        paper_masks.append(paper_mask)

    sum_paper_mask = np.maximum.reduce(paper_masks)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    sum_paper_mask = cv2.morphologyEx(sum_paper_mask, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(os.path.join('ros/fold/pattern/mask_top_paper_2.png'), sum_paper_mask)

    top_paper_2 = cv2.bitwise_and(image, image, mask=sum_paper_mask)
    top_paper_2 = cv2.cvtColor(top_paper_2, cv2.COLOR_BGR2RGB)
    cv2.imwrite("ros/fold/pattern/top_paper_2.jpg", top_paper_2)

    # contours, _ = cv2.findContours(sum_paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # max_contour_paper = max(contours, key=cv2.contourArea)
    # hull = cv2.convexHull(max_contour_paper)
    # hull_mask = np.zeros_like(sum_paper_mask)
    # cv2.drawContours(hull_mask, [hull], 0, 255, -1)
    # convex_defects = cv2.subtract(hull_mask, sum_paper_mask)
    # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(convex_defects)
    # sizes = stats[:, cv2.CC_STAT_AREA]
    # largest_label = 1 + np.argmax(sizes[1:])
    # largest_convex_defects = np.zeros_like(convex_defects)
    # largest_convex_defects[labels == largest_label] = 255
    # visualization = cv2.cvtColor(largest_convex_defects, cv2.COLOR_GRAY2BGR)
    # cv2.imwrite("ros/fold/non.jpg", visualization)
    # coordinates = np.column_stack(np.where(largest_convex_defects > 0))
    # max_x_idx = np.argmax(coordinates[:, 1])
    # max_x_point = coordinates[max_x_idx]
    # # min_x_idx = np.argmin(coordinates[:, 1])
    # # min_x_point = coordinates[min_x_idx]
    # # extract paper mask
    # extracted_sum_paper_mask = np.zeros_like(sum_paper_mask)
    # coordinates = np.column_stack(np.where(sum_paper_mask > 0))
    # for y, x in coordinates:
    #     if x >= max_x_point[1]:
    #     # if x <= min_x_point[1]:
    #         extracted_sum_paper_mask[y, x] = 255
    # extracted_img = cv2.cvtColor(extracted_sum_paper_mask, cv2.COLOR_GRAY2BGR)
    # cv2.imwrite("ros/fold/pattern/extracted.jpg", extracted_img)
    # contours, _ = cv2.findContours(extracted_sum_paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # max_contour_paper = max(contours, key=cv2.contourArea)
    # epsilon = 0.02 * cv2.arcLength(max_contour_paper, True)
    # approx = cv2.approxPolyDP(max_contour_paper, epsilon, True)
    # if len(approx) != 4:
    #     raise ValueError("Not a rectangular: num of points: {}".format(len(approx)))
    # vertices_paper = approx[:, 0, :]
    # left_top = vertices_paper[np.argmin(vertices_paper[:, 0] + vertices_paper[:, 1])]
    # right_top = vertices_paper[np.argmax(vertices_paper[:, 0] - vertices_paper[:, 1])]
    # left_bottom = vertices_paper[np.argmin(vertices_paper[:, 0] - vertices_paper[:, 1])]
    # right_bottom = vertices_paper[np.argmax(vertices_paper[:, 0] + vertices_paper[:, 1])]
    # pts = np.float32([left_top, right_top, left_bottom, right_bottom])

    # # depth
    # # camera info
    # camera_info_file_path = os.path.join("ros/fold", "camera_info.txt")
    # with open(camera_info_file_path, "r") as file:
    #     camera_info_lines = file.readlines()
    # camera_K_values = []
    # for line in camera_info_lines:
    #     if line.strip().startswith("K:"):
    #         camera_K_values = eval(line.split("K:")[1].strip())
    #         break
    # fx = camera_K_values[0]
    # fy = camera_K_values[4]
    # cx = camera_K_values[2]
    # cy = camera_K_values[5]

    # depth_image_directory_path = "ros/fold/pattern/depth/2nd_tape_line.png"
    # depth_image = o3d.io.read_image(depth_image_directory_path)
    # depth_image = np.asarray(depth_image, dtype=np.float32)

    # pts_3d = []
    # for i in range(4):
    #     pts_3d.append(coords_to_depth(depth_image, pts[i][0], pts[i][1]))
    # pts_3d = np.array(pts_3d)

    # # transformation
    # base_to_camera_transformation_translation = np.asarray([0.099087, 0.020357, 0.56758])
    # base_to_camera_transformation_rotation = np.asarray([[0.017396, -0.896664, 0.442385], [-0.999853, 0.014582, -0.009762], [0.002302, -0.442487, -0.89678]])
    # base_to_camera_transformation = np.eye(4)
    # base_to_camera_transformation[:3, :3] = base_to_camera_transformation_rotation
    # base_to_camera_transformation[:3, 3] = base_to_camera_transformation_translation

    # pts_ones = np.ones((pts_3d.shape[0], 1))
    # pts_homogeneous = np.hstack([pts_3d, pts_ones])
    # pts_base_coords_homogeneous = (base_to_camera_transformation @ pts_homogeneous.T).T
    # pts_base_coords = pts_base_coords_homogeneous[:, :3]
    # pts_base_coords = pts_base_coords[~np.isnan(pts_base_coords).any(axis=1)]
    # print(pts_base_coords)

    # image_for_dpi = Image.open('ros/fold/pattern/rgb/2nd_tape_line.jpg')
    # dpi = image_for_dpi.info.get('dpi', (72, 72))

    # width = int(pts_base_coords[np.argmax(pts_base_coords[:, 1])][1] - pts_base_coords[np.argmin(pts_base_coords[:, 1])][1])
    # height = int(pts_base_coords[np.argmax(pts_base_coords[:, 0])][0] - pts_base_coords[np.argmin(pts_base_coords[:, 0])][0])
    # pixel_width = int(width / 25.4 * 72)
    # pixel_height = int(height / 25.4 * 72)

    # pts = np.float32([[left_top], [right_top], [left_bottom], [right_bottom]])
    # dst_pts = np.array([[[0, 0]], [[pixel_width, 0]], [[0, pixel_height]], [[pixel_width, pixel_height]]], dtype="int32")
    # matrix = cv2.getPerspectiveTransform(pts, np.float32(dst_pts))
    # result = cv2.warpPerspective(image, matrix, (pixel_width, pixel_height))
    # result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("ros/fold/pattern/result2.jpg", result)
