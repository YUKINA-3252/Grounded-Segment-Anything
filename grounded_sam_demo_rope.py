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


if __name__ == "__main__":
    directory_path = "ros/rope/image"
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
    # parser.add_argument("--text_prompt", type=str, required=False, help="text prompt", default="Ribbon over the box")
    parser.add_argument("--text_prompt", type=str, required=False, help="text prompt", default="rope. plaid pattern box")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="ros/rope/outputs", required=False, help="output directory"
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

    boxes_filt_np = boxes_filt.cpu().numpy()
    # box mask
    box_indices = [index for index, item in enumerate(pred_phrases) if 'box' in item]
    # box_center_point = np.asarray([np.abs(boxes_filt_np[box_indices[0]][2] - boxes_filt_np[box_indices[0]][0]) / 2.0, np.abs(boxes_filt_np[box_indices[0]][3] - boxes_filt_np[box_indices[0]][1]) / 2.0])
    box_area = abs(boxes_filt_np[box_indices[0]][2] - boxes_filt_np[box_indices[0]][0]) * np.abs(boxes_filt_np[box_indices[0]][3] - boxes_filt_np[box_indices[0]][1])
    mask_image_box = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    mask_np_box = masks[box_indices[0]].cpu().numpy()[0]
    mask_image_box[mask_np_box] = 255
    x1 = int(min(boxes_filt_np[box_indices[0]][0], boxes_filt_np[box_indices[0]][2]))
    x2 = int(max(boxes_filt_np[box_indices[0]][0], boxes_filt_np[box_indices[0]][2]))
    y1 = int(min(boxes_filt_np[box_indices[0]][1], boxes_filt_np[box_indices[0]][3]))
    y2 = int(max(boxes_filt_np[box_indices[0]][1], boxes_filt_np[box_indices[0]][3]))
    cv2.imwrite(os.path.join(output_dir, 'box_mask.png'), mask_image_box)
    # ribbon mask
    ribbon_indices = [index for index, item in enumerate(pred_phrases) if 'ribbon' in item]
    ribbon_masks = []
    for i in(ribbon_indices):
        ribbon_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        ribbon_mask_np = masks[i].cpu().numpy()[0]
        box_bbox = np.zeros_like(ribbon_mask_np, dtype=bool)
        box_bbox[y1:y2+1, x1:x2+1] = True
        ribbon_mask[ribbon_mask_np & box_bbox] = 255
        white_mask = (mask_image_box == 255) & (ribbon_mask == 255)
        fix_ribbon_mask = ribbon_mask.copy()
        fix_ribbon_mask[white_mask] = 0
        ribbon_masks.append(fix_ribbon_mask)

    # true_counts = [np.sum(ribbon_mask) for ribbon_mask in ribbon_masks]
    # max_index = np.argmax(true_counts)
    # cv2.imwrite(os.path.join(output_dir, 'ribbon_mask.png'), ribbon_masks[max_index])
    sum_ribbon_mask = np.maximum.reduce(ribbon_masks)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    sum_ribbon_mask = cv2.morphologyEx(sum_ribbon_mask, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(os.path.join(output_dir, 'ribbon_mask.png'), sum_ribbon_mask)

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

    image = cv2.imread(os.path.join(output_dir, "ribbon_mask.png"))
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

    cv2.imwrite(os.path.join(output_dir, 'Regions.png'), image)

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
    # distance = math.sqrt((standard_box_centroid[0] - knot_point_base_coords[0][0]) ** 2 + (standard_box_centroid[1] - knot_point_base_coords[0][1]) ** 2 + (standard_box_centroid[2] - knot_point_base_coords[0][2]) ** 2)
    distance = math.sqrt((standard_box_centroid[0] - knot_point_base_coords[0][0]) ** 2 + (standard_box_centroid[1] - knot_point_base_coords[0][1]) ** 2)

    # compare standard box and arrow direction
    arrow_point_3d = []
    for i in range(4):
        start_3d_arrow_point = np.array(coords_to_depth(depth_image, np.float32(arrow_point[i][0]), np.float32(arrow_point[i][1])))
        end_3d_arrow_point = np.array(coords_to_depth(depth_image, np.float32(arrow_point[i][2]), np.float32(arrow_point[i][3])))
        arrow_point_3d.append([start_3d_arrow_point, end_3d_arrow_point])
    arrow_directions = [pair[1] - pair[0] for pair in arrow_point_3d]
    normalized_arrow_directions = [v / np.linalg.norm(v) for v in arrow_directions]
    # normalize and caluculate angle with "tate" of standard box
    tate = normalize(np.array(yaml_data.get("tate")))
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
    with open(os.path.join("ros/rope", "results.yaml"), "w") as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)
    with open("ros/rope/output.txt", "w") as file:
        json.dump(data, file)
