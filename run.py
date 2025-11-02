from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import cv2
import numpy as np
import os
import random
from pydantic import BaseModel
from typing import List, Dict
import onnxruntime as ort
from datetime import datetime
import time
import traceback
from pathlib import Path
from ultralytics import YOLO
import google.generativeai as genai
from PIL import Image as PILImage
import os
import re

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Optional: configure model
gemini_model = genai.GenerativeModel('gemini-2.5-flash')  # or 'gemini-1.5-pro

# ----------------------------
# Utility Functions (YOLO helpers for ONNX)
# ----------------------------
def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

def nms(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []
    keep_indices = []
    sorted_indices = np.argsort(scores)[::-1]
    for i in sorted_indices:
        should_keep = True
        for j in keep_indices:
            if calculate_iou(boxes[i], boxes[j]) > iou_threshold:
                should_keep = False
                break
        if should_keep:
            keep_indices.append(i)
    return keep_indices

def multiclass_nms(boxes, scores, class_ids, iou_threshold):
    indices = []
    for class_id in np.unique(class_ids):
        class_mask = class_ids == class_id
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]
        class_indices = nms(class_boxes, class_scores, iou_threshold)
        indices.extend(np.where(class_mask)[0][class_indices])
    return np.array(indices, dtype=int)

# ----------------------------
# Custom YOLOv8 ONNX Class (for /predict only)
# ----------------------------
class YOLOv8ONNX:
    def __init__(self, path, conf_thres=0.7, iou_thres=0.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       ):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.initialize_model(path)

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path):
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            path,
            sess_options,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.get_input_details()
        self.get_output_details()

    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)
        outputs = self.inference(input_tensor)
        self.boxes, self.scores, self.class_ids = self.process_output(outputs)
        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def inference(self, input_tensor):
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = self.extract_boxes(predictions)
        indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)
        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        boxes = predictions[:, :4]
        boxes = self.rescale_boxes(boxes)
        boxes = xywh2xyxy(boxes)
        return boxes

    def rescale_boxes(self, boxes):
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

# ----------------------------
# FastAPI App Setup
# ----------------------------
app = FastAPI()

# Paths
ONNX_MODEL_PATH = "/home/ashu/langchain_sql/biscuit_07_05_2025_sim.onnx"
PT_MODEL_PATH = "/home/ashu/langchain_sql/13_may_biscuit.pt"
CLASSES_FILE_PATH = "/home/ashu/langchain_sql/classes.txt"
OUTPUT_DIR = "output_images"
TEMP_DIR = "temp_videos"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Load Class Names
# ----------------------------
def load_class_names(file_path: str) -> List[str]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Class names file not found: {file_path}")
    with open(file_path, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]
    if not class_names:
        raise ValueError("Class names file is empty.")
    return class_names

# ONNX uses external class file
ONNX_CLASS_NAMES = load_class_names(CLASSES_FILE_PATH)
print(f"[INFO] Loaded {len(ONNX_CLASS_NAMES)} classes for ONNX model")

# PT model uses its own internal names
yolo_pt_model = YOLO(PT_MODEL_PATH)
PT_CLASS_NAMES = yolo_pt_model.names  # dict like {0: 'classA', 1: 'classB', ...}
print(f"[INFO] Loaded {len(PT_CLASS_NAMES)} classes from .pt model")

# Use ONNX model for static images
yolo_onnx_model = YOLOv8ONNX(ONNX_MODEL_PATH, conf_thres=0.25, iou_thres=0.3)

# ----------------------------
# Helper: Check if product is Britannia (case-insensitive prefix)
# ----------------------------
def is_britannia(label: str) -> bool:
    return label.lower().startswith("britannia")

# ----------------------------
# Pydantic Models
# ----------------------------
class RowsProduct(BaseModel):
    rows_no: int
    britannia_share_percent: float
    total_products_in_row: int

class PredictionResponse(BaseModel):
    rows: List[RowsProduct]
    overall_britannia_share_percent: float
    output_image_path: str

# ----------------------------
# Row Grouping Logic (for /predict only)
# ----------------------------
def merge_overlapping_boxes(rows):
    merged_rows = []
    while rows:
        current_row = rows.pop(0)
        if isinstance(current_row['elements'], set):
            current_row['elements'] = list(current_row['elements'])
        i = 0
        while i < len(rows):
            other_row = rows[i]
            if not (current_row['x_max'] < other_row['x_min'] or current_row['x_min'] > other_row['x_max'] or
                    current_row['y_max'] < other_row['y_min'] or current_row['y_min'] > other_row['y_max']):
                current_row['x_min'] = min(current_row['x_min'], other_row['x_min'])
                current_row['y_min'] = min(current_row['y_min'], other_row['y_min'])
                current_row['x_max'] = max(current_row['x_max'], other_row['x_max'])
                current_row['y_max'] = max(current_row['y_max'], other_row['y_max'])
                if isinstance(other_row['elements'], set):
                    other_row['elements'] = list(other_row['elements'])
                for element in other_row['elements']:
                    if element not in current_row['elements']:
                        current_row['elements'].append(element)
                rows.pop(i)
            else:
                i += 1
        merged_rows.append(current_row)
    return merged_rows

def merge_adjacent_rows(rows):
    merged_rows = []
    while rows:
        current_row = rows.pop(0)
        i = 0
        while i < len(rows):
            other_row = rows[i]
            height_gap = min(abs(current_row['y_max'] - other_row['y_min']), abs(current_row['y_min'] - other_row['y_max']))
            if height_gap < 20 and any(element in current_row['elements'] for element in other_row['elements']):
                current_row['x_min'] = min(current_row['x_min'], other_row['x_min'])
                current_row['y_min'] = min(current_row['y_min'], other_row['y_min'])
                current_row['x_max'] = max(current_row['x_max'], other_row['x_max'])
                current_row['y_max'] = max(current_row['y_max'], other_row['y_max'])
                if isinstance(other_row['elements'], set):
                    other_row['elements'] = list(other_row['elements'])
                for element in other_row['elements']:
                    if element not in current_row['elements']:
                        current_row['elements'].append(element)
                rows.pop(i)
            else:
                i += 1
        merged_rows.append(current_row)
    return merged_rows

def group_ids_by_row(json_data, img):
    rows = []
    for element, boxes in json_data.items():
        for box in boxes:
            x_min, y_min, x_max, y_max = box[0][0], box[0][1], box[1][0], box[1][1]
            row_found = False
            for row in rows:
                if not (y_max < row['y_min'] or y_min > row['y_max']):
                    row['y_min'] = min(row['y_min'], y_min)
                    row['y_max'] = max(row['y_max'], y_max)
                    if 'elements' not in row:
                        row['elements'] = set()
                    row['elements'].add(element)
                    row['x_min'] = min(row.get('x_min', x_min), x_min)
                    row['x_max'] = max(row.get('x_max', x_max), x_max)
                    row_found = True
                    break
            if not row_found:
                rows.append({'y_min': y_min, 'y_max': y_max, 'x_min': x_min, 'x_max': x_max, 'elements': {element}})
    rows = merge_overlapping_boxes(rows)
    rows = merge_adjacent_rows(rows)
    rows.sort(key=lambda row: row['y_min'])
    row_dict = {}
    for i, row in enumerate(rows):
        x_min_row, y_min_row, x_max_row, y_max_row = row['x_min'], row['y_min'], row['x_max'], row['y_max']
        box_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(img, (x_min_row, y_min_row), (x_max_row, y_max_row), box_color, 4)
        text = f"Row {i+1}"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       , 1)
        text_x, text_y = x_min_row, y_min_row - 10
        cv2.rectangle(img, (text_x, text_y - text_height - 2), (text_x + text_width, text_y), (255, 255, 255), -1)
        cv2.putText(img, text, (text_x, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        row_dict[f"Row {i+1}"] = {
            'elements': list(row['elements']) if isinstance(row['elements'], set) else row['elements'],
            'box_dimensions': [x_min_row, y_min_row, x_max_row, y_max_row]
        }
    return row_dict

# ----------------------------
# Endpoints
# ----------------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    print("\n" + "="*60)
    print(f"[REQUEST] Received file: {file.filename}")

    file_bytes = await file.read()
    np_img = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})
    img_copy = img.copy()

    boxes, scores, class_ids = yolo_onnx_model(img)

    if len(boxes) == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_image_path = os.path.join(OUTPUT_DIR, f"output_{timestamp}.jpg")
        cv2.imwrite(output_image_path, img_copy)
        return PredictionResponse(rows=[], overall_britannia_share_percent=0.0, output_image_path=output_image_path)

    boxes = boxes.astype(int)
    json_data: Dict[str, List[List[List[int]]]] = {}
    britannia_boxes = []

    for box, cls_id in zip(boxes, class_ids):
        cls_id = int(cls_id)
        if cls_id >= len(ONNX_CLASS_NAMES):
            continue
        cls_name = ONNX_CLASS_NAMES[cls_id]
        x1, y1, x2, y2 = box.tolist()
        if cls_name not in json_data:
            json_data[cls_name] = []
        json_data[cls_name].append([[x1, y1], [x2, y2]])
        if is_britannia(cls_name):
            britannia_boxes.append((x1, y1, x2, y2))

    overlay = img_copy.copy()
    mask_color = (0, 0, 255)
    alpha = 0.4
    for (x1, y1, x2, y2) in britannia_boxes:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), mask_color, -1)
    cv2.addWeighted(overlay, alpha, img_copy, 1 - alpha, 0, img_copy)

    for cls_name, boxes_list in json_data.items():
        color = (0, 255, 0) if is_britannia(cls_name) else (255, 0, 0)
        for box in boxes_list:
            x1, y1 = box[0]
            x2, y2 = box[1]
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_copy, cls_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    try:
        row_dict = group_ids_by_row(json_data, img_copy)
    except Exception as e:
        print(f"[ERROR] Row grouping failed: {e}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_image_path = os.path.join(OUTPUT_DIR, f"output_{timestamp}.jpg")
        cv2.imwrite(output_image_path, img_copy)
        return PredictionResponse(rows=[], overall_britannia_share_percent=0.0, output_image_path=output_image_path)

    if not row_dict:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_image_path = os.path.join(OUTPUT_DIR, f"output_{timestamp}.jpg")
        cv2.imwrite(output_image_path, img_copy)
        return PredictionResponse(rows=[], overall_britannia_share_percent=0.0, output_image_path=output_image_path)

    rows_result = []
    total_britannia_area = 0.0
    total_all_row_area = 0.0

    for i, (row_id, info) in enumerate(row_dict.items()):
        x_min_r, y_min_r, x_max_r, y_max_r = info['box_dimensions']
        row_width = x_max_r - x_min_r
        row_height = y_max_r - y_min_r
        total_row_area = row_width * row_height

        britannia_area = 0.0
        row_product_count = 0

        for cls_name, boxes_list in json_data.items():
            for box in boxes_list:
                x1, y1 = box[0]
                x2, y2 = box[1]
                area = (x2 - x1) * (y2 - y1)
                y_center = (y1 + y2) / 2
                if y_min_r <= y_center <= y_max_r:
                    row_product_count += 1
                    if is_britannia(cls_name):
                        britannia_area += area

        share_percent = (britannia_area / total_row_area) * 100 if total_row_area > 0 else 0.0
        share_percent = round(share_percent, 2)

        rows_result.append(RowsProduct(
            rows_no=i + 1,
            britannia_share_percent=share_percent,
            total_products_in_row=row_product_count
        ))

        total_britannia_area += britannia_area
        total_all_row_area += total_row_area

    overall_share = (total_britannia_area / total_all_row_area) * 100 if total_all_row_area > 0 else 0.0
    overall_share = round(overall_share, 2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_image_path = os.path.join(OUTPUT_DIR, f"output_{timestamp}.jpg")
    cv2.imwrite(output_image_path, img_copy)

    print(f"[RESULT] Rows: {len(rows_result)}, Overall Britannia share: {overall_share:.2f}%")
    print(f"[RESULT] Output image saved to: {output_image_path}")
    print("="*60 + "\n")

    return PredictionResponse(
        rows=rows_result,
        overall_britannia_share_percent=overall_share,
        output_image_path=output_image_path
    )

@app.post("/predict_video")
async def predict_video(file: UploadFile = File(...)):
    start_time = time.time()
    print("\n" + "="*70)
    print(f"[VIDEO REQUEST] Received video: {file.filename}")
    print("="*70)

    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        raise HTTPException(status_code=400, detail="Invalid video format. Supported: mp4, avi, mov, mkv, webm")

    timestamp = int(time.time())
    temp_input = os.path.join(TEMP_DIR, f"input_{timestamp}_{file.filename}")
    temp_output = os.path.join(TEMP_DIR, f"output_{timestamp}_{file.filename}")

    # ✅ Track: (class_name, track_id) → frame count
    track_appearance_count = {}

    try:
        with open(temp_input, "wb") as f:
            f.write(await file.read())

        cap = cv2.VideoCapture(temp_input)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Could not open input video")

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

        frame_count = 0
        processed_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ✅ Only process every 5th frame (0, 5, 10, ...)
            if frame_count % 5 != 0:
                # Write original frame (no detection)
                out.write(frame)
                frame_count += 1
                continue

            processed_frames += 1

            # ✅ Run inference on this frame
            results = yolo_pt_model.track(
                frame,
                persist=True,
                conf=0.7,
                iou=0.3,
                verbose=False
            )

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                confs = results[0].boxes.conf.cpu().numpy()

                for box, cls_id, track_id, conf in zip(boxes, class_ids, track_ids, confs):
                    x1, y1, x2, y2 = map(int, box)
                    if cls_id in PT_CLASS_NAMES:
                        class_name = PT_CLASS_NAMES[cls_id]
                    else:
                        class_name = "unknown"

                    # ✅ Count appearance per (class, track_id)
                    key = (class_name, int(track_id))
                    track_appearance_count[key] = track_appearance_count.get(key, 0) + 1

                    color = (0, 255, 0) if is_britannia(class_name) else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{class_name} ID:{track_id}", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()

        print(f"\n[SUMMARY] Total frames: {frame_count}, Processed frames: {processed_frames}")

        # ✅ Filter: Only count tracks seen in at least MIN_FRAMES (in processed frames)
        MIN_FRAMES = 2  # Since we skip 4/5 frames, 2 processed frames ≈ 10 real frames

        total_products = 0
        britannia_count = 0

        for (cls_name, track_id), count in track_appearance_count.items():
            if count >= MIN_FRAMES:
                total_products += 1
                if is_britannia(cls_name):
                    britannia_count += 1

        britannia_share = round((britannia_count / total_products * 100) if total_products > 0 else 0.0, 2)

        print("\n" + "="*70)
        print(f"[FINAL RESULT]")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Frames actually processed (every 5th): {processed_frames}")
        print(f"  Unique tracked products (min {MIN_FRAMES} processed frames): {total_products}")
        print(f"  Britannia products: {britannia_count}")
        print(f"  Britannia share: {britannia_share}%")
        print("="*70)

        return JSONResponse(content={
            "total_unique_products": total_products,
            "britannia_products": britannia_count,
            "britannia_share_percent": britannia_share,
            "annotated_video_url": f"/video/{Path(temp_output).name}",
            "processing_time_seconds": round(time.time() - start_time, 2)
        })

    except Exception as e:
        print(f"[EXCEPTION] {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")
    finally:
        if os.path.exists(temp_input):
            try:
                os.remove(temp_input)
            except:
                pass
# ----------------------------
# NMS Helper
# ----------------------------
def non_max_suppression_fast(boxes, iou_threshold=0.5):
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(np.float32)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    indices = np.arange(len(boxes))
    keep = []
    while len(indices) > 0:
        i = indices[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[indices[1:]])
        yy1 = np.maximum(y1[i], y1[indices[1:]])
        xx2 = np.minimum(x2[i], x2[indices[1:]])
        yy2 = np.minimum(y2[i], y2[indices[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / (areas[i] + areas[indices[1:]] - w * h + 1e-6)
        to_remove = np.where(overlap > iou_threshold)[0] + 1
        indices = np.delete(indices, to_remove)
        indices = np.delete(indices, 0)
    return keep

@app.get("/video/{filename}")
async def get_video(filename: str):
    file_path = os.path.join(TEMP_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="video/mp4", filename=filename)
    return JSONResponse(status_code=404, content={"error": "Video not found"})

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image, ImageDraw
import io
import os
import uuid
import torch
from ultralytics import YOLO
from collections import defaultdict

# Configuration
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load models
all_product_model = YOLO("/home/ashu/langchain_sql/best.pt")
biscuit_model = YOLO("/home/ashu/langchain_sql/13_may_biscuit.pt")
IOU_THRESHOLD = 0.5

# --- Helper: IoU ---
def box_iou(box1, box2):
    box1 = box1.unsqueeze(1)
    box2 = box2.unsqueeze(0)
    x1 = torch.max(box1[..., 0], box2[..., 0])
    y1 = torch.max(box1[..., 1], box2[..., 1])
    x2 = torch.min(box1[..., 2], box2[..., 2])
    y2 = torch.min(box1[..., 3], box2[..., 3])
    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)

# --- Helper: Draw boxes ---
def draw_boxes(image: Image.Image, boxes, labels):
    draw = ImageDraw.Draw(image)
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box)
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        draw.rectangle([x1, y1 - 20, x1 + len(label) * 8 + 10, y1], fill="green")
        draw.text((x1 + 2, y1 - 18), label, fill="white")
    return image

# --- Endpoint 1: Detect & Save Image ---


@app.post("/detect")
async def detect_products(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        input_image = PILImage.open(io.BytesIO(contents)).convert("RGB")

        # Run inference
        all_results = all_product_model(input_image, verbose=False)
        biscuit_results = biscuit_model(input_image, verbose=False)

        # Get all boxes
        all_boxes = torch.empty((0, 4))
        for r in all_results:
            if r.boxes is not None and len(r.boxes) > 0:
                all_boxes = torch.cat([all_boxes, r.boxes.xyxy.cpu()], dim=0)

        # Get biscuit data
        biscuit_boxes = torch.empty((0, 4))
        biscuit_classes = []
        for r in biscuit_results:
            if r.boxes is not None and len(r.boxes) > 0:
                biscuit_boxes = torch.cat([biscuit_boxes, r.boxes.xyxy.cpu()], dim=0)
                biscuit_classes.extend([biscuit_model.names[int(cls)] for cls in r.boxes.cls.cpu()])

        # Classify
        product_counts = defaultdict(int)
        final_labels = []

        if all_boxes.shape[0] == 0:
            total = 0
        else:
            if biscuit_boxes.shape[0] == 0:
                total = all_boxes.shape[0]
                product_counts["others_competitor_product"] = total
                final_labels = ["others_competitor_product"] * total
            else:
                iou_matrix = box_iou(all_boxes, biscuit_boxes)
                max_iou, best_match = torch.max(iou_matrix, dim=1)
                for i in range(all_boxes.shape[0]):
                    if max_iou[i] >= IOU_THRESHOLD:
                        cls_name = biscuit_classes[best_match[i].item()]
                        product_counts[cls_name] += 1
                        final_labels.append(cls_name)
                    else:
                        product_counts["others_competitor_product"] += 1
                        final_labels.append("others_competitor_product")
                total = all_boxes.shape[0]

        # Annotate original image
        annotated_img = input_image.copy()
        annotated_img = draw_boxes(annotated_img, all_boxes.tolist(), final_labels)

        # --- NEW: Extract text from "others_competitor_product" crops using Gemini ---
        competitor_texts = []

        for idx, label in enumerate(final_labels):
            if label == "others_competitor_product":
                box = all_boxes[idx].tolist()
                x1, y1, x2, y2 = map(int, box)
                cropped_img = input_image.crop((x1, y1, x2, y2))

                try:
                    # Convert to bytes
                    img_byte_arr = io.BytesIO()
                    cropped_img.save(img_byte_arr, format='JPEG')
                    img_byte_arr.seek(0)

                    # Use Gemini for OCR (text extraction only)
                    prompt = "Extract all visible text from this product packaging. Return only the raw text, nothing else."
                    response = gemini_model.generate_content([prompt, PILImage.open(img_byte_arr)])
                    extracted_text = response.text.strip()

                    competitor_texts.append({
                        "box": box,
                        "extracted_text": extracted_text
                    })

                except Exception as e:
                    competitor_texts.append({
                        "box": box,
                        "extracted_text": "",
                        "error": str(e)
                    })

        # Save annotated image
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(OUTPUT_DIR, filename)
        annotated_img.save(filepath, quality=90)

        # Return response — labels unchanged, just add extracted text
        return JSONResponse({
            "total_products": total,
            "products": dict(product_counts),
            "filename": filename,
            "competitor_ocr_texts": competitor_texts  # <-- ONLY the extracted text
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# --- Endpoint 2: Serve Saved Image ---
# @app.get("/image/{filename}")
# async def get_image(filename: str):
#     filepath = os.path.join(OUTPUT_DIR, filename)
#     if not os.path.isfile(filepath):
#         raise HTTPException(status_code=404, detail="Image not found")

#     def iterfile():
#         with open(filepath, mode="rb") as f:
#             yield f.read()

#     return StreamingResponse(iterfile(), media_type="image/jpeg")