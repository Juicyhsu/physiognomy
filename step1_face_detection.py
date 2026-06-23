"""
================================================
面相辨識專題 — Step 1: 人臉偵測 & 關鍵點提取
================================================
使用 MediaPipe Face Mesh 偵測 468 個臉部關鍵點
並計算面相所需的五官幾何特徵值

執行方式:
    pip install -r requirements.txt
    python step1_face_detection.py

輸出:
    - annotated_face.jpg (標記關鍵點的人臉圖)
    - features.json     (特徵值 JSON)
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import math
from pathlib import Path

# ─────────────────────────────────────────
# MediaPipe 關鍵點索引 (468點中挑重要的)
# 參考: https://mediapipe.dev/solutions/face_mesh
# ─────────────────────────────────────────
LANDMARKS = {
    # 眼睛
    "left_eye_inner":   133,
    "left_eye_outer":   33,
    "left_eye_top":     159,
    "left_eye_bottom":  145,
    "right_eye_inner":  362,
    "right_eye_outer":  263,
    "right_eye_top":    386,
    "right_eye_bottom": 374,

    # 眉毛
    "left_brow_inner":  55,
    "left_brow_outer":  46,
    "right_brow_inner": 285,
    "right_brow_outer": 276,
    "left_brow_top":    52,
    "right_brow_top":   282,

    # 鼻子
    "nose_tip":         1,
    "nose_left":        129,
    "nose_right":       358,
    "nose_bridge_top":  6,
    "nose_bottom":      2,

    # 嘴巴
    "mouth_left":       61,
    "mouth_right":      291,
    "mouth_top":        13,
    "mouth_bottom":     14,
    "upper_lip_top":    0,
    "lower_lip_bottom": 17,

    # 臉部輪廓
    "face_top":         10,
    "face_bottom":      152,
    "face_left":        234,
    "face_right":       454,
    "chin":             175,
    "left_cheek":       116,
    "right_cheek":      345,

    # 耳朵
    "left_ear":         93,
    "right_ear":        323,

    # 虹膜中心
    "left_iris_center": 468,
    "right_iris_center": 473,
}

# ─────────────────────────────────────────
# 工具函式
# ─────────────────────────────────────────
def dist(p1, p2):
    """計算兩點歐氏距離"""
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def safe_div(num, denom, fallback=0.0):
    """安全除法，防止 float division by zero"""
    return float(num) / float(denom) if denom != 0 else fallback

def get_landmark_coords(landmarks, idx, img_w, img_h):
    """把 MediaPipe 歸一化座標轉成像素座標"""
    lm = landmarks[idx]
    return (int(lm.x * img_w), int(lm.y * img_h))

def get_all_coords(landmarks, img_w, img_h):
    """取得所有重要關鍵點的像素座標"""
    return {
        name: get_landmark_coords(landmarks, idx, img_w, img_h)
        for name, idx in LANDMARKS.items()
    }

# ─────────────────────────────────────────
# 特徵提取函式
# ─────────────────────────────────────────
def extract_features(coords):
    """
    從關鍵點計算面相分析所需的幾何特徵值

    回傳 dict，包含:
    - 三庭比例 (臉的上中下三段)
    - 五眼比例 (臉寬 vs 眼距)
    - 各部位寬高比
    """
    features = {}

    # ── 臉部整體尺寸 ──
    face_height = dist(coords["face_top"], coords["face_bottom"])
    face_width  = dist(coords["face_left"], coords["face_right"])
    features["face_height"] = face_height
    features["face_width"]  = face_width
    features["face_ratio"]  = round(safe_div(face_width, face_height, 0.75), 3)  # 越接近1越圓

    # ── 三庭 (臉的上中下三段比例) ──
    # 上庭: 髮際線到眉毛; 中庭: 眉毛到鼻底; 下庭: 鼻底到下巴
    upper_zone  = dist(coords["face_top"],    coords["left_brow_top"])
    middle_zone = dist(coords["left_brow_top"], coords["nose_bottom"])
    lower_zone  = dist(coords["nose_bottom"], coords["face_bottom"])
    total_zone  = upper_zone + middle_zone + lower_zone

    features["upper_zone_ratio"]  = round(safe_div(upper_zone, total_zone, 0.333), 3)
    features["middle_zone_ratio"] = round(safe_div(middle_zone, total_zone, 0.333), 3)
    features["lower_zone_ratio"]  = round(safe_div(lower_zone, total_zone, 0.333), 3)

    # 理想三庭比例接近 1:1:1 (各約 0.333)
    features["san_ting_balance"] = round(
        1 - max(
            abs(features["upper_zone_ratio"]  - 0.333),
            abs(features["middle_zone_ratio"] - 0.333),
            abs(features["lower_zone_ratio"]  - 0.333)
        ) * 3, 3
    )  # 1.0 = 完美三庭均等

    # ── 眼睛特徵 ──
    left_eye_w  = dist(coords["left_eye_inner"],  coords["left_eye_outer"])
    left_eye_h  = dist(coords["left_eye_top"],    coords["left_eye_bottom"])
    right_eye_w = dist(coords["right_eye_inner"], coords["right_eye_outer"])
    right_eye_h = dist(coords["right_eye_top"],   coords["right_eye_bottom"])

    features["left_eye_ratio"]  = round(safe_div(left_eye_h, left_eye_w, 0.28), 3)  # 開眼度
    features["right_eye_ratio"] = round(safe_div(right_eye_h, right_eye_w, 0.28), 3)
    features["eye_width_avg"]   = round((left_eye_w + right_eye_w) / 2, 1)

    # 眼距 (兩眼內角距離) vs 眼寬 → 五眼中間那「眼」
    eye_gap = dist(coords["left_eye_inner"], coords["right_eye_inner"])
    features["eye_gap_ratio"] = round(safe_div(eye_gap, features["eye_width_avg"], 1.0), 3)
    # 理想值約 1.0 (兩眼相距一個眼寬)

    # ── 眉毛特徵 ──
    left_brow_w  = dist(coords["left_brow_inner"],  coords["left_brow_outer"])
    right_brow_w = dist(coords["right_brow_inner"], coords["right_brow_outer"])
    features["brow_eye_ratio"] = round(
        safe_div((left_brow_w + right_brow_w) / 2, features["eye_width_avg"], 1.0), 3
    )  # 眉長 vs 眼長，>1.0 代表眉毛較長

    # 眉眼距離
    brow_eye_dist = dist(coords["left_brow_top"], coords["left_eye_top"])
    features["brow_eye_distance"] = round(safe_div(brow_eye_dist, face_height, 0.08), 3)

    # ── 鼻子特徵 ──
    nose_w = dist(coords["nose_left"], coords["nose_right"])
    nose_h = dist(coords["nose_bridge_top"], coords["nose_tip"])
    features["nose_width_ratio"]  = round(safe_div(nose_w, face_width, 0.3), 3)  # 鼻寬 vs 臉寬
    features["nose_height_ratio"] = round(safe_div(nose_h, face_height, 0.3), 3)
    features["nose_shape_ratio"]  = round(safe_div(nose_w, nose_h, 0.6), 3)  # 越大越寬扁，越小越長挺

    # 鼻孔外露度 (nostril_exposure_ratio)
    nose_wings_y = (coords["nose_left"][1] + coords["nose_right"][1]) / 2.0
    features["nostril_exposure_ratio"] = round(safe_div(nose_wings_y - coords["nose_tip"][1], face_height, 0.0), 3)

    # ── 嘴巴特徵 ──
    mouth_w = dist(coords["mouth_left"],  coords["mouth_right"])
    mouth_h = dist(coords["mouth_top"],   coords["mouth_bottom"])
    features["mouth_width_ratio"]  = round(safe_div(mouth_w, face_width, 0.38), 3)
    features["mouth_shape_ratio"]  = round(safe_div(mouth_w, mouth_h, 3.0), 3)  # 嘴形寬度比
    features["lip_thickness"]      = round(safe_div(mouth_h, face_height, 0.05), 3)

    # 嘴角上揚度 (mouth_corner_upward_ratio)
    mouth_center_y = (coords["mouth_top"][1] + coords["mouth_bottom"][1]) / 2.0
    mouth_corners_y = (coords["mouth_left"][1] + coords["mouth_right"][1]) / 2.0
    features["mouth_corner_upward_ratio"] = round(safe_div(mouth_center_y - mouth_corners_y, max(1.0, mouth_w), 0.0), 3)

    # ── 下巴特徵 ──
    jaw_w = dist(coords["left_cheek"], coords["right_cheek"])
    features["jaw_ratio"] = round(safe_div(jaw_w, face_width, 0.65), 3)  # 下巴相對臉寬

    # ── 三白眼指數 (iris_vertical_position) ──
    if "left_iris_center" in coords and "right_iris_center" in coords:
        left_eye_height = max(1.0, coords["left_eye_bottom"][1] - coords["left_eye_top"][1])
        right_eye_height = max(1.0, coords["right_eye_bottom"][1] - coords["right_eye_top"][1])
        left_pos = safe_div(coords["left_eye_bottom"][1] - coords["left_iris_center"][1], left_eye_height, 0.5)
        right_pos = safe_div(coords["right_eye_bottom"][1] - coords["right_iris_center"][1], right_eye_height, 0.5)
        features["iris_vertical_position"] = round((left_pos + right_pos) / 2.0, 3)
    else:
        features["iris_vertical_position"] = 0.50

    # ── 新增面相學幾何特徵點分析 ──
    # 1. 印堂寬度比 (glabella_width_ratio): 兩眉心間距 / 臉寬
    glabella_w = dist(coords["left_brow_inner"], coords["right_brow_inner"])
    features["glabella_width_ratio"] = round(safe_div(glabella_w, face_width, 0.15), 3)

    # 2. 眉壓眼垂直距離比 (brow_eye_distance): 眉毛到眼睛上緣 / 臉高
    brow_eye_dist = dist(coords["left_brow_top"], coords["left_eye_top"])
    features["brow_eye_distance"] = round(safe_div(brow_eye_dist, face_height, 0.08), 3)

    # 3. 人中長度比 (philtrum_ratio): 鼻底到上唇中央 / 臉高
    philtrum_len = dist(coords["nose_bottom"], coords["upper_lip_top"])
    features["philtrum_ratio"] = round(safe_div(philtrum_len, face_height, 0.035), 3)

    return features

# ─────────────────────────────────────────
# 視覺化繪製
# ─────────────────────────────────────────
def draw_annotations(image, coords, features):
    """在圖像上繪製五官標記與特徵值"""
    img = image.copy()
    h, w = img.shape[:2]

    colors = {
        "eye":   (255, 180, 50),
        "brow":  (100, 200, 255),
        "nose":  (50,  255, 150),
        "mouth": (255, 100, 150),
        "face":  (200, 200, 200),
    }

    def draw_point(name, color, r=4):
        if name in coords:
            cv2.circle(img, coords[name], r, color, -1)

    def draw_line(p1_name, p2_name, color, thickness=1):
        if p1_name in coords and p2_name in coords:
            cv2.line(img, coords[p1_name], coords[p2_name], color, thickness)

    # 眼睛
    for side in ["left", "right"]:
        for p in ["inner", "outer", "top", "bottom"]:
            draw_point(f"{side}_eye_{p}", colors["eye"])
        draw_line(f"{side}_eye_inner", f"{side}_eye_outer", colors["eye"])
        draw_line(f"{side}_eye_top",   f"{side}_eye_bottom", colors["eye"])

    # 眉毛
    for side in ["left", "right"]:
        draw_point(f"{side}_brow_inner", colors["brow"])
        draw_point(f"{side}_brow_outer", colors["brow"])
        draw_line(f"{side}_brow_inner", f"{side}_brow_outer", colors["brow"], 2)

    # 鼻子
    for p in ["tip", "left", "right", "bridge_top", "bottom"]:
        draw_point(f"nose_{p}", colors["nose"])
    draw_line("nose_left",  "nose_right",    colors["nose"])
    draw_line("nose_bridge_top", "nose_tip", colors["nose"])

    # 嘴巴
    for p in ["left", "right", "top", "bottom"]:
        draw_point(f"mouth_{p}", colors["mouth"])
    draw_line("mouth_left",  "mouth_right", colors["mouth"])
    draw_line("mouth_top",   "mouth_bottom", colors["mouth"])

    # 臉框
    for p in ["top", "bottom", "left", "right"]:
        draw_point(f"face_{p}", colors["face"])

    # 三庭線 (水平虛線)
    for pt_name, label in [
        ("left_brow_top", "上庭/中庭"),
        ("nose_bottom",   "中庭/下庭"),
    ]:
        if pt_name in coords:
            y = coords[pt_name][1]
            cv2.line(img, (0, y), (w, y), (255, 255, 100), 1, cv2.LINE_AA)
            cv2.putText(img, label, (5, y-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,100), 1)

    # 右側資訊欄
    info_lines = [
        f"臉寬/臉高: {features['face_ratio']:.2f}",
        f"三庭均衡: {features['san_ting_balance']:.2f}",
        f"眼神開度(左): {features['left_eye_ratio']:.2f}",
        f"眼距比: {features['eye_gap_ratio']:.2f}",
        f"眉長比: {features['brow_eye_ratio']:.2f}",
        f"鼻寬比: {features['nose_width_ratio']:.2f}",
        f"嘴寬比: {features['mouth_width_ratio']:.2f}",
    ]
    panel_x = w - 180
    cv2.rectangle(img, (panel_x - 5, 10), (w - 5, 15 + len(info_lines)*18),
                  (30, 30, 30), -1)
    for i, line in enumerate(info_lines):
        cv2.putText(img, line, (panel_x, 24 + i*18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

    return img

# ─────────────────────────────────────────
# 主函式
# ─────────────────────────────────────────
def analyze_face(image_path: str, output_dir: str = "."):
    """
    完整分析一張人臉圖片

    Args:
        image_path: 輸入圖片路徑
        output_dir: 輸出目錄

    Returns:
        features (dict): 特徵值
        coords   (dict): 關鍵點座標
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # 讀圖
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"找不到圖片: {image_path}")

    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # MediaPipe 偵測
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        print("❌ 未偵測到人臉，請確認圖片清晰且正面朝向")
        return None, None

    landmarks = results.multi_face_landmarks[0].landmark

    # 提取座標與特徵
    coords   = get_all_coords(landmarks, w, h)
    features = extract_features(coords)

    # 繪製與儲存
    annotated = draw_annotations(image, coords, features)
    out_img   = str(output_dir / "annotated_face.jpg")
    out_json  = str(output_dir / "features.json")

    cv2.imwrite(out_img, annotated)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)

    print(f"✅ 偵測完成")
    print(f"   標記圖: {out_img}")
    print(f"   特徵值: {out_json}")
    print(f"\n📊 主要特徵:")
    for k, v in features.items():
        print(f"   {k}: {v}")

    return features, coords


# ─────────────────────────────────────────
# 測試入口 (改成你的圖片路徑)
# ─────────────────────────────────────────
if __name__ == "__main__":
    # ⚠️ 把下面路徑換成你的測試圖片
    TEST_IMAGE = "test_face.jpg"

    # 若沒有圖，先用攝影機拍一張
    if not Path(TEST_IMAGE).exists():
        print("找不到測試圖，用攝影機拍照...")
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            cv2.imwrite(TEST_IMAGE, frame)
            print(f"已存成 {TEST_IMAGE}")
        else:
            print("攝影機無法開啟，請提供圖片")
            exit(1)

    analyze_face(TEST_IMAGE, output_dir="output")
