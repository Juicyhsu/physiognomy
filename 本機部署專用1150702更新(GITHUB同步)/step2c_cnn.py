"""
================================================
面相引擎 路線 B 升級版 — CNN 深度學習分類器
================================================
架構：
  MediaPipe → 裁切五官 ROI → CNN 分類 → 面相解讀

為什麼用 CNN 不用 RandomForest：
  RF  輸入：人工計算的幾何數值（17個數字）
  CNN 輸入：五官的實際圖片像素，自己學視覺特徵

這才是深度學習課程要求的核心！

執行：
  python step2c_cnn.py --build   # 建立ROI資料集
  python step2c_cnn.py --train   # 訓練CNN
  python step2c_cnn.py --predict your_face.jpg
"""

import argparse
import json
import warnings
from pathlib import Path
import tempfile

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── TensorFlow / Keras ──
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

# ── 自己的模組 ──
from step1_face_detection import analyze_face, get_landmark_coords, LANDMARKS
import mediapipe as mp

print(f"TensorFlow 版本: {tf.__version__}")
print(f"GPU 可用: {len(tf.config.list_physical_devices('GPU')) > 0}")

# ── Keras 3 載入舊版 Functional 模型所需的自訂運算層 ──
class TrueDivide(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
    def __call__(self, *args, **kwargs):
        new_args = []
        for arg in args:
            if not tf.is_tensor(arg) and not hasattr(arg, '_keras_history') and not isinstance(arg, tf.keras.KerasTensor):
                new_args.append(tf.constant(arg, dtype=tf.float32))
            else:
                new_args.append(arg)
        return super().__call__(*new_args, **kwargs)
    def call(self, x, y):
        return x / y

class CustomSubtract(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
    def __call__(self, *args, **kwargs):
        new_args = []
        for arg in args:
            if not tf.is_tensor(arg) and not hasattr(arg, '_keras_history') and not isinstance(arg, tf.keras.KerasTensor):
                new_args.append(tf.constant(arg, dtype=tf.float32))
            else:
                new_args.append(arg)
        return super().__call__(*new_args, **kwargs)
    def call(self, x, y):
        return x - y



# ═══════════════════════════════════════
# 常數設定
# ═══════════════════════════════════════

ROI_SIZE    = 64          # CNN 輸入大小 (64x64)
BATCH_SIZE  = 32
EPOCHS      = 30
LR          = 1e-4


# ═══════════════════════════════════════
# Focal Loss（解決類別不平衡）
# ═══════════════════════════════════════

def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """
    Sigmoid Focal Loss 的 Categorical 版本（純 TF，無需 tensorflow_addons）

    為什麼用 Focal Loss：
      CelebA 標籤極度不平衡（例如圓潤臉 8,743 vs 標準臉型 83,826，比例 1:9.6）。
      一般 Cross-Entropy 讓模型傾向預測多數類。
      Focal Loss 透過 (1-p_t)^gamma 加權，迫使模型多關注難分樣本（少數類）。

    Args:
        gamma: focusing 參數，越大對簡單樣本的抑制越強（建議 1.0~2.0）
        alpha: 正類的權重係數（建議 0.25）

    Returns:
        Keras 可接受的 loss 函式
    """
    def loss_fn(y_true, y_pred):
        import tensorflow as tf
        # 防止 log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        # 每類的 cross-entropy
        ce = -y_true * tf.math.log(y_pred)
        # 每個樣本預測到正確類別的機率 p_t
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        # focal weight：難分樣本（p_t 小）得到更高權重
        focal_weight = alpha * tf.pow(1.0 - p_t, gamma)
        return tf.reduce_mean(focal_weight * ce)
    return loss_fn

# 五個分類任務的設定
TASKS = {
    "nose": {
        "desc":    "鼻型分類器",
        "classes": ["挺鼻型", "寬鼻型", "標準鼻型"],
        # MediaPipe 關鍵點圍出鼻子 ROI 的範圍（像素padding）
        "landmarks": ["nose_bridge_top", "nose_left", "nose_right",
                      "nose_tip", "nose_bottom"],
        "padding": 20,
        # CelebA 標籤 → 我們的類別
        "label_fn": lambda r: (
            "挺鼻型" if r.get("Pointy_Nose", 0) else
            "寬鼻型" if r.get("Big_Nose", 0) else
            "標準鼻型"
        ),
    },
    "eye": {
        "desc":    "眼型分類器",
        "classes": ["明亮眼型", "細眼型", "疲態眼型"],
        "landmarks": ["left_eye_inner", "left_eye_outer",
                      "left_eye_top",   "left_eye_bottom"],
        "padding": 18,
        "label_fn": lambda r: (
            "細眼型"   if r.get("Narrow_Eyes", 0) else
            "疲態眼型" if r.get("Bags_Under_Eyes", 0) else
            "明亮眼型"
        ),
    },
    "brow": {
        "desc":    "眉型分類器",
        "classes": ["上揚眉", "濃眉", "標準眉"],
        "landmarks": ["left_brow_inner", "left_brow_outer",
                      "left_brow_top",   "left_eye_top"],
        "padding": 16,
        "label_fn": lambda r: (
            "上揚眉" if r.get("Arched_Eyebrows", 0) else
            "濃眉"   if r.get("Bushy_Eyebrows", 0) else
            "標準眉"
        ),
    },
    "mouth": {
        "desc":    "嘴型分類器",
        "classes": ["豐唇型", "笑容型", "標準嘴型"],
        "landmarks": ["mouth_left", "mouth_right",
                      "upper_lip_top", "lower_lip_bottom"],
        "padding": 18,
        "label_fn": lambda r: (
            "豐唇型"  if r.get("Big_Lips", 0) else
            "笑容型"  if r.get("Smiling", 0) else
            "標準嘴型"
        ),
    },
    "face": {
        "desc":    "臉型分類器",
        "classes": ["瓜子臉", "圓潤臉", "稜角臉", "標準臉型"],
        "landmarks": ["face_top", "face_bottom", "face_left", "face_right"],
        "padding": 30,
        "label_fn": lambda r: (
            "瓜子臉"  if r.get("Oval_Face", 0) else
            "圓潤臉"  if r.get("Chubby", 0) else
            "稜角臉"  if r.get("High_Cheekbones", 0) else
            "標準臉型"
        ),
    },
}


# ═══════════════════════════════════════
# 1. 裁切五官 ROI
# ═══════════════════════════════════════

def crop_roi(image_bgr: np.ndarray, coords: dict,
             landmark_names: list, padding: int, size: int) -> np.ndarray:
    """
    根據 MediaPipe 關鍵點座標，裁切五官的局部圖片（ROI）

    步驟：
      1. 找出指定關鍵點的邊界框
      2. 加 padding 擴大一點
      3. Resize 到固定大小 (size x size)

    Args:
        image_bgr:       OpenCV 圖片 (H, W, 3)
        coords:          {關鍵點名稱: (x, y)} dict
        landmark_names:  要圍住的關鍵點名稱列表
        padding:         邊界擴展像素數
        size:            輸出圖片大小

    Returns:
        roi: (size, size, 3) numpy array，或 None（失敗時）
    """
    h, w = image_bgr.shape[:2]
    pts  = [coords[name] for name in landmark_names if name in coords]

    if not pts:
        return None

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    x1 = max(0, min(xs) - padding)
    y1 = max(0, min(ys) - padding)
    x2 = min(w, max(xs) + padding)
    y2 = min(h, max(ys) + padding)

    if x2 <= x1 or y2 <= y1:
        return None

    roi = image_bgr[y1:y2, x1:x2]
    roi = cv2.resize(roi, (size, size))
    return roi


def extract_all_rois(image_path: str) -> dict:
    """
    對一張圖片提取所有五官的 ROI

    Returns:
        {"nose": np.ndarray, "eye": ..., ...}  或 None
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    h, w = image.shape[:2]
    rgb  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    lm     = results.multi_face_landmarks[0].landmark
    coords = {
        name: get_landmark_coords(lm, idx, w, h)
        for name, idx in LANDMARKS.items()
    }

    rois = {}
    for task_name, cfg in TASKS.items():
        roi = crop_roi(
            image, coords,
            cfg["landmarks"], cfg["padding"], ROI_SIZE
        )
        if roi is not None:
            rois[task_name] = roi

    return rois if rois else None


# ═══════════════════════════════════════
# 2. 建立 ROI 資料集
# ═══════════════════════════════════════

def build_roi_dataset(
    celeba_img_dir:   str = "data/celeba/img_align_celeba/img_align_celeba",
    celeba_attr_file: str = "data/celeba/list_attr_celeba.csv",
    output_dir:       str = "data/roi_dataset",
    max_images:       int = 3000,
):
    """
    對 CelebA 每張圖：
      1. 用 MediaPipe 找關鍵點
      2. 裁切五官 ROI
      3. 存成 data/roi_dataset/{task}/{class}/{img}.jpg

    目錄結構（Keras ImageDataGenerator 可直接讀）：
      data/roi_dataset/
        nose/
          挺鼻型/  001.jpg  002.jpg ...
          寬鼻型/  ...
          標準鼻型/...
        eye/
          明亮眼型/...
          ...
    """
    output_dir = Path(output_dir)

    # 預建目錄
    for task_name, cfg in TASKS.items():
        for cls in cfg["classes"]:
            (output_dir / task_name / cls).mkdir(parents=True, exist_ok=True)

    # 讀 CelebA 屬性
    print("📂 讀取 CelebA 標籤...")
    attr_df = pd.read_csv(celeba_attr_file, index_col="image_id")
    attr_df = attr_df.replace(-1, 0).head(max_images)

    img_dir = Path(celeba_img_dir)
    stats   = {t: {c: 0 for c in TASKS[t]["classes"]} for t in TASKS}
    success = 0

    print(f"✂️  裁切五官 ROI（{len(attr_df)} 張圖）...")
    for img_name, row in tqdm(attr_df.iterrows(), total=len(attr_df)):
        img_path = str(img_dir / img_name)
        if not Path(img_path).exists():
            continue

        rois = extract_all_rois(img_path)
        if rois is None:
            continue

        stem = Path(img_name).stem
        for task_name, cfg in TASKS.items():
            if task_name not in rois:
                continue
            label    = cfg["label_fn"](row)
            save_path = output_dir / task_name / label / f"{stem}.jpg"
            # 用 imencode + write_bytes 繞過 OpenCV 不支援中文路徑的問題
            ret, buf = cv2.imencode('.jpg', rois[task_name])
            if ret:
                save_path.write_bytes(buf.tobytes())
            stats[task_name][label] += 1

        success += 1

    print(f"\n✅ 完成！成功處理 {success} 張圖")
    print("\n各類別數量：")
    for task_name, counts in stats.items():
        print(f"\n  {TASKS[task_name]['desc']}:")
        for cls, n in counts.items():
            bar = "█" * (n // 20)
            print(f"    {cls:10s}: {n:4d}  {bar}")

    return stats


# ═══════════════════════════════════════
# 3. 建立 CNN 模型
# ═══════════════════════════════════════

def build_cnn(num_classes: int, task_name: str,
              backbone: str = "mobilenetv2",
              use_focal_loss: bool = False) -> tf.keras.Model:
    """
    建立 CNN 分類器

    Args:
        backbone: "mobilenetv2"（預設）或 "efficientnetb0"（選用升級）

    架構選擇：
      MobileNetV2   — 深度可分離卷積，3.4M 參數，速度快，本專題主力
      EfficientNetB0 — 複合縮放設計，5.3M 參數，準確率稍高 ~0.5%

    兩者都是 CNN 家族，都用卷積層提取圖像特徵，差在內部設計策略。
    切換只需改 backbone 參數，其餘程式碼完全相同。

    遷移學習架構：
      Input (64x64x3)
        ↓
      CNN Backbone（預訓練 ImageNet，凍結前 80% 層）
        ↓
      GlobalAveragePooling2D
        ↓
      Dense(256) + ReLU + Dropout(0.4)
        ↓
      Dense(num_classes) + Softmax
    """
    # 選擇 backbone
    if backbone == "efficientnetb0":
        from tensorflow.keras.applications import EfficientNetB0
        base_model = EfficientNetB0(
            input_shape=(ROI_SIZE, ROI_SIZE, 3),
            include_top=False,
            weights="imagenet",
        )
        preprocess_fn = tf.keras.applications.efficientnet.preprocess_input
        print(f"  使用 EfficientNetB0（{len(base_model.layers)} 層，5.3M 參數）")
    else:
        base_model = MobileNetV2(
            input_shape=(ROI_SIZE, ROI_SIZE, 3),
            include_top=False,
            weights="imagenet",
        )
        preprocess_fn = tf.keras.applications.mobilenet_v2.preprocess_input
        print(f"  使用 MobileNetV2（{len(base_model.layers)} 層，3.4M 參數）")

    # 凍結前 80% 的層（只 fine-tune 後面幾層）
    freeze_until = int(len(base_model.layers) * 0.8)
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    for layer in base_model.layers[freeze_until:]:
        layer.trainable = True

    # 建立完整模型
    inputs = layers.Input(shape=(ROI_SIZE, ROI_SIZE, 3), name=f"{task_name}_input")
    x = preprocess_fn(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           name=f"{task_name}_output")(x)

    model = models.Model(inputs, outputs, name=f"cnn_{backbone}_{task_name}")
    loss_fn = focal_loss(gamma=2.0, alpha=0.25) if use_focal_loss else "categorical_crossentropy"
    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    return model


# ═══════════════════════════════════════
# 4. 訓練 CNN
# ═══════════════════════════════════════

def train_all_cnns(
    dataset_dir:     str  = "data/roi_dataset",
    model_dir:       str  = "models_cnn",
    backbone:        str  = "mobilenetv2",   # "mobilenetv2" 或 "efficientnetb0"
    use_focal_loss:  bool = False,           # True → Focal Loss；False → CrossEntropy
):
    """
    對每個任務訓練一個 CNN，並存模型

    資料增強（Data Augmentation）：
      訓練時隨機翻轉、旋轉、亮度調整
      → 讓模型更強健，減少 overfitting
      → 這也是深度學習實務的重要技術！
    """
    model_dir   = Path(model_dir)
    dataset_dir = Path(dataset_dir)
    model_dir.mkdir(exist_ok=True)

    results = {}

    for task_name, cfg in TASKS.items():
        # ── 斷點續訓檢測：若模型、設定與歷史均存在則跳過 ──
        if (model_dir / f"{task_name}_best.h5").exists() and \
           (model_dir / f"{task_name}_config.json").exists() and \
           (model_dir / f"{task_name}_history.json").exists():
            print(f"  ⏭️ 檢測到 {cfg['desc']} 已訓練完成，自動跳過並載入歷史精度。")
            try:
                with open(model_dir / f"{task_name}_history.json", encoding="utf-8") as f:
                    hist_data = json.load(f)
                    best_val_acc = max(hist_data.get("val_accuracy", [0.0]))
            except Exception:
                best_val_acc = 0.0
            results[task_name] = {
                "best_val_accuracy": best_val_acc,
                "classes": cfg["classes"],
            }
            continue

        task_dir    = dataset_dir / task_name
        num_classes = len(cfg["classes"])

        print(f"\n{'═'*50}")
        print(f"🧠 訓練 {cfg['desc']}  ({num_classes} 類)")
        print(f"{'═'*50}")

        # ── 資料增強 ──
        train_gen = ImageDataGenerator(
            rescale=1.0/255,
            validation_split=0.2,
            rotation_range=15,           # 隨機旋轉 ±15度
            width_shift_range=0.1,       # 水平平移
            height_shift_range=0.1,      # 垂直平移
            horizontal_flip=True,        # 水平翻轉
            brightness_range=[0.8, 1.2], # 亮度調整
            zoom_range=0.1,
        )
        val_gen = ImageDataGenerator(
            rescale=1.0/255,
            validation_split=0.2,
        )

        train_data = train_gen.flow_from_directory(
            task_dir,
            target_size=(ROI_SIZE, ROI_SIZE),
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            subset="training",
            classes=cfg["classes"],
            seed=42,
        )
        val_data = val_gen.flow_from_directory(
            task_dir,
            target_size=(ROI_SIZE, ROI_SIZE),
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            subset="validation",
            classes=cfg["classes"],
            seed=42,
        )

        print(f"  訓練集: {train_data.samples} 張")
        print(f"  驗證集: {val_data.samples} 張")
        print(f"  類別對應: {train_data.class_indices}")

        # ── 建立模型 ──
        model = build_cnn(num_classes, task_name, backbone=backbone,
                          use_focal_loss=use_focal_loss)
        print(f"  backbone: {backbone}")

        # ── Callbacks ──
        cb_list = [
            # Early Stopping：驗證 loss 不改善就停
            callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=6,
                restore_best_weights=True,
                verbose=1,
            ),
            # ReduceLR：學習率自動衰減
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                verbose=1,
            ),
            # 存最好的模型
            callbacks.ModelCheckpoint(
                filepath=str(model_dir / f"{task_name}_best.h5"),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=0,
            ),
        ]

        # ── 訓練 ──
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=EPOCHS,
            callbacks=cb_list,
            verbose=1,
        )

        # ── 評估 ──
        best_val_acc = max(history.history["val_accuracy"])
        print(f"\n  ✅ {cfg['desc']} 最佳驗證準確率: {best_val_acc:.1%}")

        # 存設定（類別順序）
        config_path = model_dir / f"{task_name}_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({
                "classes": cfg["classes"],
                "class_indices": train_data.class_indices,
            }, f, ensure_ascii=False, indent=2)

        results[task_name] = {
            "best_val_accuracy": best_val_acc,
            "classes": cfg["classes"],
        }

        # 畫訓練曲線（存圖）
        _save_training_curve(history, task_name, model_dir)

        # ── 釋放記憶體以防止 OOM 垃圾累積 ──
        import gc
        tf.keras.backend.clear_session()
        gc.collect()

    # 總結
    print(f"\n{'═'*50}")
    print("📊 訓練總結：")
    for task, r in results.items():
        print(f"  {TASKS[task]['desc']:12s}: {r['best_val_accuracy']:.1%}")

    summary_path = model_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n模型存在: {model_dir}/")
    return results


def _save_training_curve(history, task_name: str, model_dir: Path):
    """存訓練曲線圖（用 matplotlib）並保存原始訓練數據"""
    # ── 1. 保存原始 epoch-by-epoch 數據 ──
    try:
        raw_history_path = model_dir / f"{task_name}_history.json"
        # 將 numpy float32 轉換為標準 float 以進行 json 序列化
        serializable_history = {
            k: [float(v) for v in l]
            for k, l in history.history.items()
        }
        with open(raw_history_path, "w", encoding="utf-8") as f:
            json.dump(serializable_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        pass

    # ── 2. 存畫圖 ──
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.plot(history.history["accuracy"],     label="train")
        ax1.plot(history.history["val_accuracy"], label="val")
        ax1.set_title(f"{task_name} Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.legend()

        ax2.plot(history.history["loss"],     label="train")
        ax2.plot(history.history["val_loss"], label="val")
        ax2.set_title(f"{task_name} Loss")
        ax2.set_xlabel("Epoch")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(str(model_dir / f"{task_name}_curve.png"), dpi=120)
        plt.close()
    except Exception:
        pass   # 畫圖失敗不影響訓練


# ═══════════════════════════════════════
# 5. 載入模型 & 預測
# ═══════════════════════════════════════

_loaded_models  = {}
_loaded_configs = {}

def load_cnn_models(model_dir: str = "models_cnn"):
    """載入所有訓練好的 CNN 模型"""
    global _loaded_models, _loaded_configs
    model_dir = Path(model_dir)

    for task_name in TASKS:
        model_path  = model_dir / f"{task_name}_best.h5"
        config_path = model_dir / f"{task_name}_config.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"找不到模型: {model_path}\n"
                f"請先執行: python step2c_cnn.py --build && python step2c_cnn.py --train"
            )

        _loaded_models[task_name]  = tf.keras.models.load_model(
            str(model_path),
            custom_objects={
                'TrueDivide': TrueDivide,
                'Subtract': CustomSubtract
            },
            compile=False
        )
        with open(config_path, encoding="utf-8") as f:
            _loaded_configs[task_name] = json.load(f)

    print("[OK] CNN 模型載入成功")
    return _loaded_models, _loaded_configs


def predict_with_cnn(image_path: str) -> dict:
    """
    對一張圖片做 CNN 面相預測

    完整流程：
      1. MediaPipe 找關鍵點
      2. 裁切五官 ROI
      3. 每個 CNN 分別預測
      4. 整合成面相報告

    Returns:
        {
          "nose": {"label": "挺鼻型", "confidence": 0.87, "proba": [...]},
          "eye":  {...},
          ...
        }
    """
    if not _loaded_models:
        load_cnn_models()

    rois = extract_all_rois(image_path)
    if rois is None:
        return None

    predictions = {}
    for task_name, model in _loaded_models.items():
        if task_name not in rois:
            continue

        cfg      = _loaded_configs[task_name]
        classes  = cfg["classes"]

        # 預處理：歸一化到 [0,1]
        roi_rgb  = cv2.cvtColor(rois[task_name], cv2.COLOR_BGR2RGB)
        roi_norm = roi_rgb.astype("float32") / 255.0
        roi_inp  = np.expand_dims(roi_norm, axis=0)   # (1, 64, 64, 3)

        # CNN 推理
        proba      = model.predict(roi_inp, verbose=0)[0]   # (num_classes,)
        pred_idx   = int(np.argmax(proba))
        pred_label = classes[pred_idx]
        confidence = float(proba[pred_idx])

        predictions[task_name] = {
            "label":      pred_label,
            "confidence": confidence,
            "proba":      {c: float(p) for c, p in zip(classes, proba)},
        }

    return predictions


# ═══════════════════════════════════════
# 6. 面相解讀（同 step2b）
# ═══════════════════════════════════════

# 直接從 step2b 匯入解讀字典
from step2b_train import ML_PHYSIOGNOMY


def generate_cnn_report(predictions: dict) -> dict:
    """整合 CNN 預測結果 → 面相報告"""
    if predictions is None:
        return None

    analysis    = {}
    total_score = 0.0

    for task_name, pred in predictions.items():
        label      = pred["label"]
        confidence = pred["confidence"]
        phys = ML_PHYSIOGNOMY.get(label, {
            "title":    label,
            "analysis": "此面相類型暫無詳細解讀。",
            "score":    0.7,
        })
        analysis[task_name] = {
            **pred,
            "title":    phys["title"],
            "analysis": phys["analysis"],
            "score":    phys["score"],
        }
        total_score += phys["score"] * confidence

    avg_score = total_score / max(len(analysis), 1)
    top_key   = max(analysis, key=lambda k: analysis[k]["score"] * analysis[k]["confidence"])
    top_info  = analysis[top_key]

    summary_parts = []
    chan_names = {"nose": "鼻相", "eye": "眼相", "brow": "眉相", "mouth": "口相", "face": "臉相"}
    for task_name, info in analysis.items():
        name = chan_names.get(task_name, task_name)
        summary_parts.append(
            f"• {name}通道：CNN 辨識為 {info['label']}（信心度 {info['confidence']:.0%}），{info['analysis']}"
        )
    summary_detail = "\n".join(summary_parts)

    summary = (
        f"🤖 CNN 深度學習模型（MobileNetV2 遷移學習）分析完成：\n"
        f"{summary_detail}\n\n"
        f"🌟 最優勢特質：「{top_info['title']}」（信心度 {top_info['confidence']:.0%}）\n"
        f"📊 整體面相評分 {avg_score:.0%}。"
    )

    return {
        "analysis":  analysis,
        "summary":   summary,
        "avg_score": avg_score,
    }


# ═══════════════════════════════════════
# 主程式
# ═══════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="面相 CNN 分類器")
    parser.add_argument("--build",    action="store_true", help="裁切 ROI 資料集")
    parser.add_argument("--train",    action="store_true", help="訓練 CNN")
    parser.add_argument("--predict",  type=str, default=None, help="預測圖片路徑")
    parser.add_argument("--max",      type=int, default=3000, help="最大訓練圖片數")
    parser.add_argument("--backbone", type=str, default="mobilenetv2",
                        choices=["mobilenetv2", "efficientnetb0"],
                        help="CNN backbone 選擇（預設 mobilenetv2）")
    args = parser.parse_args()

    if args.build:
        build_roi_dataset(max_images=args.max)

    elif args.train:
        train_all_cnns(backbone=args.backbone)

    elif args.predict:
        print(f"🔍 分析: {args.predict}")
        preds  = predict_with_cnn(args.predict)
        report = generate_cnn_report(preds)

        if report is None:
            print("❌ 未偵測到人臉")
            return

        print("\n" + "="*50)
        print("  ✨ CNN 面相分析報告")
        print("="*50)
        for task, info in report["analysis"].items():
            bar = "█" * round(info["confidence"] * 10)
            print(f"\n【{TASKS[task]['desc']}】{info['label']}")
            print(f"  信心度: {bar} {info['confidence']:.0%}")
            print(f"  {info['title']}")
            print(f"  {info['analysis']}")
            print(f"  各類別機率: {info['proba']}")
        print(f"\n總結: {report['summary']}")

    else:
        print("""
CNN 面相分類器使用說明：

  步驟 1: 裁切 ROI 資料集
    python step2c_cnn.py --build --max 3000

  步驟 2: 訓練 CNN（預設 MobileNetV2）
    python step2c_cnn.py --train

  步驟 2b: 改用 EfficientNetB0（選用比較實驗）
    python step2c_cnn.py --train --backbone efficientnetb0

  步驟 3: 預測
    python step2c_cnn.py --predict your_face.jpg

  兩種 backbone 都是 CNN 家族：
    mobilenetv2    - 深度可分離卷積，3.4M 參數，速度快（主力）
    efficientnetb0 - 複合縮放設計，5.3M 參數，準確率稍高（比較用）
        """)


if __name__ == "__main__":
    main()
