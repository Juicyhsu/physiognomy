"""
================================================
面相引擎 路線 B — 機器學習分類器
================================================
用 CelebA 的標籤「從資料學習」五官分類

流程：
  1. 對 CelebA 每張圖跑 MediaPipe → 取得特徵值
  2. 把 CelebA 標籤 (Big_Nose / Oval_Face...) 轉成 y
  3. 訓練 RandomForest 分類器
  4. 存模型、預測新圖片

執行：
  python step2b_train.py          # 訓練（跑一次就好）
  python step2b_train.py --predict test_face.jpg  # 預測
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── sklearn ──
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── 自己的模組 ──
from step1_face_detection import analyze_face


# ═══════════════════════════════════════
# 1. CelebA 標籤 → 面相類別  對照表
# ═══════════════════════════════════════

"""
CelebA 有 40 個二元標籤，我們挑有面相意義的，
轉成多類別的「面相類型」:

五官       CelebA 標籤              → 面相類別
────────────────────────────────────────────────
鼻型     Pointy_Nose=1             → 挺鼻型
         Big_Nose=1 & Pointy=0    → 寬鼻型
         兩者都0                   → 標準鼻型

眼型     Narrow_Eyes=1             → 細眼型
         Bags_Under_Eyes=1        → 疲態眼型
         兩者都0                   → 明亮眼型

眉型     Arched_Eyebrows=1        → 上揚眉
         Bushy_Eyebrows=1         → 濃眉
         兩者都0                   → 標準眉

嘴型     Big_Lips=1               → 豐唇型
         Mouth_Slightly_Open=1    → 開口型
         兩者都0                   → 標準嘴型

臉型     Oval_Face=1              → 瓜子臉
         Chubby=1                 → 圓潤臉
         High_Cheekbones=1        → 高顴骨臉
         兩者都0                   → 標準臉型
"""

# 特徵向量的欄位名稱（要跟 step1 輸出一致）
FEATURE_COLS = [
    "face_ratio", "jaw_ratio",
    "upper_zone_ratio", "middle_zone_ratio", "lower_zone_ratio",
    "san_ting_balance",
    "left_eye_ratio", "right_eye_ratio", "eye_gap_ratio",
    "brow_eye_ratio", "brow_eye_distance",
    "nose_width_ratio", "nose_height_ratio", "nose_shape_ratio",
    "mouth_width_ratio", "mouth_shape_ratio", "lip_thickness",
]


def celeba_to_labels(row) -> dict:
    """
    把 CelebA 的一行屬性轉成面相類別字典
    row: CelebA attr dataframe 的一行 (0/1)
    """
    def v(col):  # 安全取值
        return int(row.get(col, 0))

    # 鼻型
    if v("Pointy_Nose"):
        nose = "挺鼻型"
    elif v("Big_Nose"):
        nose = "寬鼻型"
    else:
        nose = "標準鼻型"

    # 眼型
    if v("Narrow_Eyes"):
        eye = "細眼型"
    elif v("Bags_Under_Eyes"):
        eye = "疲態眼型"
    else:
        eye = "明亮眼型"

    # 眉型
    if v("Arched_Eyebrows"):
        brow = "上揚眉"
    elif v("Bushy_Eyebrows"):
        brow = "濃眉"
    else:
        brow = "標準眉"

    # 嘴型
    if v("Big_Lips"):
        mouth = "豐唇型"
    elif v("Smiling"):
        mouth = "笑容型"
    else:
        mouth = "標準嘴型"

    # 臉型
    if v("Oval_Face"):
        face = "瓜子臉"
    elif v("Chubby"):
        face = "圓潤臉"
    elif v("High_Cheekbones") and not v("Oval_Face"):
        face = "稜角臉"
    else:
        face = "標準臉型"

    return {
        "nose_type":  nose,
        "eye_type":   eye,
        "brow_type":  brow,
        "mouth_type": mouth,
        "face_type":  face,
    }


# ═══════════════════════════════════════
# 2. 建立訓練資料集
# ═══════════════════════════════════════

def build_dataset(
    celeba_img_dir:  str = "data/celeba/img_align_celeba",
    celeba_attr_file: str = "data/celeba/list_attr_celeba.txt",
    output_csv:      str = "data/features_labeled.csv",
    max_images:      int = 2000,   # 先用 2000 張試水
):
    """
    對 CelebA 圖片跑 MediaPipe，把特徵值 + 標籤存成 CSV

    參數:
        max_images: 先用小量測試，確認流程正確後再加大
                    2000 張約需 5~10 分鐘
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # 讀 CelebA 屬性標籤
    print("📂 讀取 CelebA 標籤...")
    attr_df = pd.read_csv(celeba_attr_file, sep=r"\s+", skiprows=1)
    attr_df = attr_df.replace(-1, 0)
    attr_df = attr_df.head(max_images)

    img_dir = Path(celeba_img_dir)
    rows = []

    print(f"🔍 開始對 {len(attr_df)} 張圖跑 MediaPipe...")
    for img_name, row in tqdm(attr_df.iterrows(), total=len(attr_df)):
        img_path = str(img_dir / img_name)
        if not Path(img_path).exists():
            continue

        # 用暫存目錄避免寫太多檔案
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            features, _ = analyze_face(img_path, output_dir=tmp)

        if features is None:
            continue  # 偵測失敗 → 跳過

        # 合併特徵值 + 面相標籤
        labels = celeba_to_labels(row)
        entry = {**features, **labels, "img_name": img_name}
        rows.append(entry)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"✅ 資料集建立完成：{len(df)} 筆")
    print(f"   儲存至: {output_csv}")
    print(f"\n標籤分布：")
    for col in ["nose_type", "eye_type", "brow_type", "mouth_type", "face_type"]:
        print(f"\n  {col}:")
        print(df[col].value_counts().to_string(header=False))
    return df


# ═══════════════════════════════════════
# 3. 訓練分類器
# ═══════════════════════════════════════

# 要訓練的目標 (五官類別) → 對應的特徵欄位
CLASSIFIER_CONFIG = {
    "nose_type": {
        "features": ["nose_width_ratio", "nose_height_ratio", "nose_shape_ratio"],
        "description": "鼻型分類器",
    },
    "eye_type": {
        "features": ["left_eye_ratio", "right_eye_ratio", "eye_gap_ratio", "brow_eye_distance"],
        "description": "眼型分類器",
    },
    "brow_type": {
        "features": ["brow_eye_ratio", "brow_eye_distance", "left_eye_ratio"],
        "description": "眉型分類器",
    },
    "mouth_type": {
        "features": ["mouth_width_ratio", "mouth_shape_ratio", "lip_thickness"],
        "description": "嘴型分類器",
    },
    "face_type": {
        "features": ["face_ratio", "jaw_ratio", "upper_zone_ratio",
                     "middle_zone_ratio", "lower_zone_ratio", "san_ting_balance"],
        "description": "臉型分類器",
    },
}


def train_classifiers(
    csv_path:    str = "data/features_labeled.csv",
    model_dir:   str = "models",
):
    """
    從 CSV 讀取訓練資料，對每個五官訓練一個 RandomForest 分類器

    模型存到 models/ 目錄：
      models/nose_type.pkl
      models/eye_type.pkl
      models/brow_type.pkl
      models/mouth_type.pkl
      models/face_type.pkl
      models/label_encoders.pkl
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)

    print("📖 讀取訓練資料...")
    df = pd.read_csv(csv_path, encoding="utf-8")
    print(f"   共 {len(df)} 筆資料")

    classifiers    = {}
    label_encoders = {}

    for target, config in CLASSIFIER_CONFIG.items():
        print(f"\n{'─'*45}")
        print(f"🎯 訓練 {config['description']}...")

        feat_cols = config["features"]
        X = df[feat_cols].fillna(df[feat_cols].median())
        y_raw = df[target]

        # Label encoding (字串 → 整數)
        le = LabelEncoder()
        y  = le.fit_transform(y_raw)
        label_encoders[target] = le

        print(f"   類別: {list(le.classes_)}")
        print(f"   特徵: {feat_cols}")

        # 切訓練/測試集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Pipeline: 標準化 + RandomForest
        # RandomForest 不太需要標準化，但加了不會更差，對別的模型有幫助
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=200,   # 200棵樹
                max_depth=8,        # 限制深度防過擬合
                min_samples_leaf=3,
                class_weight="balanced",  # 處理類別不平衡
                random_state=42,
                n_jobs=-1,          # 用全部 CPU
            )),
        ])

        pipe.fit(X_train, y_train)

        # 評估
        y_pred   = pipe.predict(X_test)
        acc      = accuracy_score(y_test, y_pred)
        cv_score = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")

        print(f"   測試準確率: {acc:.1%}")
        print(f"   交叉驗證:   {cv_score.mean():.1%} ± {cv_score.std():.1%}")
        print(classification_report(
            y_test, y_pred,
            target_names=le.classes_,
            zero_division=0
        ))

        classifiers[target] = pipe

        # 存模型
        model_path = model_dir / f"{target}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(pipe, f)

    # 存所有 label encoders
    with open(model_dir / "label_encoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)

    print(f"\n✅ 所有模型已存到 {model_dir}/")
    print("   下次直接 load 不用重訓！")
    return classifiers, label_encoders


# ═══════════════════════════════════════
# 4. 載入模型 & 預測
# ═══════════════════════════════════════

def load_models(model_dir: str = "models"):
    """載入訓練好的模型"""
    model_dir = Path(model_dir)
    classifiers    = {}
    label_encoders = {}

    for target in CLASSIFIER_CONFIG:
        path = model_dir / f"{target}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"找不到模型: {path}\n請先執行訓練: python step2b_train.py")
        with open(path, "rb") as f:
            classifiers[target] = pickle.load(f)

    le_path = model_dir / "label_encoders.pkl"
    with open(le_path, "rb") as f:
        label_encoders = pickle.load(f)

    print("✅ 模型載入成功")
    return classifiers, label_encoders


def predict_face(features: dict, classifiers: dict, label_encoders: dict) -> dict:
    """
    用訓練好的模型預測一張臉的五官類型

    輸入:  features dict (step1 的輸出)
    輸出:  predictions dict {
               "nose_type":  ("挺鼻型", 0.82),  # (預測類別, 信心度)
               "eye_type":   ("明亮眼型", 0.74),
               ...
           }
    """
    predictions = {}

    for target, config in CLASSIFIER_CONFIG.items():
        clf = classifiers[target]
        le  = label_encoders[target]

        # 取該分類器需要的特徵
        X = np.array([[features.get(f, 0.0) for f in config["features"]]])

        # 預測 + 信心度
        pred_idx  = clf.predict(X)[0]
        proba     = clf.predict_proba(X)[0]
        pred_label = le.inverse_transform([pred_idx])[0]
        confidence = float(proba[pred_idx])

        predictions[target] = (pred_label, confidence)

    return predictions


# ═══════════════════════════════════════
# 5. 面相解讀字典 (ML 版)
# ═══════════════════════════════════════

ML_PHYSIOGNOMY = {
    # ── 鼻型 ──
    "挺鼻型": {
        "title": "鼻梁高挺，財運亨通",
        "analysis": (
            "鼻梁高而挺直，在面相學中是「財帛宮」豐盛的表現。"
            "此類人自信心強，行事果斷，善於把握財富機遇。"
            "中年後財運尤為旺盛，適合投資理財或自行創業。"
        ),
        "score": 0.9,
    },
    "寬鼻型": {
        "title": "鼻翼寬廣，人脈帶財",
        "analysis": (
            "鼻翼較寬，代表財運多來自人際關係與合作。"
            "個性豪爽，慷慨待人，人緣好，貴人多。"
            "適合需要廣泛人脈的行業，如業務、公關、餐飲。"
        ),
        "score": 0.75,
    },
    "標準鼻型": {
        "title": "鼻型端正，財運穩健",
        "analysis": (
            "鼻型勻稱，財運不偏不倚，穩健而持續。"
            "靠自身踏實努力累積財富，不喜歡走捷徑。"
            "細水長流型的財富積累，晚年較為豐厚。"
        ),
        "score": 0.72,
    },

    # ── 眼型 ──
    "明亮眼型": {
        "title": "眼神明亮，觀察敏銳",
        "analysis": (
            "眼睛明亮有神，代表心思開明、洞察力強。"
            "善於觀察人心，在人際交往上具備高情商。"
            "直覺準確，對新事物接受度高，適應力強。"
        ),
        "score": 0.85,
    },
    "細眼型": {
        "title": "眼神深邃，思慮縝密",
        "analysis": (
            "眼型較細，代表個性謹慎、思維深度高。"
            "做事一絲不苟，追求完美，適合需要精密分析的工作。"
            "內心世界豐富，感情忠誠，不輕易表露情緒。"
        ),
        "score": 0.78,
    },
    "疲態眼型": {
        "title": "眼底有痕，需重視休養",
        "analysis": (
            "眼底有疲態紋，提示近期生活壓力較大或作息不規律。"
            "面相學中此型主「勞碌奔波」，但同時代表努力不懈。"
            "建議調整作息，適度休息，運勢將因此提升。"
        ),
        "score": 0.6,
    },

    # ── 眉型 ──
    "上揚眉": {
        "title": "眉型上揚，積極進取",
        "analysis": (
            "眉尾上揚，主個性積極、充滿幹勁，凡事力求上進。"
            "領導特質明顯，不甘居人後，事業心旺盛。"
            "在競爭環境中能快速冒尖，前途看好。"
        ),
        "score": 0.83,
    },
    "濃眉": {
        "title": "眉毛濃密，個性鮮明",
        "analysis": (
            "眉毛濃密，代表個性鮮明、感情豐沛、行動力強。"
            "說話直接，不喜繞彎，重義氣，對朋友真誠。"
            "情緒表達直率，兄弟朋友緣佳。"
        ),
        "score": 0.80,
    },
    "標準眉": {
        "title": "眉型端正，均衡穩重",
        "analysis": (
            "眉毛不濃不淡、不長不短，代表個性均衡理性。"
            "處事不偏激，情緒穩定，是職場上可靠的夥伴。"
            "人際關係和諧，長期運勢穩定上升。"
        ),
        "score": 0.73,
    },

    # ── 嘴型 ──
    "豐唇型": {
        "title": "嘴唇豐厚，感情豐沛",
        "analysis": (
            "嘴唇豐潤，主感情豐沛，對伴侶溫柔體貼，愛家顧家。"
            "享受生活，重視感官體驗，品味不俗。"
            "口才佳，表達感情直接，感情運勢旺盛。"
        ),
        "score": 0.82,
    },
    "笑容型": {
        "title": "嘴角上揚，人緣極佳",
        "analysis": (
            "嘴角自然上揚，給人親切愉快的第一印象。"
            "人緣極佳，走到哪裡都受到歡迎，是天生的社交達人。"
            "樂觀正向的態度帶來好運，事業上多有貴人相助。"
        ),
        "score": 0.87,
    },
    "標準嘴型": {
        "title": "嘴型適中，言行一致",
        "analysis": (
            "嘴型大小適中，主說話算話、信用好、重承諾。"
            "感情專一，對伴侶忠誠，適合穩定的長期關係。"
            "職場上因誠信而建立好口碑。"
        ),
        "score": 0.74,
    },

    # ── 臉型 ──
    "瓜子臉": {
        "title": "瓜子臉，均衡貴相",
        "analysis": (
            "橢圓臉型被視為五行最均衡的貴相，一生運勢平穩。"
            "性格圓融，多才多藝，適應力強，貴人緣佳。"
            "各年齡段皆有收穫，晚年尤為福壽雙全。"
        ),
        "score": 0.90,
    },
    "圓潤臉": {
        "title": "圓潤臉，人緣財氣旺",
        "analysis": (
            "面相豐腴圓潤，主財氣旺、人緣好、晚年有福。"
            "天生親和力強，讓人感到親近舒適。"
            "財運多來自人際關係，適合服務業或管理職。"
        ),
        "score": 0.78,
    },
    "稜角臉": {
        "title": "顴骨高挺，意志堅定",
        "analysis": (
            "顴骨高聳，主意志力強、不輕易妥協、有主見。"
            "行事果斷，在職場上有威信，適合擔任管理職。"
            "獨立自主，靠自身實力開創局面。"
        ),
        "score": 0.80,
    },
    "標準臉型": {
        "title": "臉型標準，穩健格局",
        "analysis": (
            "臉型比例均衡，個性踏實、處事中庸，不走極端。"
            "適應各種環境，是職場上的通用型人才。"
            "運勢隨努力穩定累積，中年後有明顯提升。"
        ),
        "score": 0.72,
    },
}


def generate_ml_report(features: dict, classifiers: dict, label_encoders: dict) -> dict:
    """
    完整的 ML 面相報告生成

    回傳:
      {
        "predictions": {"nose_type": ("挺鼻型", 0.82), ...},
        "analysis":    {"nose_type": {...面相解讀...}, ...},
        "summary":     "綜合總結文字",
        "avg_score":   0.81,
      }
    """
    # 預測
    predictions = predict_face(features, classifiers, label_encoders)

    # 對應面相解讀
    analysis = {}
    total_score = 0.0

    for target, (label, confidence) in predictions.items():
        phys = ML_PHYSIOGNOMY.get(label, {
            "title":    label,
            "analysis": "此面相類型暫無詳細解讀。",
            "score":    0.7,
        })
        analysis[target] = {
            "label":      label,
            "confidence": confidence,        # 模型信心度 (0~1)
            "title":      phys["title"],
            "analysis":   phys["analysis"],
            "score":      phys["score"],
        }
        total_score += phys["score"] * confidence  # 加權平均

    avg_score = total_score / len(predictions)

    # 找最突出特質
    top_key  = max(analysis, key=lambda k: analysis[k]["score"] * analysis[k]["confidence"])
    top_info = analysis[top_key]

    summary = (
        f"機器學習模型分析完成，共評估五個面相維度。\n"
        f"您最突出的特質為「{top_info['title']}」（模型信心度 {top_info['confidence']:.0%}）。\n"
        f"整體面相評分 {avg_score:.0%}，"
        + ("屬於格局宏大的貴人面相，運勢各方面均衡發展。" if avg_score > 0.82
           else "屬於穩健上進型，持續耕耘將有豐厚回報。" if avg_score > 0.74
           else "各有所長，善用優勢補強弱項，運勢可大幅提升。")
    )

    return {
        "predictions": predictions,
        "analysis":    analysis,
        "summary":     summary,
        "avg_score":   avg_score,
    }


def print_ml_report(report: dict):
    """格式化輸出 ML 面相報告"""
    TARGET_NAMES = {
        "nose_type":  "鼻型",
        "eye_type":   "眼型",
        "brow_type":  "眉型",
        "mouth_type": "嘴型",
        "face_type":  "臉型",
    }

    print("\n" + "="*55)
    print("    ✨ 機器學習面相分析報告 (路線 B) ✨")
    print("="*55)

    for target, info in report["analysis"].items():
        name = TARGET_NAMES.get(target, target)
        bar  = "█" * round(info["confidence"] * 10)
        print(f"\n【{name}】{info['label']}  (信心度 {info['confidence']:.0%}  {bar})")
        print(f"  → {info['title']}")
        print(f"  {info['analysis']}")

    print(f"\n{'─'*55}")
    print(f"📋 綜合總結：")
    print(f"  {report['summary']}")
    print(f"\n整體評分: {report['avg_score']:.0%}")
    print("="*55 + "\n")


# ═══════════════════════════════════════
# 主程式
# ═══════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="面相引擎 路線B — 機器學習分類器")
    parser.add_argument("--build",   action="store_true", help="從 CelebA 建立訓練資料集")
    parser.add_argument("--train",   action="store_true", help="訓練分類器")
    parser.add_argument("--predict", type=str, default=None, help="預測指定圖片")
    parser.add_argument("--max",     type=int, default=2000, help="最大訓練圖片數")
    args = parser.parse_args()

    if args.build:
        # Step A: 建立資料集
        build_dataset(max_images=args.max)

    elif args.train:
        # Step B: 訓練模型
        train_classifiers()

    elif args.predict:
        # Step C: 預測新圖片
        import tempfile
        print(f"🔍 分析圖片: {args.predict}")

        with tempfile.TemporaryDirectory() as tmp:
            features, _ = analyze_face(args.predict, output_dir=tmp)

        if features is None:
            print("❌ 未偵測到人臉")
            return

        clfs, les = load_models()
        report = generate_ml_report(features, clfs, les)
        print_ml_report(report)

    else:
        # 沒有參數：顯示完整教學
        print("""
面相引擎 路線 B — 機器學習分類器
══════════════════════════════════

使用步驟：

第一步：建立訓練資料集（跑 MediaPipe 對 CelebA 圖片）
  python step2b_train.py --build --max 2000
    ↑ 先用 2000 張，確認流程正確後改成 10000

第二步：訓練分類器
  python step2b_train.py --train

第三步：預測你的臉
  python step2b_train.py --predict test_face.jpg

完成後，step3_app.py 的 process_face() 函式
會自動呼叫這裡的 generate_ml_report()
        """)


if __name__ == "__main__":
    main()
