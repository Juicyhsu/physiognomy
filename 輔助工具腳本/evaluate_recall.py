import os
import sys
import json
import numpy as np
from pathlib import Path

# 強制將 stdout 設置為 utf-8
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 將專案根目錄加入路徑以載入 step2c_cnn.py 中定義的自訂層
sys.path.append(str(Path(__file__).resolve().parents[1]))
from step2c_cnn import TrueDivide, CustomSubtract

# 設定 TensorFlow 警告級別
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

def evaluate_task_recall(task_name, classes, ce_model_path, focal_model_path, dataset_dir="data/roi_dataset"):
    task_dir = Path(dataset_dir) / task_name
    if not task_dir.exists():
        print(f"Error: {task_dir} does not exist.")
        return None

    # 建立 Validation Generator (不打亂順序以利對齊 Label)
    val_gen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
    val_data = val_gen.flow_from_directory(
        str(task_dir),
        target_size=(64, 64),
        batch_size=32,
        class_mode="categorical",
        subset="validation",
        classes=classes,
        seed=42,
        shuffle=False,
    )
    
    y_true = val_data.classes
    
    print(f"\n==========================================")
    print(f"📊 開始評估部位 [{task_name.upper()}] 各類別 Recall")
    print(f"==========================================")
    
    # 1. 評估 CE 模型
    try:
        model_ce = load_model(
            str(ce_model_path), 
            custom_objects={
                'TrueDivide': TrueDivide,
                'Subtract': CustomSubtract
            }, 
            compile=False
        )
        preds_ce = model_ce.predict(val_data, verbose=0)
        y_pred_ce = np.argmax(preds_ce, axis=1)
        # 加入 labels 參數以防預測結果漏掉類別時崩潰
        report_ce = classification_report(
            y_true, 
            y_pred_ce, 
            labels=list(range(len(classes))), 
            target_names=classes, 
            output_dict=True
        )
    except Exception as e:
        print(f"評估 CE 模型時出錯: {e}")
        return None
        
    # 2. 評估 Focal Loss 模型
    try:
        model_fl = load_model(
            str(focal_model_path), 
            custom_objects={
                'TrueDivide': TrueDivide,
                'Subtract': CustomSubtract
            }, 
            compile=False
        )
        preds_fl = model_fl.predict(val_data, verbose=0)
        y_pred_fl = np.argmax(preds_fl, axis=1)
        report_fl = classification_report(
            y_true, 
            y_pred_fl, 
            labels=list(range(len(classes))), 
            target_names=classes, 
            output_dict=True
        )
    except Exception as e:
        print(f"評估 Focal Loss 模型時出錯: {e}")
        return None

    # 3. 印出對照表
    print(f"\n💡 [{task_name.upper()} 各類別 Recall 對照結果]")
    print(f"| {'類別名稱':<12} | {'原本 CE':^10} | {'Focal Loss':^12} | {'Recall 增幅':^12} |")
    print(f"|{'-'*14}|{'-'*12}|{'-'*14}|{'-'*14}|")
    
    for cls in classes:
        rec_ce = report_ce[cls]['recall']
        rec_fl = report_fl[cls]['recall']
        diff = rec_fl - rec_ce
        sign = "+" if diff >= 0 else ""
        print(f"| {cls:<10} | {rec_ce:>9.1%} | {rec_fl:>11.1%} | {sign}{diff:>10.1%} |")
        
    # 印出整體 Macro F1-score
    macro_f1_ce = report_ce['macro avg']['f1-score']
    macro_f1_fl = report_fl['macro avg']['f1-score']
    diff_f1 = macro_f1_fl - macro_f1_ce
    sign_f1 = "+" if diff_f1 >= 0 else ""
    print(f"\n📈 整體均衡度 (Macro F1-Score) 變化: {macro_f1_ce:.1%} -> {macro_f1_fl:.1%} ({sign_f1}{diff_f1:.1%})")

def main():
    # 1. 臉型 (Exp C vs Exp G)
    evaluate_task_recall(
        task_name="face",
        classes=["瓜子臉", "圓潤臉", "稜角臉", "標準臉型"],
        ce_model_path=Path("models_cnn_expC_mobilenet_lowLR/face_best.h5"),
        focal_model_path=Path("models_cnn_expG_mobilenet_lowLR_focal/face_best.h5")
    )
    
    # 2. 鼻型 (Exp C vs Exp G)
    evaluate_task_recall(
        task_name="nose",
        classes=["挺鼻型", "寬鼻型", "標準鼻型"],
        ce_model_path=Path("models_cnn_expC_mobilenet_lowLR/nose_best.h5"),
        focal_model_path=Path("models_cnn_expG_mobilenet_lowLR_focal/nose_best.h5")
    )

    # 3. 眼型 (Exp D vs Exp H) - 批次量 16 的單一控制變量
    evaluate_task_recall(
        task_name="eye",
        classes=["明亮眼型", "細眼型", "疲態眼型"],
        ce_model_path=Path("models_cnn_expD_mobilenet_batch16/eye_best.h5"),
        focal_model_path=Path("models_cnn_expH_mobilenet_batch16_focal/eye_best.h5")
    )

    # 4. 眉型 (Exp D vs Exp H) - 批次量 16 的單一控制變量
    evaluate_task_recall(
        task_name="brow",
        classes=["上揚眉", "濃眉", "標準眉"],
        ce_model_path=Path("models_cnn_expD_mobilenet_batch16/brow_best.h5"),
        focal_model_path=Path("models_cnn_expH_mobilenet_batch16_focal/brow_best.h5")
    )

    # 5. 嘴型 (Exp A vs Exp E) - 預設參數的單一控制變量
    evaluate_task_recall(
        task_name="mouth",
        classes=["豐唇型", "笑容型", "標準嘴型"],
        ce_model_path=Path("models_cnn_expA_mobilenet/mouth_best.h5"),
        focal_model_path=Path("models_cnn_expE_mobilenet_focal/mouth_best.h5")
    )

if __name__ == "__main__":
    main()
