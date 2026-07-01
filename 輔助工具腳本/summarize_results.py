import json
import sys
from pathlib import Path

# 強制將 stdout 設置為 utf-8 以防 cp950 編碼錯誤
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    exp_dirs = {
        "A (MobileNet, CE)": "models_cnn_expA_mobilenet",
        "B (EfficientNet, CE)": "models_cnn_expB_efficientnet",
        "C (MobileNet, lowLR, CE)": "models_cnn_expC_mobilenet_lowLR",
        "D (MobileNet, batch16, CE)": "models_cnn_expD_mobilenet_batch16",
        "E (MobileNet, Focal)": "models_cnn_expE_mobilenet_focal",
        "F (EfficientNet, Focal)": "models_cnn_expF_efficientnet_focal",
        "G (MobileNet, lowLR, Focal)": "models_cnn_expG_mobilenet_lowLR_focal",
        "H (MobileNet, batch16, Focal)": "models_cnn_expH_mobilenet_batch16_focal"
    }
    
    tasks = ["nose", "eye", "brow", "mouth", "face"]
    task_names = {
        "nose": "鼻型分類器",
        "eye": "眼型分類器",
        "brow": "眉型分類器",
        "mouth": "嘴型分類器",
        "face": "臉型分類器"
    }

    results = {}
    for label, folder_name in exp_dirs.items():
        folder = Path(folder_name)
        summary_file = folder / "training_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    results[label] = data
            except Exception as e:
                print(f"Error reading {summary_file}: {e}")
        else:
            print(f"Warning: {summary_file} does not exist.")

    print("# 8組對照實驗精度總結\n")
    
    # 建立表格頭部
    header = "| 實驗組別 | " + " | ".join([task_names[t] for t in tasks]) + " |"
    divider = "| --- | " + " | ".join(["---" for _ in tasks]) + " |"
    print(header)
    print(divider)
    
    for label in exp_dirs.keys():
        row_values = []
        for t in tasks:
            if label in results and t in results[label]:
                acc = results[label][t].get("best_val_accuracy", 0.0)
                row_values.append(f"{acc:.1%}")
            else:
                row_values.append("N/A")
        print(f"| {label} | " + " | ".join(row_values) + " |")

    # 另外分析一下 Focal Loss 帶來的改變
    print("\n## Focal Loss 效能增幅分析 (Focal Loss 組 - CrossEntropy 組)")
    print("\n| 對照組別 | 鼻型 | 眼型 | 眉型 | 嘴型 | 臉型 |")
    print("| --- | --- | --- | --- | --- | --- |")
    
    comparisons = [
        ("A vs E (MobileNet V2)", "A (MobileNet, CE)", "E (MobileNet, Focal)"),
        ("B vs F (EfficientNet B0)", "B (EfficientNet, CE)", "F (EfficientNet, Focal)"),
        ("C vs G (MobileNet V2, lowLR)", "C (MobileNet, lowLR, CE)", "G (MobileNet, lowLR, Focal)"),
        ("D vs H (MobileNet V2, batch16)", "D (MobileNet, batch16, CE)", "H (MobileNet, batch16, Focal)"),
    ]
    
    for comp_name, ce_key, focal_key in comparisons:
        diffs = []
        for t in tasks:
            if ce_key in results and focal_key in results and t in results[ce_key] and t in results[focal_key]:
                acc_ce = results[ce_key][t].get("best_val_accuracy", 0.0)
                acc_focal = results[focal_key][t].get("best_val_accuracy", 0.0)
                diff = acc_focal - acc_ce
                sign = "+" if diff >= 0 else ""
                diffs.append(f"{sign}{diff:.1%}")
            else:
                diffs.append("N/A")
        print(f"| {comp_name} | " + " | ".join(diffs) + " |")

if __name__ == "__main__":
    main()
