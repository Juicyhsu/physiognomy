import os
import shutil
from pathlib import Path

# 定義最佳模型來源 (已整合新版 Focal Loss 優化模型)
BEST_MODELS = {
    "nose": "models_cnn_expG_mobilenet_lowLR_focal",
    "eye": "models_cnn_expH_mobilenet_batch16_focal",
    "brow": "models_cnn_expH_mobilenet_batch16_focal",
    "mouth": "models_cnn_expA_mobilenet",
    "face": "models_cnn_expC_mobilenet_lowLR",
}

def setup_ensemble():
    target_dir = Path("models_cnn")
    target_dir.mkdir(exist_ok=True)
    
    print("🚀 開始建立最強『明星隊』(Dream Team) 模型組合目錄...")
    
    for part, source_dir_name in BEST_MODELS.items():
        source_dir = Path(source_dir_name)
        print(f"\n🔹 處理部位 [{part.upper()}] (來源: {source_dir_name})")
        
        # 複製對應部位的所有檔案：h5, config.json, history.json, curve.png
        copied_any = False
        for ext in ["_best.h5", "_config.json", "_history.json", "_curve.png"]:
            src_file = source_dir / f"{part}{ext}"
            dst_file = target_dir / f"{part}{ext}"
            
            if src_file.exists():
                shutil.copy2(src_file, dst_file)
                print(f"  ✅ 複製: {src_file.name} -> models_cnn/{dst_file.name}")
                copied_any = True
            else:
                if ext != "_curve.png" and ext != "_history.json":
                    print(f"  ❌ 警告: 找不到關鍵檔案 {src_file}")
        
    print("\n🎉 明星隊模型組合複製完成！現在 `models_cnn/` 已經準備就緒。")

if __name__ == "__main__":
    setup_ensemble()
