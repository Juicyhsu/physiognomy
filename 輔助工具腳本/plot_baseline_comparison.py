import os
import sys
from pathlib import Path

# 強制將 stdout 設置為 utf-8 以防 cp950 編碼錯誤
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 設定非 GUI 後端以防止 matplotlib 在沒有顯示器的環境下出錯
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# ── 🔍 自動偵測與載入微軟正黑體 (相容 Windows 與 Linux/WSL) ──
font_path = None
possible_paths = [
    "C:\\Windows\\Fonts\\msjh.ttc",
    "C:\\Windows\\Fonts\\msjh.ttf",
    "C:\\Windows\\Fonts\\msjhbd.ttc",
    "/mnt/c/Windows/Fonts/msjh.ttc",
    "/mnt/c/Windows/Fonts/msjh.ttf",
    "/mnt/c/Windows/Fonts/msjhbd.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]

for p in possible_paths:
    if os.path.exists(p):
        font_path = p
        break

# 初始化 FontProperties
if font_path:
    my_font = fm.FontProperties(fname=font_path)
    print(f"成功取得中文字型檔，將使用極致相容模式渲染: {font_path}")
else:
    # 備用設定
    my_font = fm.FontProperties(family='sans-serif')
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans', 'sans-serif']

def plot_baseline_comparison():
    tasks = ["眼型", "鼻型", "眉型", "嘴型", "臉型"]
    
    # 基準數據 (捨棄隨機猜測，只保留多數類基準與最佳集成模型)
    majority_class = [0.705, 0.554, 0.628, 0.393, 0.358]
    new_ensemble = [0.7280, 0.5840, 0.6440, 0.6822, 0.5322] # 採用最佳集成結果 (A-H 最優選)

    x = np.arange(len(tasks))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    plt.style.use('seaborn-v0_8-whitegrid') if 'seaborn-v0_8-whitegrid' in plt.style.available else plt.grid(True, linestyle='--', alpha=0.5)

    # 繪製對比柱狀圖 (金黃色代表懶人基準，深紅色代表最佳集成模型)
    rects1 = ax.bar(x - width/2, [v*100 for v in majority_class], width, 
                    label="多數類別基準 (懶人模型)", color="#fe9929", edgecolor='black', linewidth=0.5)
    rects2 = ax.bar(x + width/2, [v*100 for v in new_ensemble], width, 
                    label="本系統最佳集成 CNN 模型", color="#d7301f", edgecolor='black', linewidth=0.5)

    # 設定中文字型與標題，使用 fontproperties 參數進行強制渲染，避開 rcParams 的快取 bug
    ax.set_title("各五官部位 CNN 預測準確率與多數類別基準對照圖", fontproperties=my_font, fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    
    # 手動給 xticks 設置中文字型
    ax.set_xticklabels(tasks, fontproperties=my_font, fontsize=11, fontweight='bold')
    ax.set_ylabel("驗證集準確率 (%)", fontproperties=my_font, fontsize=11)
    ax.set_ylim(0, 95)
    
    # 設定 Legend 字型
    legend = ax.legend(loc="upper right", frameon=True, facecolor='white', edgecolor='lightgray')
    for text in legend.get_texts():
        text.set_fontproperties(my_font)
        text.set_fontsize(10)
    
    # 標註柱子上的數值
    def autolabel(rects, is_bold=False):
        for rect in rects:
            height = rect.get_height()
            weight = 'bold' if is_bold else 'normal'
            # 設定數值標註字型
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight=weight, fontproperties=my_font)

    autolabel(rects1)
    autolabel(rects2, is_bold=True)

    plt.tight_layout()
    output_path = "報告圖表/03_引入FocalLoss對照/comparison_with_baselines.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"學術對照圖已成功輸出至: {output_path}")

if __name__ == "__main__":
    plot_baseline_comparison()
