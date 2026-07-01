import re
import sys
import os
from pathlib import Path

# 強制將 stdout 設置為 utf-8 以防 cp950 編碼錯誤
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def get_compressed_lines(file_path: Path):
    epoch_pat = re.compile(r"^\s*Epoch\s+\d+/\d+")
    metric_pat = re.compile(r"loss:.*accuracy:")
    skip_pat = re.compile(r"[\r\x08]|(^\s*\d+/\d+\s+\[=+\].*)|(^\s*\d+/\d+\s+\[\.+\].*)")

    compressed_lines = []
    
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                compressed_lines.append(line)
                continue
                
            if skip_pat.search(line):
                continue
                
            is_important = (
                epoch_pat.match(stripped) or
                metric_pat.search(stripped) or
                "GPU" in stripped or
                "TensorFlow" in stripped or
                "Device" in stripped or
                "實驗" in stripped or
                "訓練" in stripped or
                "完成" in stripped or
                "耗時" in stripped or
                "EarlyStopping" in stripped or
                "ReduceLROnPlateau" in stripped or
                "Killed" in stripped or
                "---" in stripped or
                "===" in stripped or
                "⏱" in stripped or
                "🎉" in stripped or
                "📂" in stripped or
                "✅" in stripped or
                "❌" in stripped
            )
            
            if is_important:
                clean_line = line.replace('\r', '').replace('\x08', '')
                compressed_lines.append(clean_line)
                
    return compressed_lines

def main():
    log_dir = Path("訓練過程紀錄")
    log3_path = log_dir / "訓練過程紀錄3.txt"
    log4_path = log_dir / "訓練過程紀錄4.txt"
    output_path = log_dir / "訓練過程紀錄_綜合壓縮版.txt"

    print("開始進行日誌綜合與壓縮...")

    lines_log3 = []
    if log3_path.exists():
        print(f"1. 讀取並過濾 Log 3 ({log3_path.stat().st_size / 1024 / 1024:.2f} MB)...")
        lines_log3 = get_compressed_lines(log3_path)
    else:
        print("警告: 訓練過程紀錄3.txt 不存在")

    lines_log4 = []
    if log4_path.exists():
        print(f"2. 讀取並過濾 Log 4 ({log4_path.stat().st_size / 1024 / 1024:.2f} MB)...")
        lines_log4 = get_compressed_lines(log4_path)
    else:
        print("警告: 訓練過程紀錄4.txt 不存在")

    # 綜合寫入
    all_lines = []
    all_lines.append("============================================================\n")
    all_lines.append("  🔍 訓練過程紀錄 - 綜合壓縮精簡版 (整合 Log 3 & Log 4)\n")
    all_lines.append("============================================================\n\n")
    
    if lines_log3:
        all_lines.append("=== 【前半段訓練歷程 (實驗 A~G, E~G完成, H被Killed)】 ===\n")
        all_lines.extend(lines_log3)
        all_lines.append("\n\n")

    if lines_log4:
        all_lines.append("=== 【後半段重啟歷程 (實驗 H完成，包含最終總結)】 ===\n")
        all_lines.extend(lines_log4)

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(all_lines)

    print(f"🎉 綜合壓縮成功！已寫入: {output_path.name} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")

    # 刪除原始巨大檔案與臨時壓縮檔
    deleted_size = 0.0
    for file_to_del in [log3_path, log4_path, log_dir / "訓練過程紀錄3_compressed.txt"]:
        if file_to_del.exists():
            size = file_to_del.stat().st_size
            os.remove(file_to_del)
            deleted_size += size
            print(f"🗑️ 已刪除原始巨大檔: {file_to_del.name} ({size / 1024 / 1024:.2f} MB)")
            
    print(f"✨ 硬碟空間共釋放了: {deleted_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()
