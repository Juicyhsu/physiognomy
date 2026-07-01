import re
import sys
from pathlib import Path

# 強制將 stdout 設置為 utf-8 以防 emoji 輸出失敗
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def compress_file(input_path: Path, output_path: Path):
    print(f"Reading and compressing: {input_path.name} ({input_path.stat().st_size / 1024 / 1024:.2f} MB)...")
    
    epoch_pat = re.compile(r"^\s*Epoch\s+\d+/\d+")
    metric_pat = re.compile(r"loss:.*accuracy:")
    skip_pat = re.compile(r"[\r\x08]|(^\s*\d+/\d+\s+\[=+\].*)|(^\s*\d+/\d+\s+\[\.+\].*)")

    important_lines = []
    skipped_count = 0

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                important_lines.append(line)
                continue
                
            if skip_pat.search(line):
                skipped_count += 1
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
                important_lines.append(clean_line)
            else:
                skipped_count += 1

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(important_lines)

    print(f"Success! Compressed file saved to: {output_path.name} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"Filtered out {skipped_count} lines of progress bars.")

if __name__ == "__main__":
    # 使用當前工作目錄的相對路徑以防編碼問題
    log_dir = Path("訓練過程紀錄")
    compress_file(log_dir / "訓練過程紀錄3.txt", log_dir / "訓練過程紀錄3_compressed.txt")
