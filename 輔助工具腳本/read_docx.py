import zipfile
import re
import sys
from pathlib import Path

def extract_docx_text(docx_path, txt_path):
    if not os.path.exists(docx_path):
        print(f"Error: {docx_path} does not exist.")
        return
        
    with zipfile.ZipFile(docx_path) as z:
        doc_xml = z.read('word/document.xml').decode('utf-8')
        # 移除 XML 標籤取得純文字
        text = re.sub(r'<[^>]+>', '', doc_xml)
        
        # 簡單的分段處理
        text = text.replace('</w:p>', '\n').replace('</w:r>', '')
        
        # 清理多餘空行
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            
    print(f"Successfully extracted text to: {txt_path}")

if __name__ == "__main__":
    import os
    docx_file = r"報告圖表/03_引入FocalLoss對照/乾坤AI_完整實驗報告_8組含FocalLoss.docx"
    txt_file = r"輔助工具腳本/report_text_focal.txt"
    extract_docx_text(docx_file, txt_file)

