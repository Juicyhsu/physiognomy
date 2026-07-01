"""
================================================
面相 AI 辨識系統 — FastAPI 後端服務 v3.0
================================================
支援 Claude / OpenAI / Gemini 三大 LLM 切換
用戶可輸入自己的 API Key，沒有的話自動 fallback 到開發者預設金鑰
"""

import os
import sys
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Optional

# ── Monkeypatch httpx for proxies compatibility with newer httpx 0.28+ ──
try:
    import httpx
    # Patch httpx.Client.__init__
    original_client_init = httpx.Client.__init__
    def patched_client_init(self, *args, **kwargs):
        if "proxies" in kwargs:
            kwargs.pop("proxies")
        original_client_init(self, *args, **kwargs)
    httpx.Client.__init__ = patched_client_init

    # Patch httpx.AsyncClient.__init__
    original_async_init = httpx.AsyncClient.__init__
    def patched_async_init(self, *args, **kwargs):
        if "proxies" in kwargs:
            kwargs.pop("proxies")
        original_async_init(self, *args, **kwargs)
    httpx.AsyncClient.__init__ = patched_async_init
except Exception as patch_err:
    print(f"[WARN] Failed to patch httpx: {patch_err}")

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# 確保能導入根目錄的腳本
sys.path.append(str(Path(__file__).parent.parent))

# ── 載入 .env 檔案（如果存在的話） ──
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        k = parts[0].strip()
                        v = parts[1].strip().strip('"').strip("'")
                        os.environ[k] = v
                except Exception as env_err:
                    print(f"[WARN] 讀取 .env 行失敗 [{line}]: {env_err}")

from step1_face_detection import analyze_face
from step2_physiognomy    import generate_report as generate_rule_report

# ── 開發者預設 API 金鑰（用戶沒有輸入時的 fallback）──
# 設定方式：可直接在專案根目錄建立 .env 檔案，內容為：
#   DEVELOPER_ANTHROPIC_KEY="sk-ant-..."
#   DEVELOPER_OPENAI_KEY="sk-..."
#   DEVELOPER_GEMINI_KEY="AIza..."
DEV_KEYS = {
    "claude": os.environ.get("DEVELOPER_ANTHROPIC_KEY", ""),
    "openai": os.environ.get("DEVELOPER_OPENAI_KEY", ""),
    "gemini": os.environ.get("DEVELOPER_GEMINI_KEY", ""),
}

app = FastAPI(
    title="面相 AI 辨識系統",
    description="結合 MediaPipe 關鍵點、MobileNetV2 CNN 深度學習與 Claude/OpenAI/Gemini LLM 的面相分析系統",
    version="3.0.0"
)

# ── 載入 CNN 模組 ──
CNN_READY = False
try:
    from step2c_cnn import predict_with_cnn, generate_cnn_report, load_cnn_models
    load_cnn_models()
    CNN_READY = True
    print("[OK] CNN 模型載入成功，Web App 已啟用深度學習路線！")
except Exception as e:
    print(f"[WARN] CNN 模型未載入，可能尚未訓練完畢或找不到檔案: {e}")
    print("   後端將自動以 Mock 測試數據替代，以便測試介面！")


from pydantic import BaseModel, Field
from typing import Dict
import json

class PalaceScores(BaseModel):
    天庭: int = Field(description="天庭（額頭/青年運）評分，0-100")
    監察: int = Field(description="監察（眉眼/心性）評分，0-100")
    財帛: int = Field(description="財帛（鼻子/中年財運）評分，0-100")
    出納: int = Field(description="出納（嘴巴/信用與表達）評分，0-100")
    地閣: int = Field(description="地閣（下巴臉型/晚年福澤）評分，0-100")

class PhysiognomyReportSchema(BaseModel):
    llm_report: str = Field(description="人臉面相完整鑑定書，包含五官解析與整體格局運勢總結，以美觀的 Markdown 格式輸出，字字見血、深刻且實用，字數務必達 800 字以上。")
    custom_lucky_hint: str = Field(description="高度客製化、非套套話的開運與修身建議，直接針對使用者的具體相法特質或缺陷（如嘴角下垂、鼻尖露骨、眼神疲態等）給出具體且實用的改運指引，字數約 80-120 字。")
    palace_scores: PalaceScores = Field(description="五宮專業面相評分")

import re
def clean_json_string(s: str) -> str:
    if not s:
        return s
    # 轉義任何不是合法 JSON 轉義序列的反斜線，防止 LLM 生成 markdown 或反斜線時破壞引號閉合
    pattern = r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})'
    return re.sub(pattern, r'\\\\', s)

def _dispatch_llm(provider: str, api_key: str, cnn_preds: dict, features: dict, style: str, image_path: Optional[str] = None) -> dict:
    """
    根據所選 LLM provider，分派到對應的 API 呼叫。
    回傳 dict: { "llm_report": str, "custom_lucky_hint": str, "palace_scores": dict, "model_name": str, "used_dev_key": bool }
    """
    # 決定最終使用的 key（優先用戶輸入，其次開發者預設）
    user_key = api_key.strip() if api_key else ""
    dev_key  = DEV_KEYS.get(provider, "")
    final_key = user_key or dev_key
    used_dev_key = (not user_key) and bool(dev_key)

    if not final_key:
        fallback_text = f"⚠️ 尚未設定 {provider.upper()} API Key。\n\n請在上方的「LLM 設定」區塊輸入您的個人 API Key，或聯繫系統管理員設定預設金鑰後重試！\n\n---\n\n**如需申請 API Key：**\n- Claude：[console.anthropic.com](https://console.anthropic.com)\n- OpenAI：[platform.openai.com](https://platform.openai.com/api-keys)\n- Gemini：[aistudio.google.com](https://aistudio.google.com/app/apikey)"
        return {
            "llm_report": fallback_text,
            "custom_lucky_hint": "請先設定您的 API 金鑰以進行深度面相鑑定。",
            "palace_scores": {"天庭": 60, "監察": 60, "財帛": 60, "出納": 60, "地閣": 60},
            "model_name": "未啟用",
            "used_dev_key": False,
        }

    from step2d_llm import _format_cnn_for_prompt, _format_geometry_for_prompt
    cnn_summary = _format_cnn_for_prompt(cnn_preds)
    geo_summary = _format_geometry_for_prompt(features)

    style_instruction = {
        "traditional": (
            "【古籍格局風格】：採用《麻衣相法》、《冰鑑》的古典命理術語與宮位框架"
            "（天庭、印堂、監察官、保壽官、審辨官/財帛宮、出納官、地閣等）逐一論斷。"
            "每一條推斷必須引用具體的面相學依據，例如「鼻準豐圓，財庫充盈」「眉壓眼，六親緣薄」等，"
            "嚴禁使用任何非命理專業的通用性讚美或批評，每句話必須直接對應當事人臉部的可觀察特徵。"
            "結尾需有完整的『格局總評』，指出此相的整體命格優劣與注意事項。"
        ),
        "modern": (
            "【性格天賦風格】：以中華傳統面相學為核心，將五官相法轉化為精準的個人特質分析。"
            "必須針對臉部的具體特徵（鼻型、眼形、眉距、嘴型、臉型、法令紋等）給出有命理依據的性格論斷，"
            "例如「眼距寬主性格包容、思路開闊」「嘴角下垂主行事謹慎保守、適合守成」，"
            "並說明其對事業方向、財富模式、人際格局的具體影響。"
            "嚴禁輸出『你是個很有潛力的人』等無針對性廢話，所有結論必須有面相學根據且個人化。"
        ),
        "playful": (
            "【流年運勢風格】：以三庭（天庭/上庭主早年運、中庭主中年運、下庭主晚年運）為核心框架，"
            "分析人生不同階段的運勢走向，並結合五官判斷財運、事業、婚姻與健康的具體趨勢，"
            "例如「天庭飽滿，三十歲前貴人相助，仕途平順」「人中短淺，晚年子女緣份待加強」。"
            "同時針對當事人臉部的具體缺陷或特徵提供日常改運建議（如漏財跡象如何防範、印堂暗沉如何化解），"
            "所有建議必須有命理學依據，嚴禁套用放諸四海皆準的通用祝福語或泛泛之詞。"
        ),
    }.get(style, "")


    system_prompt = """你是一位精通中華古典面相學與計算幾何學的 AI 命理大師。
你將結合 MediaPipe 的人臉幾何特徵數據、CNN 深度學習分類結果，並透過你的多模態視覺能力，對使用者的臉部相片進行極具專業感、富有命理學深度的「面相鑑定報告」。

【專業鑑定規則】：
1. 你的分析對象是一張真實的人臉圖片。請你發揮多模態視覺能力，仔細觀測這張圖片中的微觀特徵：
   - 【痣相】：尋找面部（如印堂、眼角、嘴角、鼻翼、臉頰、下巴等）是否有痣，並分析其精準位置對應的命理吉凶（如淚堂痣、食祿痣等）。若無明顯的痣，則不必憑空捏造，可說明「面部無明顯斑痣，主格局清純」。
   - 【法令紋】：觀測鼻翼兩側法令紋的深淺、長度、走向（如是否寬廣、是否雙線、是否入口等），解析其對應的事業與晚年運勢。
   - 【眼神與氣色】：觀測雙眼的神態（是否藏神、清澈或疲態）與面部膚色氣色（明潤或暗沉），進行氣色判定。
2. 結合 MediaPipe 的幾何數值（包括鼻孔外露度、嘴角上揚度、三白眼指數、印堂寬度、眉壓眼、人中長度比等）與 CNN 分類結果進行整體論證。
3. 嚴禁任何非命理專業的空泛玩笑或現代科技術語（如「大數據中的一股清流」、「命運的伺服器」等），也不要給出「你人很好，但有時候很孤單」等無針對性的星座套話。
4. 每個五官部位的解讀要互相呼應，構成一個整體格局。
5. 字數要求：請寫出一篇 800 字以上、架構清晰的完整繁體中文鑑定書，包含各部位解析、整體格局總結、以及「專業開運建議」（字數務必達 800 字以上）。
6. 格式與顏色著色要求：為了使報告中的重點易於閱讀，凡是提及關鍵命理宮位、部位或五官術語（例如：天庭、印堂、監察官、保壽官、審辨官/財帛宮、出納官、地閣、五宮評分、以及各部位的解讀結論如「挺鼻型」、「仰月口」等），務必使用 Markdown 粗體格式。必須【僅包裹單個特定術語或短語本身】（例如：**天庭**、**監察官（眼）**），嚴禁將整句文字、整段話、或多個不同的術語連同中間的文字包裹在同一對雙星號內（例如：必須分開寫成 **天庭** 與 **印堂**，絕對不可寫成 **天庭 ... 印堂** 這樣的一整段話）。"""

    user_prompt = (
        f"{system_prompt}\n\n"
        f"請根據以下幾何與深度學習數據，並親自觀測使用者相片，生成完整的面相分析報告：\n\n"
        f"## CNN 分類結果（模型信心度）\n{cnn_summary}\n\n"
        f"## 人臉幾何特徵（MediaPipe 測量值）\n{geo_summary}\n\n"
        f"## 風格要求\n{style_instruction}\n\n"
        f"請以 JSON 格式回傳，必須包含三個欄位：\n"
        f"1. 'llm_report': 各五官特質解讀與整體格局總結的完整 Markdown 鑑定書 (字數務必在 800 字以上，請完整生成且絕不要截斷)。\n"
        f"2. 'custom_lucky_hint': 高度客製化、字數約 80-120 字的開運與修身建議。\n"
        f"3. 'palace_scores': 五宮專業面相評分，值為 0 到 100 之間的整數。鍵值必須為：'天庭'、'監察'、'財帛'、'出納'、'地閣'。"
    )

    # ── Gemini 3.5 Flash ──
    if provider == "gemini":
        try:
            from google import genai as google_genai
            from PIL import Image
            client_gemini = google_genai.Client(api_key=final_key)
            
            contents = [user_prompt]
            if image_path and Path(image_path).exists():
                try:
                    img = Image.open(image_path)
                    contents = [img, user_prompt]
                except Exception as img_err:
                    print(f"多模態載入圖片失敗: {img_err}")

            from google.genai import types

            response = client_gemini.models.generate_content(
                model="gemini-3.5-flash",
                contents=contents,
                config=google_genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=PhysiognomyReportSchema,
                    max_output_tokens=8192,
                    temperature=0.75,
                    safety_settings=[
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                    ]
                )
            )
            
            clean_text = clean_json_string(response.text)
            res_data = json.loads(clean_text, strict=False)
            llm_report = (res_data.get("llm_report") or res_data.get("llmReport") or "").replace("\\*", "*")
            custom_lucky_hint = (res_data.get("custom_lucky_hint") or res_data.get("customLuckyHint") or "").replace("\\*", "*")
            palace_scores = res_data.get("palace_scores") or res_data.get("palaceScores") or {}
            
            return {
                "llm_report": llm_report,
                "custom_lucky_hint": custom_lucky_hint,
                "palace_scores": palace_scores,
                "model_name": "Gemini 3.5 Flash",
                "used_dev_key": used_dev_key
            }
        except ImportError:
            return {
                "llm_report": "❌ 缺少 `google-genai` 套件，請在 CMD 執行：`pip install -U google-genai`",
                "custom_lucky_hint": "環境缺失，請安裝 google-genai 套件。",
                "palace_scores": {"天庭": 50, "監察": 50, "財帛": 50, "出納": 50, "地閣": 50},
                "model_name": "錯誤",
                "used_dev_key": False
            }
        except Exception as e:
            traceback.print_exc()
            try:
                debug_path = Path(__file__).parent.parent / "gemini_debug.txt"
                with open(debug_path, "w", encoding="utf-8") as df:
                    raw_text = response.text if 'response' in locals() and hasattr(response, 'text') else 'No response text available'
                    df.write(f"Error: {e}\n\nRaw Response Text:\n{raw_text}\n")
            except Exception as debug_err:
                print(f"無法寫入 debug 檔案: {debug_err}")
            return {
                "llm_report": f"❌ Gemini API 呼叫失敗：{e}",
                "custom_lucky_hint": f"連線錯誤：{e}",
                "palace_scores": {"天庭": 50, "監察": 50, "財帛": 50, "出納": 50, "地閣": 50},
                "model_name": "錯誤",
                "used_dev_key": False
            }

    # ── Claude ──
    elif provider == "claude":
        try:
            import anthropic
            client_claude = anthropic.Anthropic(api_key=final_key)
            # 請求 Claude 以 JSON 格式回傳
            json_prompt = user_prompt + "\n\n請務必只回傳合法的 JSON 物件字串，不要包含任何開頭或結尾的說明文字（不要包含 ```json 標記）。"
            response = client_claude.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4096,
                messages=[{"role": "user", "content": json_prompt}],
                system=system_prompt,
                temperature=0.75
            )
            text_resp = response.content[0].text.strip()
            # 兼容有可能帶 ```json 的包裝
            if "```json" in text_resp:
                text_resp = text_resp.split("```json", 1)[-1].split("```", 1)[0].strip()
            elif "```" in text_resp:
                text_resp = text_resp.split("```", 1)[-1].split("```", 1)[0].strip()
            clean_text = clean_json_string(text_resp)
            res_data = json.loads(clean_text, strict=False)
            llm_report = (res_data.get("llm_report") or res_data.get("llmReport") or "").replace("\\*", "*")
            custom_lucky_hint = (res_data.get("custom_lucky_hint") or res_data.get("customLuckyHint") or "").replace("\\*", "*")
            palace_scores = res_data.get("palace_scores") or res_data.get("palaceScores") or {}
            return {
                "llm_report": llm_report,
                "custom_lucky_hint": custom_lucky_hint,
                "palace_scores": palace_scores,
                "model_name": "Claude 3.5 Sonnet",
                "used_dev_key": used_dev_key
            }
        except ImportError:
            return {
                "llm_report": "❌ 缺少 `anthropic` 套件，請在 CMD 執行：`pip install anthropic`",
                "custom_lucky_hint": "環境缺失，請安裝 anthropic 套件。",
                "palace_scores": {"天庭": 50, "監察": 50, "財帛": 50, "出納": 50, "地閣": 50},
                "model_name": "錯誤",
                "used_dev_key": False
            }
        except Exception as e:
            traceback.print_exc()
            return {
                "llm_report": f"❌ Claude API 呼叫失敗：{e}",
                "custom_lucky_hint": f"連線錯誤：{e}",
                "palace_scores": {"天庭": 50, "監察": 50, "財帛": 50, "出納": 50, "地閣": 50},
                "model_name": "錯誤",
                "used_dev_key": False
            }

    # ── OpenAI ──
    elif provider == "openai":
        try:
            import openai as openai_sdk
            client_openai = openai_sdk.OpenAI(api_key=final_key)
            json_prompt = user_prompt + "\n\n請務必以 JSON 格式回傳。"
            response = client_openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json_prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=4096,
                temperature=0.75
            )
            clean_text = clean_json_string(response.choices[0].message.content)
            res_data = json.loads(clean_text, strict=False)
            llm_report = (res_data.get("llm_report") or res_data.get("llmReport") or "").replace("\\*", "*")
            custom_lucky_hint = (res_data.get("custom_lucky_hint") or res_data.get("customLuckyHint") or "").replace("\\*", "*")
            palace_scores = res_data.get("palace_scores") or res_data.get("palaceScores") or {}
            return {
                "llm_report": llm_report,
                "custom_lucky_hint": custom_lucky_hint,
                "palace_scores": palace_scores,
                "model_name": "GPT-4o",
                "used_dev_key": used_dev_key
            }
        except ImportError:
            return {
                "llm_report": "❌ 缺少 `openai` 套件，請在 CMD 執行：`pip install openai`",
                "custom_lucky_hint": "環境缺失，請安裝 openai 套件。",
                "palace_scores": {"天庭": 50, "監察": 50, "財帛": 50, "出納": 50, "地閣": 50},
                "model_name": "錯誤",
                "used_dev_key": False
            }
        except Exception as e:
            traceback.print_exc()
            return {
                "llm_report": f"❌ OpenAI API 呼叫失敗：{e}",
                "custom_lucky_hint": f"連線錯誤：{e}",
                "palace_scores": {"天庭": 50, "監察": 50, "財帛": 50, "出納": 50, "地閣": 50},
                "model_name": "錯誤",
                "used_dev_key": False
            }

    return {
        "llm_report": "❌ 未知的 LLM Provider",
        "custom_lucky_hint": "未知的 LLM Provider",
        "palace_scores": {"天庭": 50, "監察": 50, "財帛": 50, "出納": 50, "地閣": 50},
        "model_name": "未知",
        "used_dev_key": False
    }


@app.get("/api/status")
async def get_status():
    """回傳後端服務狀態，讓前端可以顯示開發者金鑰是否已設定"""
    return {
        "cnn_ready": CNN_READY,
        "dev_keys_available": {
            "claude": bool(DEV_KEYS["claude"]),
            "openai": bool(DEV_KEYS["openai"]),
            "gemini": bool(DEV_KEYS["gemini"]),
        }
    }


@app.post("/api/analyze")
async def analyze(
    file:     UploadFile       = File(...),
    style:    str              = Form("traditional"),
    provider: str              = Form("gemini"),      # claude | openai | gemini
    api_key:  Optional[str]    = Form(None)
):
    """
    主要分析 API 路由：
    1. 接收圖片 + LLM 設定
    2. MediaPipe → 幾何特徵 + 標註圖
    3. 路線 A：傳統幾何規則引擎
    4. 路線 B：CNN 深度學習分類
    5. 路線 C：多 LLM 選擇器（Claude / OpenAI / Gemini）
    """
    temp_dir = tempfile.mkdtemp()
    try:
        # 保存上傳的檔案
        file_ext = Path(file.filename).suffix or ".jpg"
        img_path = Path(temp_dir) / f"input{file_ext}"
        with open(img_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Step 1: MediaPipe
        features, _ = analyze_face(str(img_path), output_dir=temp_dir)
        if features is None:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "❌ 無法偵測到清晰的人臉，請重新上傳正面清晰照片！"}
            )

        # Base64 標註圖
        import base64
        annotated_path = Path(temp_dir) / "annotated_face.jpg"
        annotated_base64 = ""
        if annotated_path.exists():
            with open(annotated_path, "rb") as image_file:
                annotated_base64 = base64.b64encode(image_file.read()).decode("utf-8")

        # Step 2: 路線 A
        rule_report = generate_rule_report(features)

        # Step 3: 路線 B (CNN)
        cnn_preds  = {}
        cnn_report = {}
        if CNN_READY:
            try:
                cnn_preds  = predict_with_cnn(str(img_path))
                cnn_report = generate_cnn_report(cnn_preds)
            except Exception as e:
                print(f"CNN 推理失敗: {e}")

        if not cnn_preds:
            cnn_preds = {
                "nose":  {"label": "挺鼻型",   "confidence": 0.88, "proba": {"挺鼻型": 0.88, "寬鼻型": 0.08, "標準鼻型": 0.04}},
                "eye":   {"label": "明亮眼型", "confidence": 0.79, "proba": {"明亮眼型": 0.79, "細眼型": 0.12, "疲態眼型": 0.09}},
                "brow":  {"label": "上揚眉",   "confidence": 0.72, "proba": {"上揚眉": 0.72, "濃眉": 0.18, "標準眉": 0.10}},
                "mouth": {"label": "笑容型",   "confidence": 0.85, "proba": {"豐唇型": 0.05, "笑容型": 0.85, "標準嘴型": 0.10}},
                "face":  {"label": "瓜子臉",   "confidence": 0.81, "proba": {"瓜子臉": 0.81, "圓潤臉": 0.05, "稜角臉": 0.04, "標準臉型": 0.10}},
            }
            cnn_report = {"avg_score": 0.86, "summary": "CNN 深度學習模擬分析完成。最突出特質：「挺鼻型」（信心度 88%）。整體面相評分 86%。"}

        # 注入 Ensemble Metadata
        ENSEMBLE_METADATA = {
            "eye":   {"best_exp": "Ensemble-A (眼部特化模型)", "val_acc": "71.7%"},
            "nose":  {"best_exp": "Ensemble-D (鼻型細化模型)", "val_acc": "55.0%"},
            "brow":  {"best_exp": "Ensemble-D (眉部細化模型)", "val_acc": "63.3%"},
            "mouth": {"best_exp": "Ensemble-A (嘴型優化模型)", "val_acc": "68.2%"},
            "face":  {"best_exp": "Ensemble-C (臉型微調模型)", "val_acc": "53.2%"},
        }
        for k, pred in cnn_preds.items():
            if k in ENSEMBLE_METADATA:
                pred["best_exp"] = ENSEMBLE_METADATA[k]["best_exp"]
                pred["val_acc"]  = ENSEMBLE_METADATA[k]["val_acc"]

        # Step 4: 路線 C (多 LLM 分派)
        llm_style = style if style in ["traditional", "modern", "playful"] else "traditional"
        llm_result = _dispatch_llm(
            provider=provider.lower(),
            api_key=api_key or "",
            cnn_preds=cnn_preds,
            features=features,
            style=llm_style,
            image_path=str(img_path)
        )

        # 覆蓋幾何規則引擎的隨機開運建議，使用 LLM 個性化客製的開運建議
        rule_report_dict = {
            "face_type": rule_report.face_type,
            "summary": rule_report.summary,
            "lucky_hint": llm_result.get("custom_lucky_hint") or rule_report.lucky_hint
        }

        return {
            "success":        True,
            "annotated_image": f"data:image/jpeg;base64,{annotated_base64}",
            "features":       features,
            "rule_report":    rule_report_dict,
            "cnn_preds":      cnn_preds,
            "cnn_report":     cnn_report,
            "llm_report":     llm_result["llm_report"],
            "llm_model_name": llm_result["model_name"],
            "used_dev_key":   llm_result["used_dev_key"],
            "is_real_cnn":    CNN_READY,
            "palace_scores":  llm_result.get("palace_scores", {
                "天庭": 75, "監察": 75, "財帛": 75, "出納": 75, "地閣": 75
            })
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"伺服器內部錯誤: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# 掛載靜態網頁
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
