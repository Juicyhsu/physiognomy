"""
================================================
面相引擎 升級版 — 接 LLM API 動態解讀
================================================
把 CNN 輸出的五官分類結果 + 幾何特徵值
傳給 Claude API，讓 LLM 生成個性化面相解讀

優點（比查表好的原因）：
  1. 根據「這個人」的具體數值動態生成，不是套公版
  2. 語言自然流暢，不像機器填空
  3. 可以綜合多個特徵做整體判斷（鼻寬但眼大 → 怎麼說）
  4. 報告裡可以說「使用 LLM 輔助生成自然語言報告」

執行需求：
  Claude API Key 已設定在環境變數 ANTHROPIC_API_KEY
  或直接在 Gradio App 裡的瀏覽器端呼叫（見 step3_app_v3.py）
"""

import os
import json
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

# ── Anthropic SDK ──
# pip install anthropic
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("提示：pip install anthropic 以啟用 LLM 面相解讀")


# =====================================================================
# 💡 [備用選項 A：接 OpenAI API 的完整用法 (註解備用)]
# =====================================================================
# 步驟 1: 在終端機執行：pip install openai
# 步驟 2: 以下是完整程式碼範例，要啟用時只需取消註解並在 main.py 或 Gradio 中調用：
#
# import openai
# def generate_openai_physiognomy(
#     cnn_predictions: dict,
#     features: dict,
#     style: str = "traditional",
#     api_key: Optional[str] = None,
# ) -> str:
#     key = api_key or os.environ.get("OPENAI_API_KEY", "")
#     if not key:
#         return "⚠️ 未設定 OPENAI_API_KEY，無法使用 OpenAI 解析。"
#
#     client = openai.OpenAI(api_key=key)
#     cnn_summary = _format_cnn_for_prompt(cnn_predictions)
#     geo_summary = _format_geometry_for_prompt(features)
#     style_instruction = {
#         "traditional": "請使用傳統中華面相學的語言風格，引用「三庭五眼」、「財帛宮」等專業術語，語氣莊重典雅。",
#         "modern":      "請使用現代白話文，結合心理學與個性分析 of 語言，讓年輕人容易理解，不要太玄學。",
#         "playful":     "請使用輕鬆有趣的語氣，加入生活化比喻，像朋友在幫你看面相一樣，可以適度幽默。",
#     }.get(style, "")
#
#     system_prompt = "你是一位精通中華面相學的 AI 分析師。你會根據深度學習模型提供的五官分類結果與人臉幾何特徵，生成個性化的面相解讀報告。正面積極為主，但要真實，不要過度美化。每段解讀 2~3 句，最後給一段 50 字以內的整體總結。輸出用 Markdown 格式。"
#     user_prompt = f"請根據以下深度學習分析結果，生成面相解讀報告：\n\n## CNN 分類結果\n{cnn_summary}\n\n## 幾何特徵\n{geo_summary}\n\n## 風格要求\n{style_instruction}\n\n請生成完整的面相分析報告，包含各五官特質解讀、整體格局總結與一句開運建議。"
#
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o",  # 亦可選用 gpt-4o-mini 降低成本
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt}
#             ],
#             max_tokens=1000,
#             temperature=0.7,
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         return f"❌ OpenAI API 呼叫失敗: {e}"


# =====================================================================
# 💡 [備用選項 B：接 Google Gemini API 的完整用法 (註解備用)]
# =====================================================================
# 步驟 1: 在終端機執行：pip install google-genai
# 步驟 2: 以下是完整程式碼範例，要啟用時只需取消註解並在 main.py 或 Gradio 中調用：
#
# from google import genai as google_genai
# def generate_gemini_physiognomy(
#     cnn_predictions: dict,
#     features: dict,
#     style: str = "traditional",
#     api_key: Optional[str] = None,
# ) -> str:
#     key = api_key or os.environ.get("GEMINI_API_KEY", "")
#     if not key:
#         return "⚠️ 未設定 GEMINI_API_KEY，無法使用 Gemini 解析。"
#
#     # 使用最新的 Google GenAI SDK (CP 值最高的 2026 穩定首選模型：gemini-3.5-flash)
#     client = google_genai.Client(api_key=key)
#     cnn_summary = _format_cnn_for_prompt(cnn_predictions)
#     geo_summary = _format_geometry_for_prompt(features)
#     style_instruction = {
#         "traditional": "請使用傳統中華面相學的語言風格，引用「三庭五眼」、「財帛宮」等專業術語，語氣莊重典雅。",
#         "modern":      "請使用現代白話文，結合心理學與個性分析的語言，讓年輕人容易理解，不要太玄學。",
#         "playful":     "請使用輕鬆有趣的語氣，加入生活化比喻，像朋友在幫你看面相一樣，可以適度幽默。",
#     }.get(style, "")
#
#     system_prompt = "你是一位精通中華面相學的 AI 分析師。你會根據深度學習模型提供的五官分類結果與人臉幾何特徵，生成個性化的面相解讀報告。正面積極為主，但要真實，不要過度美化。每段解讀 2~3 句，最後給一段 50 字以內的整體總結。輸出用 Markdown 格式。"
#     user_prompt = (
#         f"{system_prompt}\n\n"
#         f"請根據以下深度學習分析結果，生成面相解讀報告：\n\n"
#         f"## CNN 分類結果\n{cnn_summary}\n\n"
#         f"## 幾何特徵\n{geo_summary}\n\n"
#         f"## 風格要求\n{style_instruction}\n\n"
#         f"請生成完整的面相分析報告，包含各五官特質解讀、整體格局總結與一句開運建議。"
#     )
#
#     try:
#         # gemini-3.5-flash 是當前最推薦的免費額度高速度高 CP 值穩定模型
#         response = client.models.generate_content(
#             model="gemini-3.5-flash",
#             contents=user_prompt,
#             config=google_genai.types.GenerateContentConfig(
#                 max_output_tokens=1000,
#                 temperature=0.7,
#             )
#         )
#         return response.text
#     except Exception as e:
#         return f"❌ Gemini API 呼叫失敗: {e}"


def generate_llm_physiognomy(
    cnn_predictions: dict,
    features: dict,
    style: str = "traditional",   # "traditional" | "modern" | "playful"
    api_key: Optional[str] = None,
) -> str:
    """
    用 Gemini API 生成面相解讀報告

    Args:
        cnn_predictions: CNN 分類結果
        features: MediaPipe 幾何特徵值 dict
        style: 解讀風格
        api_key: Gemini API Key（None 則從環境變數讀）

    Returns:
        面相解讀報告字串（Markdown 格式）
    """
    key = api_key or os.environ.get("DEVELOPER_GEMINI_KEY") or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return "⚠️ 未設定 DEVELOPER_GEMINI_KEY，使用備用規則引擎。\n\n" + _fallback_report(cnn_predictions)

    try:
        from google import genai as google_genai
    except ImportError:
        return "❌ 缺少 `google-genai` 套件，請在 CMD 執行：`pip install -U google-genai`"

    # ── 整理輸入給 LLM 的資料 ──
    cnn_summary = _format_cnn_for_prompt(cnn_predictions)
    geo_summary = _format_geometry_for_prompt(features)
    style_instruction = {
        "traditional": "【古籍實用白話風格】：請使用流暢易懂的現代白話文，結合中華傳統面相學大師的專業厚重感，深入解析五官部位在《麻衣相法》或《冰鑑》中的古典命理（如天庭、印堂、監察官、保壽官、審辨官/財帛宮、出納官/嘴相、地閣）吉凶與格局，用詞務必精準且充滿命理學深度。",
        "modern":      "【現代性格天賦風格】：請將中華傳統面相學的道理與現代性格分析相結合。著重解讀五官幾何型態與個人性格特質、事業行為模式、潛在天賦及職涯發展的關係，不落入空泛星座套話，字字見血、客觀專業。",
        "playful":     "【流年運勢與修身防範風格】：請著重分析面相在三庭所對應的流年運勢（天庭主早年、中庭主中年、下庭主晚年），並給出具體的日常修身建議與運勢防範（例如如何防止漏財、避開人際口舌小人等），內容必須切合面相學理且實用。"
    }.get(style, "")

    system_prompt = """你是一位精通中華古典面相學與計算幾何學的 AI 命理大師。
你將結合 MediaPipe 的人臉幾何特徵數據、CNN 深度學習分類結果，對使用者的臉部進行極具專業感、富有命理學深度的「面相鑑定報告」。

【專業鑑定規則】：
1. 你的分析應將 MediaPipe 的幾何數值（包括鼻孔外露度、嘴角上揚度、三白眼指數、印堂寬度、眉壓眼、人中長度比等）與 CNN 分類結果進行整體論證，具體映射到面相學部位。
2. 嚴禁任何非命理專業的空泛玩笑或現代科技術語（如「大數據中的一股清流」、「命運的伺服器」等），也不要給出「你人很好，但有時候很孤單」等無針對性的星座套話。
3. 每個部位的解讀要互相呼應，構成一個整體格局。
4. 輸出用繁體中文。請寫出一篇 800 字以上、架構清晰的完整鑑定書，包含各部位解析、整體格局總結、以及一句點睛的「專業開運與修身建議」。
5. 格式與顏色著色要求：為了使報告中的重點易於閱讀，凡是提及關鍵命理宮位、部位或五官術語（例如：天庭、印堂、監察官、保壽官、審辨官/財帛宮、出納官、地閣、五宮評分、以及各部位的解讀結論如「挺鼻型」、「仰月口」等），務必使用 Markdown 粗體格式。必須【僅包裹單個特定術語或短語本身】（例如：**天庭**、**監察官（眼）**），嚴禁將整句文字、整段話、或多個不同的術語連同中間的文字包裹在同一對雙星號內（例如：必須分開寫成 **天庭** 與 **印堂**，絕對不可寫成 **天庭 ... 印堂** 這樣的一整段話）。"""

    user_prompt = f"""請根據以下數據，生成面相解讀報告：

## CNN 分類結果（模型信心度）
{cnn_summary}

## 人臉幾何特徵（MediaPipe 測量值）
{geo_summary}

## 風格要求
{style_instruction}

請以美觀的 Markdown 格式輸出，必須包含：
1. 各五官特質解讀（結合幾何數值與古典面相部位分析）
2. 整體格局與運勢總結
3. 專業開運與修身建議（字數務必達 800 字以上，請生成完整報告不要截斷）"""

    try:
        client = google_genai.Client(api_key=key)
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=user_prompt,
            config=google_genai.types.GenerateContentConfig(
                max_output_tokens=4096,
                temperature=0.75,
            )
        )
        return response.text.replace("\\*", "*")
    except Exception as e:
        print(f"API 呼叫失敗: {e}")
        return _fallback_report(cnn_predictions)


def _format_cnn_for_prompt(predictions: dict) -> str:
    """整理 CNN 結果成可讀文字給 LLM"""
    target_names = {
        "nose":  "鼻型",
        "eye":   "眼型",
        "brow":  "眉型",
        "mouth": "嘴型",
        "face":  "臉型",
    }
    lines = []
    for key, info in predictions.items():
        name = target_names.get(key, key)
        label = info.get("label", "未知")
        conf  = info.get("confidence", 0)
        conf_desc = "高度確定" if conf > 0.8 else "中度確定" if conf > 0.6 else "略有不確定"
        lines.append(f"- {name}：{label}（信心度 {conf:.0%}，{conf_desc}）")
    return "\n".join(lines)


def _format_geometry_for_prompt(features: dict) -> str:
    """整理幾何特徵成可讀文字給 LLM"""
    if not features:
        return "（無幾何特徵資料）"

    interpretations = []

    # 臉型比例
    r = features.get("face_ratio", 0)
    interpretations.append(
        f"- 臉寬/臉高比例 {r:.2f}（{'偏圓' if r > 0.8 else '偏長' if r < 0.65 else '適中'}）"
    )

    # 三庭均衡
    bal = features.get("san_ting_balance", 0)
    interpretations.append(
        f"- 三庭均衡度 {bal:.2f}（{'非常均衡' if bal > 0.85 else '略有偏差' if bal > 0.7 else '明顯偏重某庭'}）"
    )

    # 眼睛開度
    eye = (features.get("left_eye_ratio", 0) + features.get("right_eye_ratio", 0)) / 2
    interpretations.append(
        f"- 眼睛開闊度 {eye:.2f}（{'大眼明亮' if eye > 0.32 else '適中' if eye > 0.22 else '細長深邃'}）"
    )

    # 鼻寬
    nose = features.get("nose_width_ratio", 0)
    interpretations.append(
        f"- 鼻寬比例 {nose:.2f}（{'鼻翼寬廣' if nose > 0.35 else '鼻型標準' if nose > 0.28 else '鼻梁挺直'}）"
    )

    # 鼻孔外露度
    nose_exp = features.get("nostril_exposure_ratio", 0)
    interpretations.append(
        f"- 鼻孔外露度 {nose_exp:.3f}（{'鼻尖上翹/鼻孔外露/露財' if nose_exp > 0.02 else '鼻尖下垂/鷹鉤鼻/收斂' if nose_exp < -0.02 else '鼻型端正/不露鼻孔'}）"
    )

    # 嘴巴
    mouth = features.get("mouth_width_ratio", 0)
    interpretations.append(
        f"- 嘴巴寬度比例 {mouth:.2f}（{'嘴型寬大' if mouth > 0.42 else '嘴型適中' if mouth > 0.33 else '嘴型小巧'}）"
    )

    # 嘴角上揚度
    mouth_up = features.get("mouth_corner_upward_ratio", 0)
    interpretations.append(
        f"- 嘴角上揚度 {mouth_up:.3f}（{'嘴角上揚/仰月口/樂觀聚財' if mouth_up > 0.05 else '嘴角下垂/覆舟口/易漏財招非' if mouth_up < -0.05 else '嘴角平直/標準嘴型'}）"
    )

    # 三白眼指數
    iris_pos = features.get("iris_vertical_position", 0.5)
    interpretations.append(
        f"- 虹膜垂直偏離度 {iris_pos:.3f}（{'下三白眼' if iris_pos > 0.56 else '上三白眼' if iris_pos < 0.40 else '正常眼相'}）"
    )

    # 印堂寬度
    glabella = features.get("glabella_width_ratio", 0)
    if glabella > 0:
        interpretations.append(
            f"- 印堂寬度比 {glabella:.2f}（{'印堂開闊/心胸寬廣/貴人運佳' if glabella > 0.18 else '印堂偏窄/心思細密/易焦慮' if glabella < 0.12 else '印堂適中/符合禮法'}）"
        )

    # 眉壓眼垂直距離
    brow_eye = features.get("brow_eye_distance", 0)
    if brow_eye > 0:
        interpretations.append(
            f"- 眉壓眼指數 {brow_eye:.3f}（{'眉眼開闊/性情溫和/貴人扶持' if brow_eye > 0.08 else '眉壓眼/個性急躁/青年運受阻' if brow_eye < 0.055 else '眉眼適中/行事穩健'}）"
        )

    # 人中長度比例
    philtrum = features.get("philtrum_ratio", 0)
    if philtrum > 0:
        interpretations.append(
            f"- 人中長度比 {philtrum:.3f}（{'人中深長/壽元綿長/沉穩有耐力' if philtrum > 0.04 else '人中偏短/行事迅速/性急' if philtrum < 0.028 else '人中適中/符合法度'}）"
        )

    return "\n".join(interpretations)


def _fallback_report(predictions: dict) -> str:
    """API 不可用時的備用報告"""
    lines = ["## 面相分析報告（規則引擎版）\n"]
    for key, info in predictions.items():
        label = info.get("label", "未知")
        conf  = info.get("confidence", 0)
        lines.append(f"**{label}** （信心度 {conf:.0%}）")
    lines.append("\n*提示：設定 ANTHROPIC_API_KEY 可獲得 LLM 個性化解讀*")
    return "\n".join(lines)


# ═══════════════════════════════════════
# Gradio App 整合版（含 LLM 解讀）
# ═══════════════════════════════════════

def build_full_app():
    """
    完整 Gradio App：
    偵測 → CNN → LLM 面相解讀
    三種解讀風格可切換
    """
    import gradio as gr
    import cv2
    import tempfile
    import numpy as np
    from pathlib import Path

    from step1_face_detection import analyze_face
    from step2c_cnn import predict_with_cnn, generate_cnn_report, load_cnn_models

    # 嘗試載入 CNN 模型
    try:
        load_cnn_models()
        CNN_READY = True
    except FileNotFoundError:
        CNN_READY = False

    def process(image: np.ndarray, style: str, api_key: str):
        if image is None:
            return None, "請上傳圖片", "請上傳圖片"

        with tempfile.TemporaryDirectory() as tmp:
            img_path = str(Path(tmp) / "input.jpg")
            cv2.imwrite(img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            # Step 1: MediaPipe
            features, _ = analyze_face(img_path, output_dir=tmp)
            if features is None:
                return image, "❌ 未偵測到人臉", "❌ 未偵測到人臉"

            import cv2 as _cv2
            ann = _cv2.imread(str(Path(tmp) / "annotated_face.jpg"))
            annotated = _cv2.cvtColor(ann, _cv2.COLOR_BGR2RGB)

        # Step 2: CNN 分類
        if CNN_READY:
            cnn_preds = predict_with_cnn(img_path)
        else:
            # 示範用 mock 資料
            cnn_preds = {
                "nose":  {"label": "挺鼻型",  "confidence": 0.82},
                "eye":   {"label": "明亮眼型","confidence": 0.75},
                "brow":  {"label": "標準眉",  "confidence": 0.68},
                "mouth": {"label": "標準嘴型","confidence": 0.71},
                "face":  {"label": "橢圓臉",  "confidence": 0.79},
            }

        # CNN 結構化報告
        cnn_report = generate_cnn_report(cnn_preds) if CNN_READY else None
        cnn_text = _format_cnn_report_md(cnn_preds)

        # Step 3: LLM 面相解讀
        style_map = {"傳統典雅": "traditional", "現代白話": "modern", "輕鬆有趣": "playful"}
        llm_text = generate_llm_physiognomy(
            cnn_predictions=cnn_preds,
            features=features,
            style=style_map.get(style, "traditional"),
            api_key=api_key.strip() if api_key else None,
        )

        return annotated, cnn_text, llm_text

    def _format_cnn_report_md(preds):
        icons = {"nose":"👃","eye":"👁️","brow":"〰️","mouth":"👄","face":"🔷"}
        names = {"nose":"鼻型","eye":"眼型","brow":"眉型","mouth":"嘴型","face":"臉型"}
        lines = ["## 🤖 CNN 分類結果\n"]
        for k, v in preds.items():
            bar = "█" * round(v["confidence"]*10) + "░"*(10-round(v["confidence"]*10))
            lines.append(f"**{icons.get(k,'')} {names.get(k,k)}：{v['label']}**")
            lines.append(f"`{bar}` {v['confidence']:.0%}\n")
        return "\n".join(lines)

    with gr.Blocks(
        title="面相 AI — LLM 解讀版",
        theme=gr.themes.Soft(primary_hue="orange", neutral_hue="stone"),
    ) as demo:
        gr.HTML("""
        <div style="text-align:center;padding:20px 0 10px">
          <h1 style="font-size:1.8em;font-weight:400;letter-spacing:3px;margin:0">
            ✦ 面相 AI 辨識系統 ✦
          </h1>
          <p style="opacity:0.5;font-size:0.85em;margin:6px 0 0">
            MediaPipe × MobileNetV2 × Claude LLM · 期末專題
          </p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                img_in  = gr.Image(label="上傳正面人臉", type="numpy", height=280)
                style   = gr.Radio(
                    ["傳統典雅", "現代白話", "輕鬆有趣"],
                    value="傳統典雅",
                    label="面相解讀風格",
                )
                api_key_in = gr.Textbox(
                    label="Anthropic API Key（選填）",
                    placeholder="sk-ant-... （留空使用備用規則引擎）",
                    type="password",
                )
                btn = gr.Button("🔍 開始分析", variant="primary", size="lg")

            with gr.Column(scale=1):
                img_out = gr.Image(label="五官偵測結果", type="numpy", height=280)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🤖 CNN 分類結果")
                cnn_out = gr.Markdown("*上傳圖片後顯示...*")
            with gr.Column():
                gr.Markdown("### ✨ LLM 面相解讀")
                llm_out = gr.Markdown("*上傳圖片後顯示...*")

        btn.click(
            fn=process,
            inputs=[img_in, style, api_key_in],
            outputs=[img_out, cnn_out, llm_out],
        )
        img_in.change(
            fn=process,
            inputs=[img_in, style, api_key_in],
            outputs=[img_out, cnn_out, llm_out],
        )

    return demo


# ═══════════════════════════════════════
# 測試入口
# ═══════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--app",     action="store_true", help="啟動 Gradio App")
    parser.add_argument("--test",    action="store_true", help="測試 LLM 解讀")
    parser.add_argument("--api-key", type=str, default=None)
    args = parser.parse_args()

    if args.app:
        demo = build_full_app()
        demo.launch(server_port=7860, inbrowser=True)

    elif args.test:
        # 用示範資料測試 LLM 解讀
        mock_cnn = {
            "nose":  {"label": "挺鼻型",   "confidence": 0.87},
            "eye":   {"label": "明亮眼型", "confidence": 0.74},
            "brow":  {"label": "上揚眉",   "confidence": 0.81},
            "mouth": {"label": "豐唇型",   "confidence": 0.69},
            "face":  {"label": "瓜子臉",   "confidence": 0.83},
        }
        mock_features = {
            "face_ratio": 0.72, "san_ting_balance": 0.91,
            "left_eye_ratio": 0.33, "right_eye_ratio": 0.31,
            "nose_width_ratio": 0.26, "mouth_width_ratio": 0.41,
        }
        print("測試風格：傳統典雅")
        result = generate_llm_physiognomy(
            mock_cnn, mock_features,
            style="traditional",
            api_key=args.api_key,
        )
        print(result)

    else:
        print("""
面相 LLM 解讀模組使用說明：

  啟動完整 App（含 LLM）：
    python step2d_llm.py --app

  測試 LLM 解讀（需 API Key）：
    python step2d_llm.py --test --api-key sk-ant-...

  或設環境變數：
    export ANTHROPIC_API_KEY=sk-ant-...
    python step2d_llm.py --test

  在 Gradio App 裡直接貼入 API Key 欄位也可以。
        """)
