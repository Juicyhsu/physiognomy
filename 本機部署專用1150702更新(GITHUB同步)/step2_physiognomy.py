"""
================================================
面相辨識專題 — Step 2: 面相規則引擎
================================================
根據 Step 1 提取的特徵值，套用面相學理論
生成個人化面相分析報告

面相依據：
  - 三庭五眼比例學
  - 臉型分類（圓/方/長/倒三角/橢圓）
  - 五官個別特質分析

執行方式:
    python step2_physiognomy.py
"""

from dataclasses import dataclass
from typing import Dict, List


# ─────────────────────────────────────────
# 資料結構
# ─────────────────────────────────────────
@dataclass
class Trait:
    category: str     # 五官分類
    feature:  str     # 特徵名稱
    score:    float   # 特徵強度 0.0~1.0
    title:    str     # 面相標題
    analysis: str     # 面相解讀
    advice:   str     # 建議

@dataclass
class PhysiognomyReport:
    face_type:  str
    traits:     List[Trait]
    summary:    str
    lucky_hint: str


# ─────────────────────────────────────────
# 臉型判斷
# ─────────────────────────────────────────
def classify_face_type(features: Dict) -> str:
    """
    根據臉寬/高比例 + 下巴比例判斷臉型
    臉型: 圓形、方形、長形、倒三角（心形）、橢圓（瓜子臉）
    """
    ratio = features["face_ratio"]        # 寬/高
    jaw   = features["jaw_ratio"]         # 下巴寬 vs 臉寬
    lower = features["lower_zone_ratio"]  # 下庭比例

    if ratio > 0.85:                      # 臉很寬
        if jaw > 0.75:
            return "方形臉"               # 下巴也寬 → 方形
        else:
            return "圓形臉"               # 下巴較窄 → 圓形
    elif ratio < 0.65:                    # 臉很長
        return "長形臉"
    else:                                 # 中等比例
        if jaw < 0.62 and lower < 0.28:
            return "倒三角臉（心形）"
        else:
            return "橢圓臉（瓜子臉）"


FACE_TYPE_ANALYSIS = {
    "圓形臉": (
        "親和、人緣佳",
        "圓臉者面相主財運與人際，天生具有親和力，易獲他人信任。"
        "個性圓融，善於調和人際關係，適合需要溝通協調的工作。"
        "財運方面中年後逐漸穩定，晚年多福。",
        "多培養決斷力，避免優柔寡斷。"
    ),
    "方形臉": (
        "剛毅、意志力強",
        "方臉者骨骼剛健，個性堅毅，行動力強，適合需要領導力的職位。"
        "意志力出眾，一旦設定目標便會全力以赴，不輕易放棄。"
        "但有時過於固執，需注意人際彈性。",
        "適時展現柔和一面，剛柔並濟方能長久。"
    ),
    "長形臉": (
        "思慮深遠、有謀略",
        "長臉者思考細膩，凡事謀定而後動，適合學術研究或策略規劃。"
        "個性較為內斂，不輕易表露情緒，但內心世界豐富。"
        "適合需要深度思考的職業，如學者、分析師、策略顧問。",
        "多與人溝通，避免過於封閉自我想法。"
    ),
    "倒三角臉（心形）": (
        "聰慧、感情豐富",
        "心形臉者天庭飽滿，主聰明才智，思維靈活，學習能力強。"
        "感情方面細膩豐富，對美的事物有高度敏感度，藝術天賦出色。"
        "適合創意相關行業，或需要高度智慧的工作。",
        "注意情緒管理，感情豐富易受外界影響，需建立穩定的內心基礎。"
    ),
    "橢圓臉（瓜子臉）": (
        "均衡、多才多藝",
        "橢圓臉被視為五行最為均衡的臉型，代表性格平和、處事圓融。"
        "具備多方面才能，適應力強，在各種環境都能發揮所長。"
        "運勢較為穩定，人生各階段均有貴人相助。",
        "善用均衡特質，廣泛涉獵不同領域，打造全方位優勢。"
    ),
}


# ─────────────────────────────────────────
# 五官特徵規則
# ─────────────────────────────────────────
def analyze_eyes(features: Dict) -> Trait:
    """眼睛分析 — 主心智與運勢"""
    eye_open = (features["left_eye_ratio"] + features["right_eye_ratio"]) / 2
    gap_ratio = features["eye_gap_ratio"]

    if eye_open > 0.35:
        title = "眼神明亮開闊"
        analysis = ("眼睛明亮且開度大，主心思開明、觀察力敏銳。"
                    "此類眼型的人直覺力強，思維開放，容易接受新事物。"
                    "在人際交往上能迅速讀懂對方情緒，具有高情商。")
        advice = "善用敏銳觀察力，在需要洞察人心的領域可大放異彩。"
        score = min(eye_open * 2, 1.0)
    elif eye_open > 0.22:
        title = "眼神沉穩內斂"
        analysis = ("眼睛適中，不大不小，代表個性穩重踏實，不衝動冒進。"
                    "做事有條理，善於分析規劃，是值得信賴的好夥伴。"
                    "事業運穩定上升，中年後有較大發展空間。")
        advice = "保持穩健作風，但適時展現魄力，把握機遇時要果斷行動。"
        score = 0.7
    else:
        title = "眼神深邃謹慎"
        analysis = ("眼睛較細長，代表個性謹慎細心，善於深度思考。"
                    "此類人做事一絲不苟，追求完美，適合需要精密分析的工作。"
                    "內心世界豐富，感情深沉，忠誠度高。")
        advice = "適度放開心胸，展現真實自我，有助於人際關係的拓展。"
        score = 0.6

    # 眼距加成判斷
    if gap_ratio > 1.2:
        analysis += " 兩眼間距寬，主心胸開闊、包容力強。"
    elif gap_ratio < 0.8:
        analysis += " 兩眼間距較窄，主專注力強、目標明確。"

    return Trait("眼睛", "眼型分析", score, title, analysis, advice)


def analyze_eyebrows(features: Dict) -> Trait:
    """眉毛分析 — 主個性與兄弟運"""
    brow_ratio = features["brow_eye_ratio"]
    brow_dist  = features["brow_eye_distance"]

    if brow_ratio > 1.2:
        title    = "眉長過眼、英氣勃發"
        analysis = ("眉毛長過眼角，在面相中主貴氣，代表有氣魄、有擔當。"
                    "此類人具領導特質，做事有計劃，能帶領他人向前。"
                    "兄弟感情和睦，貴人緣佳，事業上多有助力。")
        advice   = "領導力是天賦，善加運用，但避免過於強勢忽視他人意見。"
        score    = 0.85
    elif brow_ratio > 0.9:
        title    = "眉目相稱、均衡端正"
        analysis = ("眉毛長度與眼睛相稱，代表個性均衡，處事不偏不倚。"
                    "頭腦清晰，情緒穩定，是職場上的可靠人才。"
                    "人緣佳，能與各類型人相處融洽。")
        advice   = "均衡穩定是最大優點，持續修身養性，運勢將逐步提升。"
        score    = 0.75
    else:
        title    = "眉短意志堅定"
        analysis = ("眉毛較短，主個性獨立，不依賴他人，自力更生能力強。"
                    "雖然有時顯得孤傲，但堅毅不拔，遇困難不輕言放棄。"
                    "適合自行創業或獨立作業的工作環境。")
        advice   = "學習與人合作，孤獨前行固然強大，團隊的力量更不容小覷。"
        score    = 0.65

    return Trait("眉毛", "眉型分析", score, title, analysis, advice)


def analyze_nose(features: Dict) -> Trait:
    """鼻子分析 — 主財運與自信"""
    nose_w = features["nose_width_ratio"]
    nose_s = features["nose_shape_ratio"]   # 越大越寬扁

    if nose_w < 0.28 and nose_s < 0.6:
        title    = "鼻梁高挺、財運亨通"
        analysis = ("鼻梁高而挺直，是財運佳的面相。此類人自信心強，"
                    "做事積極主動，事業心重，善於把握商機。"
                    "財帛宮豐盛，代表一生財運不缺，中年後財富積累明顯。")
        advice   = "善加運用財運優勢，同時注意財務規劃，積累財富並妥善管理。"
        score    = 0.9
    elif nose_w < 0.35:
        title    = "鼻型端正、財運穩健"
        analysis = ("鼻型端正適中，主財運穩定，不會大起大落。"
                    "個性踏實勤奮，靠自身努力累積財富，不喜歡投機。"
                    "適合穩健型投資與事業發展，細水長流型的財富積累。")
        advice   = "穩中求進，保持踏實作風，財富會隨著時間持續累積。"
        score    = 0.75
    else:
        title    = "鼻翼寬廣、人緣財運"
        analysis = ("鼻翼較寬，主人緣廣，人脈豐富，易得他人幫助。"
                    "財運多來自人際關係，善用廣大人脈可帶來財富機會。"
                    "個性豪爽，不計較小節，易贏得他人好感與信任。")
        advice   = "善用人脈資源，同時注意量入為出，避免因過於慷慨影響財務。"
        score    = 0.7

    return Trait("鼻子", "鼻型分析", score, title, analysis, advice)


def analyze_mouth(features: Dict) -> Trait:
    """嘴巴分析 — 主表達力與感情"""
    mouth_w = features["mouth_width_ratio"]
    lip_t   = features["lip_thickness"]

    if mouth_w > 0.45:
        title    = "嘴型寬大、表達力強"
        analysis = ("嘴巴寬大，主口才佳，表達能力強，善於說服他人。"
                    "感情方面熱情主動，感情生活豐富。"
                    "適合業務、公關、教學等需要溝通的職業，口才是最大資產。")
        advice   = "善用口才，但注意言多必失，在正式場合說話前先深思熟慮。"
        score    = 0.8
    elif mouth_w > 0.35:
        title    = "嘴型適中、言行一致"
        analysis = ("嘴型大小適中，主言行一致，說到做到，信用好。"
                    "感情穩定，對伴侶忠誠，重視承諾。"
                    "在職場上因誠信而贏得好口碑，長期發展穩健。")
        advice   = "繼續保持誠信原則，這是最珍貴的資產。"
        score    = 0.75
    else:
        title    = "嘴型小巧、心思細膩"
        analysis = ("嘴型小巧，主個性謹慎，說話精準，不說廢話。"
                    "心思細膩，對美有高度鑑賞力，品味出眾。"
                    "感情方面含蓄內斂，需要對方主動才能開花結果。")
        advice   = "適度主動表達自己，有時候說出口比放在心裡更有效果。"
        score    = 0.65

    if lip_t > 0.06:
        analysis += " 嘴唇豐厚，主感情豐沛，對伴侶溫柔體貼。"

    return Trait("嘴巴", "嘴型分析", score, title, analysis, advice)


def analyze_san_ting(features: Dict) -> Trait:
    """三庭分析 — 主整體運勢格局"""
    balance = features["san_ting_balance"]
    upper   = features["upper_zone_ratio"]
    middle  = features["middle_zone_ratio"]
    lower   = features["lower_zone_ratio"]

    if balance > 0.85:
        title    = "三庭均等、格局宏大"
        analysis = ("面部三庭（上中下三段）比例均衡，是面相學中的貴相。"
                    "上庭（額頭）飽滿代表青年運佳；中庭（鼻區）端正代表中年財運穩；"
                    "下庭（下巴）圓潤代表晚年福壽雙全。"
                    "此類人一生運勢平穩上升，各階段都有收穫。")
        advice   = "珍惜均衡的先天條件，後天持續努力，運勢將錦上添花。"
        score    = 0.9
    elif upper > middle and upper > lower:
        title    = "天庭飽滿、青年運佳"
        analysis = ("上庭（額頭至眉間）比例較大，主頭腦聰明，年輕時運勢旺盛。"
                    "學習能力強，在求學與初入職場階段有亮眼表現。"
                    "適合早早確立目標，趁年輕時積累資源與人脈。")
        advice   = "善用青年時期的好運，打下紮實基礎，中年後運勢將更加穩固。"
        score    = 0.78
    elif lower > middle and lower > upper:
        title    = "地庫豐隆、晚年有福"
        analysis = ("下庭（鼻底至下巴）比例較大，代表晚年福氣深厚。"
                    "此類面相主晚年衣食無憂，子女孝順，生活安康。"
                    "個性務實，重視生活品質，善於享受當下。")
        advice   = "未雨綢繆，做好人生各階段規劃，晚年的好福氣需要年輕時的努力奠基。"
        score    = 0.72
    else:
        title    = "中庭端正、中年得志"
        analysis = ("中庭（眉間至鼻底）比例佳，主中年事業有成，財運興旺。"
                    "在30~50歲的人生黃金期，事業與財富都有顯著成長。"
                    "適合在職場深耕，35歲後往往迎來人生轉折點。")
        advice   = "把握中年事業黃金期，積極佈局，此時的努力將帶來豐厚回報。"
        score    = 0.75

    return Trait("整體格局", "三庭比例", score, title, analysis, advice)


# ─────────────────────────────────────────
# 主分析引擎
# ─────────────────────────────────────────
def generate_report(features: Dict) -> PhysiognomyReport:
    """整合所有特徵，生成完整面相報告"""

    # 臉型
    face_type = classify_face_type(features)
    ft_data   = FACE_TYPE_ANALYSIS[face_type]

    # 各部位分析
    traits = [
        analyze_san_ting(features),
        analyze_eyes(features),
        analyze_eyebrows(features),
        analyze_nose(features),
        analyze_mouth(features),
    ]

    # 綜合評分
    avg_score = sum(t.score for t in traits) / len(traits)

    # ── 精準條件判斷摘要（每項特徵值對應具體面相判斷）──────────────
    summary_parts = []

    # 1. 三庭格局（san_ting_balance / zone ratios）
    balance = features.get("san_ting_balance", 0.80)
    upper   = features.get("upper_zone_ratio", 0.33)
    lower   = features.get("lower_zone_ratio", 0.33)
    if balance > 0.85:
        summary_parts.append("三庭比例均衡（均衡度 {:.2f}），上中下庭各司其職，乃天生格局宏大之貴相，一生運勢平穩上升".format(balance))
    elif upper > 0.37:
        summary_parts.append("天庭（額頭）飽滿突出（比例 {:.2f}），主青年時期頭腦靈活、貴人運旺，學業與早期事業發展順遂".format(upper))
    elif lower > 0.37:
        summary_parts.append("地庫（下巴）豐隆厚實（比例 {:.2f}），主晚年福氣深厚、子嗣孝順、衣食豐足".format(lower))
    else:
        summary_parts.append("中庭（眉至鼻底）端正（均衡度 {:.2f}），中年事業財運興旺，三十五歲後迎來人生黃金期".format(balance))

    # 2. 印堂寬窄（glabella_width_ratio）
    glabella = features.get("glabella_width_ratio", 0.15)
    if glabella > 0.20:
        summary_parts.append("印堂開闊（寬度比 {:.2f}），心胸廣闊，主貴人緣旺、財庫充盈，二十五歲前已有明顯助力".format(glabella))
    elif glabella < 0.13:
        summary_parts.append("印堂略窄（寬度比 {:.2f}），早年需多自力更生，三十歲後隨閱歷積累，運勢方逐漸開展".format(glabella))

    # 3. 眉壓眼指數（brow_eye_distance）
    broweye = features.get("brow_eye_distance", 0.07)
    if broweye < 0.055:
        summary_parts.append("眉眼間距緊縮（指數 {:.3f}），主性格急切果斷，行事魄力十足，惟情緒起伏較大，易因急躁錯失細節".format(broweye))
    elif broweye > 0.10:
        summary_parts.append("眉眼間距寬廣（指數 {:.3f}），性格溫和包容，善於聆聽，人際關係和諧，是天生的協調者".format(broweye))

    # 4. 鼻部財運（nose_width_ratio）
    nose_w = features.get("nose_width_ratio", 0.30)
    if nose_w < 0.27:
        summary_parts.append("鼻梁纖挺、財帛宮緊束（比例 {:.3f}），主自身理財能力強，財不外露，一生積財有道".format(nose_w))
    elif nose_w > 0.38:
        summary_parts.append("鼻翼寬廣（比例 {:.3f}），廣結善緣，財運多賴人脈帶動，需防慷慨漏財之象".format(nose_w))

    # 5. 人中長度（philtrum_ratio）
    philtrum = features.get("philtrum_ratio", 0.035)
    if philtrum > 0.05:
        summary_parts.append("人中深長（比例 {:.3f}），主晚年子嗣緣厚、福壽綿長，生命力旺盛，老來得享天倫之樂".format(philtrum))
    elif philtrum < 0.028:
        summary_parts.append("人中偏短（比例 {:.3f}），晚年宜注重養生修身，飲食規律可有效補強下庭福氣".format(philtrum))

    # 6. 嘴部口才（mouth_width_ratio）
    mouth_w = features.get("mouth_width_ratio", 0.38)
    if mouth_w > 0.44:
        summary_parts.append("嘴型寬大（比例 {:.3f}），口才出眾，適合公關、業務、教育等需要說服力的領域，一開口即能聚攏人心".format(mouth_w))
    elif mouth_w < 0.30:
        summary_parts.append("嘴型小巧（比例 {:.3f}），說話精準謹慎，品味細膩，言出必行，信用是最大資產".format(mouth_w))

    # 7. 眼部開闊（eye openness）
    avg_eye = (features.get("left_eye_ratio", 0.27) + features.get("right_eye_ratio", 0.27)) / 2
    if avg_eye > 0.35:
        summary_parts.append("眼神開闊明亮（開度 {:.3f}），觀察力敏銳，情商高，在人際場合能迅速讀懂對方情緒".format(avg_eye))
    elif avg_eye < 0.20:
        summary_parts.append("眼神深邃內斂（開度 {:.3f}），思慮細密，忠誠度高，適合需要長期深耕的工作與感情".format(avg_eye))

    # ── 組合總結 ────────────────────────────────────────────────────
    top_trait = max(traits, key=lambda t: t.score)
    score_desc = (
        "格局宏大，命中帶貴，前途不可限量" if avg_score > 0.82
        else "穩健上進，持續耕耘必有豐厚收穫" if avg_score > 0.72
        else "各有所長，善用優勢補強弱項，運勢可大幅提升"
    )
    detail = "\n".join(f"• {part}" for part in summary_parts) if summary_parts else "• 五官格局均衡，整體相格中正。"
    summary = (
        f"【{face_type}（{ft_data[0]}）】\n"
        + detail + "\n\n"
        + f"🌟 本次最突出之面相優勢為「{top_trait.title}」\n"
        + f"📊 整體面相評分 {avg_score:.0%}，{score_desc}。"
    )

    # ── 精準開運建議（針對最弱特徵給予具體面相學改運指引）──────────
    weak_trait = min(traits, key=lambda t: t.score)
    lucky_map = {
        "眼睛": (
            "眼部為「監察官」，神氣之聚所。建議每日入睡前以溫熱毛巾敷眼三分鐘，"
            "使眼神常保清澈有神；平日避免過度使用電子螢幕以防眼神呆滯，"
            "眼神有光者識人辨事之力大增，可有效避開小人口舌之災。"
        ),
        "眉毛": (
            "眉為「保壽官」，主兄弟緣分與貴人助力。建議適當修整眉形，"
            "使眉毛清晰整潔、不雜亂散漫；眉型宜自然流暢，"
            "眉毛清秀者人脈與職場助力明顯提升，有利突破現有格局。"
        ),
        "鼻子": (
            "鼻為「審辨官（財帛宮）」，主一生財庫豐盈與否。"
            "建議保持鼻準（鼻尖）清潔明潤，黑頭或暗沉主財運受阻；"
            "理財上宜採穩健保守策略，避免鼻孔過於外露（漏財相），"
            "切忌衝動型高風險投資，守財與聚財並重方為上策。"
        ),
        "嘴巴": (
            "嘴為「出納官」，主信用口才與晚年福澤。"
            "建議日常保持嘴角微微上揚，面帶從容笑意；"
            "嘴角下垂者宜每日練習微笑肌群，一來聚福氣，"
            "二來提升親和力，感情與職場皆受益。言出必行，信用是最大財富。"
        ),
        "整體格局": (
            "三庭為命運格局之主軸，「相由心生」是面相學的核心。"
            "建議維持規律作息與充足睡眠使氣色紅潤，"
            "印堂（兩眉間）常保光澤明亮；心態積極正向、戒除負面情緒，"
            "則面相自然呈現貴氣，吸引貴人的能量亦隨之增強。"
        ),
    }
    lucky_hint = lucky_map.get(weak_trait.category, lucky_map["整體格局"])

    return PhysiognomyReport(
        face_type  = face_type,
        traits     = traits,
        summary    = summary,
        lucky_hint = lucky_hint,
    )



def print_report(report: PhysiognomyReport):
    """格式化輸出面相報告"""
    print("\n" + "="*55)
    print("         ✨ 面相分析報告 ✨")
    print("="*55)
    print(f"\n【臉型】{report.face_type}")
    ft = FACE_TYPE_ANALYSIS[report.face_type]
    print(f"  主要特質：{ft[0]}")
    print(f"  面相解讀：{ft[1]}")
    print(f"  建議：{ft[2]}")

    for trait in report.traits:
        print(f"\n【{trait.category}】{trait.title}  (強度 {trait.score:.0%})")
        print(f"  {trait.analysis}")
        print(f"  💡 {trait.advice}")

    print(f"\n{'─'*55}")
    print(f"📋 綜合總結：\n  {report.summary}")
    print(f"\n🍀 開運建議：\n  {report.lucky_hint}")
    print("="*55 + "\n")

    return report


# ─────────────────────────────────────────
# 測試入口
# ─────────────────────────────────────────
if __name__ == "__main__":
    import json

    # 讀取 Step 1 輸出的特徵值
    try:
        with open("output/features.json", encoding="utf-8") as f:
            features = json.load(f)
        print("✅ 讀取特徵值成功")
    except FileNotFoundError:
        # 用預設測試值示範
        print("⚠️  找不到 output/features.json，使用示範數值")
        features = {
            "face_ratio": 0.72, "jaw_ratio": 0.65,
            "upper_zone_ratio": 0.34, "middle_zone_ratio": 0.33,
            "lower_zone_ratio": 0.33, "san_ting_balance": 0.92,
            "left_eye_ratio": 0.31, "right_eye_ratio": 0.30,
            "eye_width_avg": 48.0, "eye_gap_ratio": 1.05,
            "brow_eye_ratio": 1.1, "brow_eye_distance": 0.08,
            "nose_width_ratio": 0.27, "nose_height_ratio": 0.32,
            "nose_shape_ratio": 0.55, "mouth_width_ratio": 0.40,
            "mouth_shape_ratio": 3.2, "lip_thickness": 0.05,
        }

    report = generate_report(features)
    print_report(report)
