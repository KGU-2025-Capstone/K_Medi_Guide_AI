from flask import jsonify, session
from openai import OpenAI
from config import OPENAI_API_KEY
import re

client = OpenAI(api_key=OPENAI_API_KEY)

def extract_medcine_name(user_input):
    prompt = f"""다음 문장에서 의약품 이름만 한국어 또는 영어로 하나만 추출해줘. 설명 없이 결과만 출력해. 문장: "{user_input}" """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "약 이름을 추출하는 도우미야."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return jsonify({"error": translate_to_user_lang("약 이름 추출 중 오류 발생"), "details": str(e),"next": "/start"}), 500
    #리턴 오류 부분 수정 필요

def translate_to_user_lang(text_ko):
    target_lang = session.get('language')
    if not target_lang or target_lang == "ko":
        return text_ko
    prompt = f"""다음 한국어 문장을 {target_lang}로 친절하게 번역하고 설명 없이 번역된 문장만 출력해. 문장: '{text_ko}'"""
    try:
        translated = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 친절한 다국어 번역 도우미야."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        ).choices[0].message.content.strip()
        return translated
    except:
        return text_ko
    
def improved_readability(user_input):
    target_lang = session.get('language', 'ko')
    combined_name = session.get('combined_name')
    prompt_data = READABILITY_PROMPTS.get(target_lang)

    if not prompt_data:
        return jsonify({
            "error": translate_to_user_lang("현재 해당 언어는 지원되지 않습니다."),
            "details": f"지원되지 않는 언어: {target_lang}",
            "next": "/start"
        }), 400

    input_format = prompt_data.get("input_format", '문장: """{user_input}"""')
    name_preserve_notice = ("\n\n주의: <<약이름>>은 고유명사입니다. 절대 번역하거나 수정하지 말고 그대로 사용하세요.")
    prompt = f"{prompt_data['prompt']}{name_preserve_notice}\n\n{input_format.format(user_input=user_input)}"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 문장의 가독성을 개선하는 도우미입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        response_text = response.choices[0].message.content.strip()

        # 결과에서 <<약이름>>을 combined_name으로 되돌리기
        final_result = response_text.replace("<<약이름>>", combined_name)

        return final_result
    except Exception as e:
        return jsonify({"error": translate_to_user_lang("문장 가독성 개선 중 오류 발생"), "details": str(e), "next": "/start"}), 500

    
def replace_translated_name(translated_text, insert_text):
    target_lang = session.get('language')

    if target_lang == "en":
        # 영어: be 동사 기준
        match = re.search(r"is", translated_text)
    elif target_lang == "jp":
        # 일본어: 조사 'は' 기준
        match = re.search(r"は", translated_text)
    elif target_lang == "zh":
        # 중국어: '是' (shì) 기준
        match = re.search(r"是", translated_text)
    else:
        # 기타 언어는 그냥 원문 반환
        return translated_text

    if match:
        split_idx = match.start()
        return insert_text + translated_text[split_idx:]
    else:
        # 조사 못 찾은 경우에도 fallback
        return translated_text
    
READABILITY_PROMPTS = {
    "ko": {
        "prompt": """다음 문장의 가독성을 마크다운 형식으로 개선하세요.

- 줄바꿈, 항목 나열, 이모지, 굵은 글씨(**강조**)를 활용하세요.
- 이모지를 기준으로 줄바꿈을 넣어서 문단을 나누세요.
- **강조 뒤에는 조사(은/는, 이/가 등)를 붙이지 말고 띄어쓰기 후 작성하세요.**
- 예: "**타이레놀정 500밀리그람 (아세트아미노펜)(Tylenol Tablet 500mg)** 은", "**통증 완화** 에 효과가 있어요."
- 문장의 의미나 논리 흐름은 변경하지 마세요.
- 출력은 마크다운 형식으로만 작성하고, 그 외 부가 설명은 하지 마세요.
- 입력 언어를 절대 번역하지 말고 ko 그대로 출력하세요.

**입력 예시:**  
\"두통, 근육통이 있으시군요.타이레놀정 500밀리그람 (아세트아미노펜)(Tylenol Tablet 500mg)은 통증 완화에 효과가 있어요.\"

**출력 예시:**  
🤕 **두통, 근육통** 증상이 있으시군요!  
💊 **타이레놀정 500밀리그람 (아세트아미노펜)(Tylenol Tablet 500mg)** 은 **통증 완화** 에 효과가 있어요!

**입력 예시:**  
\"💊게루삼엠정(건조수산화알루미늄겔)(Gerusam M Tab.(Dried Aluminium Hydroxide Gel))🥛성인은 1회 12정(300600 mg), 1일 3회 식간에 복용하면 돼요. 연령, 증상에 따라 적절히 증감하면 돼요.‼️투석요법을 받고 있는 환자는 이 약을 복용하지 마세요. 이 약을 복용하기 전에 인산염 결핍, 신장애 환자, 신장병 경험자는 의사 또는 약사와 상의하세요. 장기연용을 하지 마세요.\"

**출력 예시:**  
💊 **게루삼엠정 (건조수산화알루미늄겔) (Gerusam M Tab. (Dried Aluminium Hydroxide Gel))**

🥛 성인은 1회 **12정 (300~600 mg)**, 1일 **3회 식간에 복용**하세요.  
연령이나 증상에 따라 **복용량을 조절**할 수 있어요.

‼️ 다음에 해당하는 경우 복용하지 마세요:  
- **투석요법 중인 환자**  
- **인산염 결핍 환자**  
- **신장애 또는 신장병 경험자** (복용 전 **의사나 약사와 상담**하세요)

⚠️ **장기간 복용은 피해주세요.**
"""
    },
    "en": {
        "prompt": """Improve the readability of the following sentence using markdown:

- Use line breaks, bullet points, emojis, and bold text (emphasis) to improve readability.
- Insert line breaks based on emojis to divide paragraphs.
- Do not attach grammatical particles directly to bolded words—add a space after the bold text.
- Do not change the meaning or logical flow of the sentence.
- Return only the formatted Markdown output. Do not add any explanations or extra comments.
- Do not translate the input language at all and output the English (en) text as is.

**Example Input:**  
\"You have symptoms like headache and muscle pain. 타이레놀정 500밀리그람 (아세트아미노펜)(Tylenol Tablet 500mg) is effective for pain relief.\"

**Example Output:**  
🤕 **Headache and muscle pain** are your symptoms!  
💊 **타이레놀정 500밀리그람 (아세트아미노펜)(Tylenol Tablet 500mg)** is effective for **pain relief**!

**Example Input:**  
\"💊게루삼엠정(건조수산화알루미늄겔) (Gerusam M Tab.(Dried Aluminium Hydroxide Gel))🥛Adults should take 12 tablets (300–600 mg) three times a day between meals. Dosage may vary based on age and symptoms.‼️Patients undergoing dialysis should not take this medicine. Consult your doctor or pharmacist if you have phosphate deficiency, kidney problems, or a history of kidney disease. Do not use long-term.\"

**Example Output:**  
💊 **게루삼엠정(건조수산화알루미늄겔) (Gerusam M Tab.(Dried Aluminium Hydroxide Gel))**

🥛 Adults: take **12 tablets (300–600 mg)**, **3 times daily between meals**.  
Adjust dosage based on **age or symptoms**.

‼️ Do **not take** if you are:
- **On dialysis**  
- **Phosphate-deficient**  
- **Suffering from or with a history of kidney problems** (Consult **doctor or pharmacist** first)

⚠️ **Avoid long-term use.**
"""
    },
    "ja": {
        "prompt": """次の文章の可読性をMarkdown形式で改善してください：

- 改行、箇条書き、絵文字、太字（強調）を使って読みやすくしてください。
- 絵文字を基準にして改行を入れ、段落を分けてください。
- 太字の直後に助詞（は、が、のなど）を付けず、スペースを空けてから記述してください。
- 例: "타이레놀정 500밀리그람 (아세트아미노펜)(Tylenol Tablet 500mg) は", "痛みの緩和 に効果があります。"
- 文章の意味や論理的な流れを変更しないでください。
- 出力はMarkdown形式のみで行い、説明や補足は一切しないでください。
- 入力言語を絶対に翻訳せず、日本語（jp）のまま出力してください。


**入力例：**  
\"頭痛と筋肉痛がありますね。타이레놀정 500밀리그람 (아세트아미노펜)(Tylenol Tablet 500mg)は痛みの緩和に効果があります。\"

**出力例：**  
🤕 **頭痛と筋肉痛** の症状がありますね！  
💊 **타이레놀정 500밀리그람 (아세트아미노펜)(Tylenol Tablet 500mg)** は **痛みの緩和** に効果があります！

**入力例：**  
\"💊게루삼엠정(건조수산화알루미늄겔) (Gerusam M Tab.(Dried Aluminium Hydroxide Gel))🥛成人は1回12錠（300〜600mg）を1日3回、食間に服用してください。年齢や症状に応じて適切に調整してください。‼️透析を受けている方は服用しないでください。服用前にリン欠乏症、腎障害、腎疾患の既往歴がある場合は医師または薬剤師に相談してください。長期間の使用は避けてください。\"

**出力例：**  
💊 게루삼엠정(건조수산화알루미늄겔) (Gerusam M Tab.(Dried Aluminium Hydroxide Gel))**

🥛 成人は **1回12錠（300〜600mg）** を **1日3回、食間に服用** してください。  
**年齢や症状に応じて調整** 可能です。

‼️ 以下に該当する方は服用しないでください：  
- **透析中の方**  
- **リン欠乏症の方**  
- **腎障害または腎疾患の既往歴のある方**（事前に **医師や薬剤師に相談** してください）

⚠️ **長期間の使用は避けてください。**
"""
    },
    "zh": {
        "prompt": """请使用Markdown格式优化以下句子的可读性：

- 请使用换行、项目符号、表情符号和加粗（强调）来提升可读性。
- 请以表情符号为标志进行换行，划分段落。
- 不要将助词直接连接在加粗文字后面，请在加粗后空一格再写。
- 请不要改变句子的原意或逻辑结构。
- 只输出Markdown格式的内容，不要添加任何解释或注释。
- 绝对不要翻译输入语言，直接原样输出中文（zh）。

**输入示例：**  
\"您有头痛和肌肉酸痛的症状。타이레놀정 500밀리그람 (아세트아미노펜)(Tylenol Tablet 500mg)对缓解疼痛有效。\"

**输出示例：**  
🤕 **头痛和肌肉酸痛** 是您的症状！  
💊 **타이레놀정 500밀리그람 (아세트아미노펜)(Tylenol Tablet 500mg)** 对于 **缓解疼痛** 非常有效！

**输入示例：**  
\"💊게루삼엠정(건조수산화알루미늄겔) (Gerusam M Tab.(Dried Aluminium Hydroxide Gel))🥛成人每次服用12片（300–600 mg），每天三次，餐间服用。根据年龄和症状适当增减。‼️正在接受透析的患者请勿服用。服用前请咨询医生或药师，若您有磷缺乏、肾功能障碍或肾病史。避免长期使用。\"

**输出示例：**  
💊 **게루삼엠정(건조수산화알루미늄겔) (Gerusam M Tab.(Dried Aluminium Hydroxide Gel))**

🥛 成人每次服用 **12片（300–600 mg）**，每天 **三次餐间服用**。  
可根据 **年龄或症状调整用量**。

‼️ 请勿在以下情况下服用：  
- **正在接受透析**  
- **磷缺乏者**  
- **肾功能障碍或有肾病史者**（服用前请咨询 **医生或药师**）

⚠️ **避免长期服用。**
"""
    }
}

