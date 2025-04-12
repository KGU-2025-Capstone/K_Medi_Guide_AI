from openai import OpenAI
from config import OPENAI_API_KEY
from services.session_store import session_lang

client = OpenAI(api_key=OPENAI_API_KEY)

def translate_to_user_lang(session_id, text_ko):
    target_lang = session_lang.get(session_id)
    if not target_lang or target_lang.lower() == "ko":
        return text_ko
    prompt = f"다음 한국어 문장을 {target_lang.upper()}로 번역해줘. 설명 없이 번역된 문장만 출력해: '{text_ko}'"
    try:
        translated = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "너는 다국어 번역 도우미야."},
                {"role": "user", "content": prompt}
            ]
        ).choices[0].message.content.strip()
        return translated
    except:
        return text_ko

# ... (생략된 import 및 초기 설정, detect_language, translate_to_user_lang 등 동일)
