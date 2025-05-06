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
    prompt = f"""다음 한국어 문장을 {target_lang}로 친절하게 번역하고 번역된 문장만 출력해. 문장: '{text_ko}'"""
    try:
        translated = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "너는 친절한 다국어 번역 도우미야."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        ).choices[0].message.content.strip()
        return translated
    except:
        return "text_ko"
    
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
