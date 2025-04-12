from flask import Blueprint, request, jsonify
from pymongo import MongoClient
from openai import OpenAI
from services.session_store import session_symptoms, session_lang, retry_count
from services.gpt_service import translate_to_user_lang
from services.utils import clean_text, softmax_with_temperature, detect_language
from bs4 import BeautifulSoup
import numpy as np
import re
import os
from config import OPENAI_API_KEY, MONGODB_URI

client = OpenAI(api_key=OPENAI_API_KEY)
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client['K_Medi_Guide']
collection = db['Api']

retry_count = {}
session_symptoms = {}
session_lang = {}

bp = Blueprint('symptom', __name__)

@bp.route('/symptom', methods=['POST'])
def recommend_medicine_by_symptom():
    data = request.get_json()
    session_id = data.get("session_id", "default")
    symptom_input = data.get("input", "")

    session_lang[session_id] = detect_language(symptom_input)

    if not symptom_input:
        return jsonify({
            "error": translate_to_user_lang(session_id, "입력이 필요합니다.")
        }), 400

    prompt = f"""
    다음 문장에서 의학적 증상 키워드만 콤마로 나열해줘. 언어는 한국어, 영어 등 다양할 수 있는데 한국어가 아닐 경우 한국어로 번역해서 나열해줘. 예시나 설명 없이 키워드만 출력해.
    문장: "{symptom_input}"
    """
    try:
        extract_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "의학적 증상을 추출하는 도우미야."},
                {"role": "user", "content": prompt}
            ]
        )
        symptoms_text = extract_response.choices[0].message.content.strip()
        symptoms_ko = [s.strip() for s in symptoms_text.split(",") if s.strip()]
    except Exception as e:
        return jsonify({
            "error": translate_to_user_lang(session_id, "증상 추출 중 오류 발생"),
            "details": str(e)
        }), 500

    if not symptoms_ko:
        return jsonify({
            "error": translate_to_user_lang(session_id, "증상 키워드를 추출하지 못했습니다.")
        }), 400

    results = []
    seen_ids = set()
    for doc in collection.find({}):
        efcy_html = doc.get("efcyQesitm", "")
        plain_efcy = BeautifulSoup(efcy_html, "html.parser").get_text()
        for symptom in symptoms_ko:
            if symptom in plain_efcy:
                _id = str(doc.get("_id"))
                if _id not in seen_ids:
                    results.append(doc)
                    seen_ids.add(_id)
                break

    if not results:
        retry_count[session_id] = retry_count.get(session_id, 0) + 1
        if retry_count[session_id] >= 3:
            return jsonify({
                "error": translate_to_user_lang(session_id, "3회 시도에도 약을 찾지 못했습니다. 처음으로 돌아갑니다.")
            }), 404
        else:
            return jsonify({
                "error": translate_to_user_lang(session_id, f"해당 증상에 맞는 약을 찾지 못했습니다. 다시 입력해주세요. ({retry_count[session_id]}/3)"),
                "extracted_symptoms": [translate_to_user_lang(session_id, s) for s in symptoms_ko]
            }), 404

    session_symptoms[session_id] = symptoms_ko

    weights = [float(r.get("weight", 1.0)) for r in results]
    probabilities = softmax_with_temperature(weights, temperature=1.0)
    sampled = np.random.choice(results, size=min(5, len(results)), replace=False, p=probabilities)

    candidates = []
    for r in sampled:
        name_ko = r.get("itemName", "")
        name_en = r.get("engName", "")
        combined_name = f"{name_ko} ({name_en})" if name_en else name_ko
        efcy_raw = clean_text(r.get("efcyQesitm", ""))
        translated_efcy = translate_to_user_lang(session_id, efcy_raw)

        candidates.append({
            "itemName": combined_name,
            "efcyQesitm": translated_efcy,
            "weight": float(r.get("weight", 1.0))
        })

    retry_count[session_id] = 0
    return jsonify({
        "extracted_symptoms": [translate_to_user_lang(session_id, s) for s in symptoms_ko],
        "medicine_candidates": candidates,
        "message": translate_to_user_lang(session_id, "다음 중 어떤 약이 궁금하신가요?")
    })
