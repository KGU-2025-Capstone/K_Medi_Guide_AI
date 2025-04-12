from flask import Blueprint, request, jsonify
from pymongo import MongoClient
from openai import OpenAI
from services.session_store import attempts, session_lang
from services.gpt_service import translate_to_user_lang
from services.utils import clean_text, softmax_with_temperature, detect_language
from config import OPENAI_API_KEY, MONGODB_URI
from bs4 import BeautifulSoup
import numpy as np
import re

client = OpenAI(api_key=OPENAI_API_KEY)
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client['K_Medi_Guide']
collection = db['Api']

bp = Blueprint('name', __name__)

@bp.route('/name', methods=['POST'])
def extract_and_match_medicine_name():
    data = request.get_json()
    session_id = data.get("session_id", "default")
    user_input = data.get("input", "")
    session_lang[session_id] = detect_language(user_input)

    if not user_input:
        return jsonify({"error": translate_to_user_lang(session_id, "입력이 필요합니다.")}), 400

    prompt = f"""다음 문장에서 의약품 이름만 한국어 또는 영어로 하나만 추출해줘. 설명 없이 결과만 출력해. 문장: "{user_input}" """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "약 이름을 추출하는 도우미야."},
                {"role": "user", "content": prompt}
            ]
        )
        extracted_name = response.choices[0].message.content.strip()
    except Exception as e:
        return jsonify({"error": translate_to_user_lang(session_id, "약 이름 추출 중 오류 발생"), "details": str(e),"next": "/start"}), 500

    query = {"itemName": {"$regex": extracted_name, "$options": "i"}} if re.search(r'[가-힣]', extracted_name) else {"engName": {"$regex": extracted_name, "$options": "i"}}
    matching_docs = list(collection.find(query))

    if not matching_docs:
        attempts[session_id] = attempts.get(session_id, 0) + 1
        if attempts[session_id] >= 2:
            return jsonify({"error": translate_to_user_lang(session_id, "2회 시도 실패. 처음부터 다시 입력해주세요."), "next": "/name"}), 404
        return jsonify({"error": translate_to_user_lang(session_id, "관련된 약 이름을 찾지 못했습니다."), "extracted_name": extracted_name, "next": "/name"}), 404

    weights = [float(doc.get("weight", 1.0)) for doc in matching_docs]
    probabilities = softmax_with_temperature(weights, temperature=1.0)
    sampled = np.random.choice(matching_docs, size=min(5, len(matching_docs)), replace=False, p=probabilities)

    candidates = []
    for doc in sampled:
        name_ko = doc.get("itemName", "")
        name_en = doc.get("engName", "")
        combined_name = f"{name_ko}({name_en})" if name_en else name_ko
        candidates.append({
            "itemName": combined_name,
            "weight": float(doc.get("weight", 1.0))
        })

    attempts[session_id] = 0
    return jsonify({
        "extracted_name": extracted_name,
        "candidates": candidates,
        "message": translate_to_user_lang(session_id, "다음 중 어떤 약이 궁금하신가요?"),
        "next": "/select"
    })
