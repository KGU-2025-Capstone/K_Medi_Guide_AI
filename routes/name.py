from flask import Blueprint, request, jsonify, session
from pymongo import MongoClient
from openai import OpenAI
from services.gpt_service import translate_to_user_lang, extract_medcine_name
from services.utils import softmax_with_temperature, detect_language
from config import OPENAI_API_KEY, MONGODB_URI
import numpy as np
import re

#환경 및 라우트 설정
client = OpenAI(api_key=OPENAI_API_KEY)
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client['K_Medi_Guide']
collection = db['Api']
bp = Blueprint('name', __name__)

def get_retry_count():
    return session.get('retry_count', 0)

@bp.route('/name', methods=['POST'])
def extract_and_match_medicine_name():
    #사용자 입력
    data = request.get_json()
    user_input = data.get("input", "")
    session['language'] = detect_language(user_input)

    #name라우트에서 select라우트로 갈 경우 select출력 문장에서 증상 부분을 제외하기 위한 설정
    session['name_to_select'] = True

    if not user_input:
        return jsonify({"error": translate_to_user_lang("입력이 필요합니다."), "next": "/name", "response_type": "name_fail"}), 400
    
    #사용자 입력에서 약 이름 추출 및 DB에서 검색
    extracted_name = extract_medcine_name(user_input)
    query = {"itemName": {"$regex": extracted_name, "$options": "i"}} if re.search(r'[가-힣]', extracted_name) else {"engName": {"$regex": extracted_name, "$options": "i"}}
    matching_docs = list(collection.find(query))

    #약 검색에 실패할 경우 3회 재시도 가능. 초과할 경우 처음으로 돌아감.
    if not matching_docs:
        session['retry_count'] = get_retry_count() + 1
        if get_retry_count >= 3:
            session['retry_count'] = 0
            return jsonify({"error": translate_to_user_lang("3회 시도에도 약을 찾지 못했습니다. 처음으로 돌아갑니다."), "next": "/start", "response_type": "name_fail"}), 404
        return jsonify({"error": translate_to_user_lang(f"관련된 약 이름을 찾지 못했습니다. 다시 입력해주세요. ({get_retry_count()}/3)"), "next": "/name", "response_type": "name_fail"}), 404

    #가중치를 고려하여 softmax알고리즘에 따라 약을 선택
    weights = [float(doc.get("weight", 1.0)) for doc in matching_docs]
    probabilities = softmax_with_temperature(weights, temperature=1.0)
    sampled = np.random.choice(matching_docs, size=min(5, len(matching_docs)), replace=False, p=probabilities)

    #약 후보 목록을 정리
    candidates = []
    for doc in sampled:
        name_ko = doc.get("itemName", "")
        name_en = doc.get("engName", "")
        combined_name = f"{name_ko}({name_en})" if name_en else name_ko
        candidates.append({
            "itemName": combined_name,
            "name_ko": name_ko
        })

    #시도 횟수 초기화 및 정보 반환
    session['retry_count'] = 0
    return jsonify({
        "medicine_candidates": candidates,
        "message": translate_to_user_lang("다음 중 어떤 약이 궁금하신가요?"),
        "next": "/select",
        "response_type": "name_success"
    })
