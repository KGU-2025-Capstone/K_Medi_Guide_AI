from flask import Flask, request, jsonify
from pymongo import MongoClient
from openai import OpenAI
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import os
import random
import re

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
mongo_uri = os.getenv("MONGODB_URI")

client = OpenAI(api_key=openai_api_key)
app = Flask(__name__)

mongo_client = MongoClient(mongo_uri)
db = mongo_client['K_Medi_Guide']
collection = db['Api']

attempts = {}
selected_medicine = {}

@app.route('/medicine/name', methods=['POST'])
def extract_and_match_medicine_name():
    data = request.get_json()
    session_id = data.get("session_id", "default")
    user_input = data.get("input", "")

    if not user_input:
        return jsonify({"error": "입력이 필요합니다."}), 400

    prompt = f"""
    다음 문장에서 의약품 이름만 한국어 또는 영어로 하나만 추출해줘. 설명 없이 결과만 출력해.
    문장: "{user_input}"
    """

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
        return jsonify({"error": "약 이름 추출 중 오류 발생", "details": str(e)}), 500

    if not extracted_name:
        attempts[session_id] = attempts.get(session_id, 0) + 1
        if attempts[session_id] >= 2:
            return jsonify({"error": "2회 시도 실패. 처음부터 다시 입력해주세요."}), 404
        return jsonify({"error": "약 이름을 추출하지 못했습니다. 다시 입력해주세요."}), 400

    # 약 이름이 한글인지 영어인지 판별 후 쿼리
    if re.search(r'[가-힣]', extracted_name):
        query = {"itemName": {"$regex": extracted_name, "$options": "i"}}
    else:
        query = {"engName": {"$regex": extracted_name, "$options": "i"}}

    matching_docs = list(collection.find(query))

    if not matching_docs:
        attempts[session_id] = attempts.get(session_id, 0) + 1
        if attempts[session_id] >= 2:
            return jsonify({"error": "2회 시도 실패. 처음부터 다시 입력해주세요."}), 404
        return jsonify({
            "error": "관련된 약 이름을 찾지 못했습니다. 다시 시도해주세요.",
            "extracted_name": extracted_name
        }), 404

    # --------------------
    # ✅ 가중치 기반 추천 로직 적용
    # --------------------
    weights = []
    for doc in matching_docs:
        weight = float(doc.get("weight", 1.0))  # 기본값 1.0
        weights.append(max(weight, 0.01))  # 음수 방지

    # normalize weights → 확률로 변환
    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]

    # 중복 없이 최대 5개 추출
    if len(matching_docs) <= 5:
        sampled_docs = matching_docs
    else:
        sampled_indices = random.choices(
            population=list(range(len(matching_docs))),
            weights=probabilities,
            k=10  # 충분히 많이 뽑아서 중복 제거
        )
        seen = set()
        sampled_docs = []
        for idx in sampled_indices:
            if idx not in seen:
                seen.add(idx)
                sampled_docs.append(matching_docs[idx])
            if len(sampled_docs) >= 5:
                break

    # --------------------
    # ✅ 출력 구성
    # --------------------
    candidates = []
    for doc in sampled_docs:
        name_ko = (doc.get("itemName") or "").strip()
        name_en = (doc.get("engName") or "").strip()
        weight = round(float(doc.get("weight", 1.0)), 2)
        combined_name = f"{name_ko} ({name_en})" if name_en else name_ko
        candidates.append({
            "name": combined_name,
            "weight": weight
        })

    # 후보 저장
    attempts[session_id] = 0
    app.config[f"{session_id}_candidates"] = [c["name"] for c in candidates]

    return jsonify({
        "extracted_name": extracted_name,
        "candidates": candidates,
        "message": "다음 중 어떤 약이 궁금하신가요?"
    })

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=False)
