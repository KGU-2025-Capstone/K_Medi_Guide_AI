from flask import Flask, request, jsonify
from pymongo import MongoClient
from openai import OpenAI
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import numpy as np
import random
import os
import re

# 환경변수 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
mongo_uri = os.getenv("MONGODB_URI")

client = OpenAI(api_key=openai_api_key)
app = Flask(__name__)

# MongoDB 연결
mongo_client = MongoClient(mongo_uri)
db = mongo_client['K_Medi_Guide']
collection = db['Api']

# 세션 상태 저장
retry_count = {}
session_data = {}
session_symptoms = {}
session_results = {}
attempts = {}

# 텍스트 클리너
def clean_text(html):
    return BeautifulSoup(html or "", "html.parser").get_text(strip=True).replace("\n", " ").replace("\r", " ")

# 소프트맥스 with temperature
def softmax_with_temperature(weights, temperature):
    weights = np.array(weights)
    scaled = weights / temperature
    exp_scaled = np.exp(scaled - np.max(scaled))
    return exp_scaled / np.sum(exp_scaled)

@app.route('/medicine/symptom', methods=['POST'])
def recommend_medicine_by_symptom():
    data = request.get_json()
    session_id = data.get("session_id", "default")
    symptom_input = data.get("input", "")

    if not symptom_input:
        return jsonify({"error": "입력이 필요합니다."}), 400

    # 증상 키워드 추출
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
        symptoms = [s.strip() for s in symptoms_text.split(",") if s.strip()]
    except Exception as e:
        return jsonify({"error": "증상 추출 중 오류 발생", "details": str(e)}), 500

    if not symptoms:
        return jsonify({"error": "증상 키워드를 추출하지 못했습니다."}), 400

    # HTML 제거 후 증상 포함 약 필터링
    results = []
    seen_ids = set()
    for doc in collection.find({}):
        efcy_html = doc.get("efcyQesitm", "")
        plain_efcy = BeautifulSoup(efcy_html, "html.parser").get_text()
        for symptom in symptoms:
            if re.search(rf"\b{re.escape(symptom)}\b", plain_efcy):
                _id = str(doc.get("_id"))
                if _id not in seen_ids:
                    results.append(doc)
                    seen_ids.add(_id)
                break

    if not results:
        retry_count[session_id] = retry_count.get(session_id, 0) + 1
        if retry_count[session_id] >= 3:
            return jsonify({"error": "3회 시도에도 약을 찾지 못했습니다. 처음으로 돌아갑니다."}), 404
        else:
            return jsonify({
                "error": f"해당 증상에 맞는 약을 찾지 못했습니다. 다시 입력해주세요. ({retry_count[session_id]}/3)",
                "extracted_symptoms": symptoms
            }), 404

    session_symptoms[session_id] = symptoms

    # 가중치 기반 softmax 샘플링
    weights = [float(r.get("weight", 1.0)) for r in results]
    probabilities = softmax_with_temperature(weights, temperature=1.0)
    sampled = np.random.choice(results, size=min(5, len(results)), replace=False, p=probabilities)

    candidates = []
    for r in sampled:
        name_ko = r.get("itemName", "")
        name_en = r.get("engName", "")
        combined_name = f"{name_ko} ({name_en})" if name_en else name_ko
        candidates.append({
            "itemName": combined_name,
            "efcyQesitm": clean_text(r.get("efcyQesitm", "")),
            "weight": float(r.get("weight", 1.0))
        })

    retry_count[session_id] = 0
    return jsonify({
        "extracted_symptoms": symptoms,
        "medicine_candidates": candidates,
        "message": "다음 중 어떤 약이 궁금하신가요?"
    })

@app.route('/medicine/select', methods=['POST'])
def select_medicine():
    data = request.get_json()
    session_id = data.get("session_id", "default")
    selected_name = data.get("selected_item", "").strip()

    if not selected_name:
        return jsonify({"error": "선택한 약 이름이 필요합니다."}), 400

    result = collection.find_one({"itemName": {"$regex": re.escape(selected_name), "$options": "i"}})
    if not result:
        return jsonify({"error": f"'{selected_name}' 이름의 약을 찾을 수 없습니다."}), 404

    symptoms = session_symptoms.get(session_id, [])
    name_ko = result.get("itemName", "")
    name_en = result.get("engName", "")
    combined_name = f"{name_ko} ({name_en})" if name_en else name_ko

    # 가중치 증가 후 DB 반영
    current_weight = float(result.get("weight", 1.0))
    new_weight = round(current_weight + 0.5, 2)
    collection.update_one({"_id": result["_id"]}, {"$set": {"weight": new_weight}})

    session_results[session_id] = {
        "itemName": combined_name,
        "itemImage": result.get("itemImage", "정보 없음"),
        "symptoms": symptoms,
        "efcyQesitm": clean_text(result.get("efcyQesitm", "")),
        "useMethodQesitm": clean_text(result.get("useMethodQesitm", "")),
        "atpnQesitm": clean_text(result.get("atpnQesitm", "")),
        "weight": new_weight
    }

    return jsonify({
        "itemName": combined_name,
        "efcyQesitm": session_results[session_id]["efcyQesitm"],
        "itemImage": session_results[session_id]["itemImage"],
        "weight": new_weight,
        "message": "복용법과 주의사항도 알려드릴까요?",
        "symptoms": symptoms
    })

@app.route('/medicine/detail', methods=['POST'])
def provide_medicine_details():
    data = request.get_json()
    session_id = data.get("session_id", "default")
    user_reply = data.get("reply", "").strip()

    if not user_reply:
        return jsonify({"error": "사용자 응답이 필요합니다."}), 400

    prompt = f"""
    사용자가 약에 대한 복용법과 주의사항을 더 알고 싶어하는지 판단해줘.
    언어는 한국어, 영어, 중국어, 일본어일 수 있어. 긍정이면 "YES", 아니면 "NO"만 대답해.
    사용자 응답: "{user_reply}"
    """

    try:
        gpt_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "다국어 응답을 이해하고 긍정 여부를 YES 또는 NO로만 판단해줘."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = gpt_response.choices[0].message.content.strip().upper()
        if "YES" in answer:
            result = session_results.get(session_id)
            if not result:
                return jsonify({"error": "세션에 저장된 약 정보가 없습니다."}), 404
            return jsonify({
                "itemName": result["itemName"],
                "useMethodQesitm": result["useMethodQesitm"],
                "atpnQesitm": result["atpnQesitm"]
            })
        else:
            return jsonify({"message": "알겠습니다. 복용법과 주의사항은 생략할게요."})
    except Exception as e:
        return jsonify({"error": "판단 중 오류가 발생했습니다.", "details": str(e)}), 500

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

    if re.search(r'[가-힣]', extracted_name):
        query = {"itemName": {"$regex": extracted_name, "$options": "i"}}
    else:
        query = {"engName": {"$regex": extracted_name, "$options": "i"}}

    matching_docs = list(collection.find(query))
    if not matching_docs:
        attempts[session_id] = attempts.get(session_id, 0) + 1
        if attempts[session_id] >= 2:
            return jsonify({"error": "2회 시도 실패. 처음부터 다시 입력해주세요."}), 404
        return jsonify({"error": "관련된 약 이름을 찾지 못했습니다.", "extracted_name": extracted_name}), 404

    weights = [float(doc.get("weight", 1.0)) for doc in matching_docs]
    probabilities = softmax_with_temperature(weights, temperature=1.0)
    sampled = np.random.choice(matching_docs, size=min(5, len(matching_docs)), replace=False, p=probabilities)

    candidates = []
    for doc in sampled:
        name_ko = doc.get("itemName", "")
        name_en = doc.get("engName", "")
        combined_name = f"{name_ko} ({name_en})" if name_en else name_ko
        candidates.append({
            "itemName": combined_name,
            "weight": float(doc.get("weight", 1.0))
        })

    attempts[session_id] = 0
    return jsonify({
        "extracted_name": extracted_name,
        "candidates": candidates,
        "message": "다음 중 어떤 약이 궁금하신가요?"
    })

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=False)
