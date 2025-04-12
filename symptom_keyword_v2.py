from flask import Flask, request, jsonify
from pymongo import MongoClient
from openai import OpenAI
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from collections import OrderedDict
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
session_medicine_docs = {}

def softmax_with_temperature(weights, temperature):
    weights = np.array(weights, dtype=np.float64)
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

    # 세션 저장
    session_symptoms[session_id] = symptoms
    session_medicine_docs[session_id] = results

    # 가중치 기반 샘플링 처리
    if session_id not in session_data:
        session_data[session_id] = {
            "weights": {r['itemName']: float(r.get("weight", 1.0)) for r in results},
            "count": 0
        }

    session_data[session_id]["count"] += 1
    count = session_data[session_id]["count"]

    temperature = max(0.3, 2.0 / (count ** 0.5))
    item_names = list(session_data[session_id]["weights"].keys())
    weights = [session_data[session_id]["weights"][name] for name in item_names]
    probabilities = softmax_with_temperature(weights, temperature)

    # 최대 5개 중복 없이 추천
    k = min(5, len(item_names))
    chosen_indices = np.random.choice(len(item_names), size=k, replace=False, p=probabilities)
    chosen_names = [item_names[i] for i in chosen_indices]
    chosen_results = [r for r in results if r['itemName'] in chosen_names]

    candidates = []
    for r in chosen_results:
        name_ko = (r.get("itemName") or "").strip()
        name_en = (r.get("engName") or "").strip()
        combined_name = f"{name_ko} ({name_en})" if name_en else name_ko
        weight = float(r.get("weight", 1.0))

        candidates.append({
            "itemName": combined_name,
            "efcyQesitm": BeautifulSoup(r.get("efcyQesitm", ""), "html.parser").get_text(strip=True, separator=' '),
            "weight": weight
        })

    retry_count[session_id] = 0

    return jsonify({
        "extracted_symptoms": symptoms,
        "temperature": round(temperature, 2),
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

    result = collection.find_one({
        "itemName": {
            "$regex": re.escape(selected_name),
            "$options": "i"
        }
    })

    if not result:
        return jsonify({"error": f"'{selected_name}' 이름의 약을 찾을 수 없습니다."}), 404

    # weight 증가
    current_weight = float(result.get("weight", 1.0))
    updated_weight = round(current_weight + 0.5, 2)
    collection.update_one({"_id": result["_id"]}, {"$set": {"weight": updated_weight}})

    symptoms = session_symptoms.get(session_id, [])
    name_ko = result.get("itemName", "")
    name_en = result.get("engName", "")
    combined_name = f"{name_ko} ({name_en})" if name_en else name_ko

    session_results[session_id] = {
        "itemName": combined_name,
        "itemImage": result.get("itemImage", "정보 없음"),
        "symptoms": symptoms,
        "efcyQesitm": BeautifulSoup(result.get("efcyQesitm", ""), 'html.parser').get_text(strip=True, separator=' '),
        "useMethodQesitm": BeautifulSoup(result.get("useMethodQesitm", ""), 'html.parser').get_text(strip=True, separator=' '),
        "atpnQesitm": BeautifulSoup(result.get("atpnQesitm", ""), 'html.parser').get_text(strip=True, separator=' '),
        "weight": updated_weight
    }

    response = OrderedDict([
        ("itemName", combined_name),
        ("weight", updated_weight),
        ("efcyQesitm", session_results[session_id]["efcyQesitm"]),
        ("itemImage", session_results[session_id]["itemImage"]),
        ("message", "복용법과 주의사항도 알려드릴까요?"),
        ("symptoms", symptoms)
    ])
    return jsonify(response)

@app.route('/medicine/detail', methods=['POST'])
def provide_medicine_details():
    data = request.get_json()
    session_id = data.get("session_id", "default")
    user_reply = data.get("reply", "").strip()

    if not user_reply:
        return jsonify({"error": "사용자 응답이 필요합니다."}), 400

    prompt = f"""
    사용자가 약에 대한 복용법과 주의사항을 더 알고 싶어하는지 판단해줘.
    언어는 한국어, 영어, 중국어, 일본어일 수 있어. 
    긍정이면 "YES", 아니면 "NO"만 대답해.
    사용자 응답: "{user_reply}"
    """

    try:
        gpt_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "너는 다국어 응답을 이해하고 긍정 여부를 YES 또는 NO로만 판단해주는 도우미야."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = gpt_response.choices[0].message.content.strip().upper()

        if answer == "YES":
            result = session_results.get(session_id)
            if not result:
                return jsonify({"error": "세션에 저장된 약 정보가 없습니다."}), 404

            return jsonify({
                "itemName": result["itemName"],
                "useMethodQesitm": result["useMethodQesitm"],
                "atpnQesitm": result["atpnQesitm"]
            })
        else:
            return jsonify({
                "message": "알겠습니다. 복용법과 주의사항은 생략할게요."
            })

    except Exception as e:
        return jsonify({
            "error": "GPT 판단 중 오류가 발생했습니다.",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=False)
