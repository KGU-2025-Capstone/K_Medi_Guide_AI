from flask import Flask, request, jsonify
from pymongo import MongoClient
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
import numpy as np
import tiktoken
import random
import os
import re

# 환경변수 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
mongo_uri = os.getenv("MONGODB_URI")

client = OpenAI(api_key=openai_api_key)
app = Flask(__name__)

fine_tune_symptom_model = os.getenv("FINE_TUNE_SYMPTOM_MODEL")
fine_tune_efcy_model = os.getenv("FINE_TUNE_EFCY_MODEL")
fine_tune_usemethod_model = os.getenv("FINE_TUNE_USEMETHOD_MODEL")
fine_tune_atpn_model = os.getenv("FINE_TUNE_ATPN_MODEL")

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
session_lang = {}

def detect_language(text):
    if re.search(r"[ㄱ-ㅎㅏ-ㅣ가-힣]", text):
        return "ko"
    elif re.search(r"[A-Za-z]", text):
        return "en"
    elif re.search(r"[\u3040-\u30ff]", text):  # 히라가나 + 가타카나
        return "ja"
    elif re.search(r"[\u4e00-\u9fff]", text):  # 한자 (중국어)
        return "zh"
    else:
        return "ko"  # 기본값은 한국어


# 텍스트 클리너
def clean_text(html):
    return BeautifulSoup(html or "", "html.parser").get_text(strip=True).replace("\n", " ").replace("\r", " ")

# 소프트맥스 with temperature
def softmax_with_temperature(weights, temperature):
    weights = np.array(weights)
    scaled = weights / temperature
    exp_scaled = np.exp(scaled - np.max(scaled))
    return exp_scaled / np.sum(exp_scaled)

def trim_to_token_limit(text, max_tokens, model="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_text = enc.decode(tokens[:max_tokens])
    if "." in truncated_text:
        last_period = truncated_text.rfind(".")
        return truncated_text[:last_period+1].strip()
    else:
        return truncated_text.strip()

# 번역 함수
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

@app.route('/medicine/symptom', methods=['POST'])
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


@app.route("/medicine/select", methods=["POST"])
def select_medicine():
    data = request.get_json()
    session_id = data.get("session_id", "default")
    selected_name = data.get("selected_item", "").strip()

    if not selected_name:
        return jsonify({"error": translate_to_user_lang(session_id, "선택한 약 이름이 필요합니다.")}), 400

    result = collection.find_one({"itemName": {"$regex": re.escape(selected_name), "$options": "i"}})
    if not result:
        return jsonify({"error": translate_to_user_lang(session_id, f"'{selected_name}' 이름의 약을 찾을 수 없습니다.")}), 404

    name_ko = result.get("itemName", "")
    name_en = result.get("engName", "")
    combined_name = f"{name_ko}({name_en})" if name_en else name_ko
    symptoms_ko = session_symptoms.get(session_id, [])

    current_weight = float(result.get("weight", 1.0))
    collection.update_one({"_id": result["_id"]}, {"$set": {"weight": round(current_weight + 0.5, 2)}})

    efcy_raw = clean_text(result.get("efcyQesitm", ""))
    if efcy_raw.startswith("이 약은"):
        efcy_raw = efcy_raw[4:]

    try:
        with ThreadPoolExecutor() as executor:
            future_symptom = executor.submit(lambda: client.chat.completions.create(
                model=fine_tune_symptom_model,
                messages=[{"role": "user", "content": ", ".join(symptoms_ko)}],
                max_tokens=60,
                temperature=0.7
            ).choices[0].message.content.strip())

            future_efcy = executor.submit(lambda: client.chat.completions.create(
                model=fine_tune_efcy_model,
                messages=[{"role": "user", "content": efcy_raw}],
                max_tokens=100,
                temperature=0.7
            ).choices[0].message.content.strip())

        symptom_response = future_symptom.result()
        efcy_response = future_efcy.result()

    except Exception as e:
        return jsonify({"error": translate_to_user_lang(session_id, "GPT 호출 중 오류 발생"), "details": str(e)}), 500

    session_results[session_id] = {"itemName": name_ko, "engName": name_en}
    final_message = f"{symptom_response} {combined_name}은(는) {efcy_response}"

    return jsonify({
        "itemName": combined_name,
        "message": translate_to_user_lang(session_id, final_message),
        "addMessage": translate_to_user_lang(session_id, "복용법과 주의사항도 알려드릴까요?"),
        "symptoms": [translate_to_user_lang(session_id, s) for s in symptoms_ko]
    })


@app.route('/medicine/detail', methods=['POST'])
def provide_medicine_details():
    data = request.get_json()
    session_id = data.get("session_id", "default")
    user_reply = data.get("reply", "").strip()

    if not user_reply:
        return jsonify({"error": translate_to_user_lang(session_id, "사용자 응답이 필요합니다.")}), 400

    prompt = f"""
    사용자가 약에 대한 복용법과 주의사항을 더 알고 싶어하는지 판단해줘.
    언어는 한국어, 영어, 중국어, 일본어일 수 있어. 긍정이면 "YES", 아니면 "NO"만 대답해.
    사용자 응답: "{user_reply}"
    """
    try:
        answer = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "다국어 응답을 YES 또는 NO로만 판단해줘."}, {"role": "user", "content": prompt}]
        ).choices[0].message.content.strip().upper()

        if "YES" not in answer:
            return jsonify({"message": translate_to_user_lang(session_id, "알겠습니다. 복용법과 주의사항은 생략할게요.")})

        result = session_results.get(session_id)
        if not result:
            return jsonify({"error": translate_to_user_lang(session_id, "세션에 저장된 약 정보가 없습니다.")}), 404

        item_name = result["itemName"]
        name_en = result.get("engName", "")
        combined_name = f"{item_name}({name_en})" if name_en else item_name
        original = collection.find_one({"itemName": {"$regex": re.escape(item_name), "$options": "i"}})
        if not original:
            return jsonify({"error": translate_to_user_lang(session_id, f"'{item_name}' 약 정보를 DB에서 찾을 수 없습니다.")}), 404

        use_text = clean_text(original.get("useMethodQesitm", ""))
        atpn_text = clean_text(original.get("atpnQesitm", ""))
        for prefix in ["이 약은", "이 약을", "이 약에"]:
            if atpn_text.startswith(prefix):
                atpn_text = atpn_text[len(prefix):].strip()
                break
        use_text = trim_to_token_limit(use_text, max_tokens=200)
        atpn_text = trim_to_token_limit(atpn_text, max_tokens=300)

        with ThreadPoolExecutor() as executor:
            future_use = executor.submit(lambda: client.chat.completions.create(
                model=fine_tune_usemethod_model,
                messages=[{"role": "user", "content": use_text}],
                max_tokens=200,
                temperature=0.7
            ).choices[0].message.content.strip())

            future_atpn = executor.submit(lambda: client.chat.completions.create(
                model=fine_tune_atpn_model,
                messages=[{"role": "user", "content": atpn_text}],
                max_tokens=300,
                temperature=0.7
            ).choices[0].message.content.strip())

        use_response = future_use.result()
        atpn_response = future_atpn.result()

        final_message = f"{combined_name}은(는) {use_response} {atpn_response}"

        return jsonify({
            "itemName": item_name,
            "detailMessage": translate_to_user_lang(session_id, final_message)
        })

    except Exception as e:
        return jsonify({"error": translate_to_user_lang(session_id, "GPT 호출 중 오류 발생"), "details": str(e)}), 500


@app.route('/medicine/name', methods=['POST'])
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
            messages=[{"role": "system", "content": "약 이름을 추출하는 도우미야."}, {"role": "user", "content": prompt}]
        )
        extracted_name = response.choices[0].message.content.strip()
    except Exception as e:
        return jsonify({"error": translate_to_user_lang(session_id, "약 이름 추출 중 오류 발생"), "details": str(e)}), 500

    query = {"itemName": {"$regex": extracted_name, "$options": "i"}} if re.search(r'[가-힣]', extracted_name) else {"engName": {"$regex": extracted_name, "$options": "i"}}
    matching_docs = list(collection.find(query))
    if not matching_docs:
        attempts[session_id] = attempts.get(session_id, 0) + 1
        if attempts[session_id] >= 2:
            return jsonify({"error": translate_to_user_lang(session_id, "2회 시도 실패. 처음부터 다시 입력해주세요.")}), 404
        return jsonify({"error": translate_to_user_lang(session_id, "관련된 약 이름을 찾지 못했습니다."), "extracted_name": extracted_name}), 404

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
        "message": translate_to_user_lang(session_id, "다음 중 어떤 약이 궁금하신가요?")
    })

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=False)
