from flask import Blueprint, request, jsonify, session
from pymongo import MongoClient
from openai import OpenAI
from services.gpt_service import translate_to_user_lang, replace_translated_name, improved_readability
from services.utils import clean_text
from concurrent.futures import ThreadPoolExecutor
from config import OPENAI_API_KEY, MONGODB_URI, FINE_TUNE_SYMPTOM_MODEL, PURE_FINE_TUNE_EFCY_MODEL
import re

#환경 및 라우트 설정
client = OpenAI(api_key=OPENAI_API_KEY)
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client['K_Medi_Guide']
collection = db['Api']
bp = Blueprint('select', __name__)

@bp.route("/select", methods=["POST"])
def select_medicine():
    #사용자 입력
    data = request.get_json()
    selected_name = data.get("input", "").strip()

    if not selected_name:
        return jsonify({"error": translate_to_user_lang("선택한 약 이름이 필요합니다."),
                        "next": "/start",
                        "response_type": "select_fail"}), 400

    #선택한 약의 정보를 DB에서 검색
    result = collection.find_one({"itemName": {"$regex": re.escape(selected_name), "$options": "i"}})
    if not result:
        return jsonify({"error": translate_to_user_lang(f"'{selected_name}' 이름의 약을 찾을 수 없습니다."),
                        "next": "/start",
                        "response_type": "select_fail"}), 404

    #약의 이름, 사용자 증상을 검색해서 저장
    name_ko = result.get("itemName", "")
    name_en = result.get("engName", "")
    combined_name = f"{name_ko}({name_en})" if name_en else name_ko
    session['combined_name'] = combined_name
    symptoms_ko = session.get('symptoms_ko')

    #선택한 약의 가중치를 업데이트
    current_weight = float(result.get("weight", 1.0))
    collection.update_one({"_id": result["_id"]}, {"$set": {"weight": round(current_weight + 0.5, 2)}})

    #효능 데이터 가공
    efcy_raw = clean_text(result.get("efcyQesitm", ""))
    if efcy_raw.startswith("이 약은"):
        efcy_raw = efcy_raw[4:]

    #증상, 효능 문장 생성 모델 병렬처리
    try:
        with ThreadPoolExecutor() as executor:
            if symptoms_ko and isinstance(symptoms_ko, list):
                future_symptom = executor.submit(lambda: client.chat.completions.create(
                model=FINE_TUNE_SYMPTOM_MODEL,
                messages=[{"role": "user", "content": ", ".join(symptoms_ko)}],
                max_tokens=60,
                temperature=0.8
            ).choices[0].message.content.strip())
            
            else:
                future_symptom = None

            future_efcy = executor.submit(lambda: client.chat.completions.create(
                model=PURE_FINE_TUNE_EFCY_MODEL,
                messages=[{"role": "user", "content": efcy_raw}],
                max_tokens=250,
                temperature=0.8
            ).choices[0].message.content.strip())
        symptom_response = future_symptom.result() if future_symptom else ""
        efcy_response = future_efcy.result()
    except Exception as e:
        return jsonify({"error": translate_to_user_lang("챗봇 호출 중 오류 발생"), "details": str(e),"next": "/start", "response_type": "select_fail"}), 500

    #약 정보를 사용자 세션에 저장
    session['results'] = {"itemName": name_ko, "engName": name_en}
    
    #이전 라우트가 name이었는지 확인, 최종 출력 메시지 가공
    if session.get('name_to_select') is True:
        final_message = f"{combined_name}은(는) {efcy_response}"
        insert_text = f"<<약이름>>"
    else:
        final_message = f"{symptom_response} {combined_name}은(는) {efcy_response}"
        insert_text = f"{translate_to_user_lang(symptom_response)} <<약이름>>"

    final_message = translate_to_user_lang(final_message)
    final_message = replace_translated_name(final_message, insert_text)
    final_message = improved_readability(final_message)

    #정보 반환
    return jsonify({
        "message": final_message,
        "addMessage": translate_to_user_lang("복용법과 주의사항도 알려드릴까요?"),
        "next": "/detail",
        "response_type": "select_success"
    })
