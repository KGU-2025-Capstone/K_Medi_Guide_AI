from flask import Blueprint, request, jsonify
from pymongo import MongoClient
from openai import OpenAI
from services.session_store import session_symptoms, session_results, session_lang
from services.gpt_service import translate_to_user_lang
from services.utils import clean_text
from concurrent.futures import ThreadPoolExecutor
from config import OPENAI_API_KEY, MONGODB_URI, FINE_TUNE_SYMPTOM_MODEL, FINE_TUNE_EFCY_MODEL
import re

client = OpenAI(api_key=OPENAI_API_KEY)
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client['K_Medi_Guide']
collection = db['Api']

session_symptoms = {}
session_results = {}
session_lang = {}

bp = Blueprint('select', __name__)

@bp.route("/select", methods=["POST"])
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
                model=FINE_TUNE_SYMPTOM_MODEL,
                messages=[{"role": "user", "content": ", ".join(symptoms_ko)}],
                max_tokens=60,
                temperature=0.7
            ).choices[0].message.content.strip())

            future_efcy = executor.submit(lambda: client.chat.completions.create(
                model=FINE_TUNE_EFCY_MODEL,
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
