from flask import Blueprint, request, jsonify
from pymongo import MongoClient
from openai import OpenAI
from services.session_store import session_results
from services.gpt_service import translate_to_user_lang
from services.utils import clean_text, trim_to_token_limit, detect_language
from concurrent.futures import ThreadPoolExecutor
from config import OPENAI_API_KEY, MONGODB_URI, FINE_TUNE_USEMETHOD_MODEL, FINE_TUNE_ATPN_MODEL
import re

client = OpenAI(api_key=OPENAI_API_KEY)
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client['K_Medi_Guide']
collection = db['Api']
session_results = {}

bp = Blueprint('detail', __name__)

@bp.route('/detail', methods=['POST'])
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
            messages=[
                {"role": "system", "content": "다국어 응답을 YES 또는 NO로만 판단해줘."},
                {"role": "user", "content": prompt}
            ]
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
                model=FINE_TUNE_USEMETHOD_MODEL,
                messages=[{"role": "user", "content": use_text}],
                max_tokens=200,
                temperature=0.7
            ).choices[0].message.content.strip())

            future_atpn = executor.submit(lambda: client.chat.completions.create(
                model=FINE_TUNE_ATPN_MODEL,
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
