from flask import Blueprint, request, jsonify, session
from pymongo import MongoClient
from openai import OpenAI
from services.gpt_service import translate_to_user_lang
from services.utils import clean_text, trim_to_token_limit
from concurrent.futures import ThreadPoolExecutor
from config import OPENAI_API_KEY, MONGODB_URI, FINE_TUNE_USEMETHOD_MODEL, FINE_TUNE_ATPN_MODEL
import re

#환경 및 라우트 설정
client = OpenAI(api_key=OPENAI_API_KEY)
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client['K_Medi_Guide']
collection = db['Api']
bp = Blueprint('detail', __name__)

@bp.route('/detail', methods=['POST'])
def provide_medicine_details():
    #사용자 입력
    data = request.get_json()
    user_reply = data.get("input", "").strip()

    if not user_reply:
        return jsonify({"error": translate_to_user_lang("사용자 응답이 필요합니다."), "next": "/detail", "response_type": "detail_fail"}), 400

    #복용법 및 주의사항 출력 여부 확인 모델
    # prompt = f"""
    # 사용자가 약에 대한 복용법과 주의사항을 더 알고 싶어하는지 판단해줘.
    # 주어진 값이 YES 면 알고싶어하는거고 NO 이면 알고싶지 않은거야 "{user_reply}"
    # """
    try:
        # answer = client.chat.completions.create(
        #     model="gpt-3.5-turbo",
        #     messages=[
        #         {"role": "system", "content": "다국어 응답을 YES 또는 NO로만 판단해줘."},
        #         {"role": "user", "content": prompt}
        #     ]
        # ).choices[0].message.content.strip().upper()

        if user_reply == "NO":
            return jsonify({"message": translate_to_user_lang("알겠습니다. 복용법과 주의사항은 생략할게요."),
                            "next": "/start",
                            "addMessage": translate_to_user_lang("더 궁금한 게 있으신가요?"),
                            "response_type": "detail_fail"})

        result = session.get('results')
        if not result:
            return jsonify({"error": translate_to_user_lang("저장된 약 정보가 없습니다."), "next": "/start", "response_type": "detail_fail"}), 404

        item_name = result["itemName"]
        name_en = result.get("engName", "")
        combined_name = f"{item_name}({name_en})" if name_en else item_name

        original = collection.find_one({"itemName": {"$regex": re.escape(item_name), "$options": "i"}})
        if not original:
            return jsonify({"error": translate_to_user_lang(f"'{item_name}'에 대한 정보를 찾을 수 없습니다."), "next": "/start", "response_type": "detail_fail"}), 404

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

        final_message = f"💊{combined_name}\n{use_response}\n{atpn_response}"

        return jsonify({
            "detailMessage": translate_to_user_lang(final_message),
            "addMessage": translate_to_user_lang("더 궁금한 게 있으신가요?"),
            "next": "/start",
            "response_type": "detail_success"
        })

    except Exception as e:
        return jsonify({"error": translate_to_user_lang("챗봇 호출 중 오류 발생"), "details": str(e), "next": "/start", "response_type": "detail_fail"}), 500
