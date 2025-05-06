from flask import Blueprint, request, jsonify, session
from services.gpt_fallback import fallback_response
from services.session_store import session_lang
from services.gpt_service import translate_to_user_lang
from services.utils import detect_language

bp = Blueprint('start', __name__)

@bp.route('/start', methods=['POST'])
def start_route():
    data = request.get_json()
    session_id = data.get("session_id", "default")
    user_input = data.get("input", "").strip().lower()

    session_lang[session_id] = detect_language(user_input)

    if user_input == "증상" or user_input == "symptom":
        return jsonify({
            "next": "/symptom",
            "message": translate_to_user_lang(session_id, "어디가 아프신가요? 증상을 자세히 말씀해주세요."),
            "response_type": "start_success"
        })

    elif user_input == "약" or user_input == "약명" or user_input == "name":
        return jsonify({
            "next": "/name",
            "message": translate_to_user_lang(session_id, "어떤 약이 궁금하신가요? 약 이름을 말해주세요."),
            "response_type": "start_success"
        })

    # 기타 입력 → fallback GPT 응답
    gpt_reply = fallback_response(session_id, user_input)
    return jsonify({
        "message": translate_to_user_lang(session_id, gpt_reply),
        "next": "/start",
        "response_type": "start_gpt_success"
    })
