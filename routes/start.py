import uuid
from flask import Blueprint, request, jsonify, session
from services.gpt_fallback import fallback_response
from services.gpt_service import translate_to_user_lang

bp = Blueprint('start', __name__)

@bp.route('/makeSession' , methods=['POST'])
def makeSession():
    new_session_id = str(uuid.uuid4())
    session['session_id'] = new_session_id
    session['retry_count'] = 0

    return jsonify({'session_id': new_session_id})

@bp.route('/start', methods=['POST'])
def start_route():
    data = request.get_json()
    user_input = data.get("input", "").strip().lower()
    session_id = session['session_id']
    session['language'] = data.get("lang")

    if not session_id:
        session['session_id'] = str(uuid.uuid4())
        session_id = session['session_id']

    session['retry_count'] = 0
    
    if user_input == "증상" or user_input == "symptom" or user_input == "症状" or user_input == "症状" :
        return jsonify({
            "next": "/symptom",
            "message": translate_to_user_lang("어디가 아프신가요? 증상을 자세히 말씀해주세요."),
            "response_type": "start_success"
        })

    elif user_input == "약" or user_input == "medicine" or user_input == "药" or user_input == "薬":
        return jsonify({
            "next": "/name",
            "message": translate_to_user_lang("어떤 약이 궁금하신가요? 약 이름을 말해주세요."),
            "response_type": "start_success"
        })

    # 기타 입력 → fallback GPT 응답
    else :
        gpt_reply = fallback_response(user_input)
    return jsonify({
        "message": translate_to_user_lang(gpt_reply),
        "next": "/start",
        "response_type": "start_gpt_success"
    })
    
