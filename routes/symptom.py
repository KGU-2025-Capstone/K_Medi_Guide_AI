from flask import Blueprint, request, jsonify, session
from pymongo import MongoClient
from openai import OpenAI
from services.gpt_service import translate_to_user_lang
from services.utils import clean_text, softmax_with_temperature
from bs4 import BeautifulSoup
import numpy as np
from config import OPENAI_API_KEY, MONGODB_URI

#환경 및 라우트 설정
client = OpenAI(api_key=OPENAI_API_KEY)
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client['K_Medi_Guide']
collection = db['Api']
bp = Blueprint('symptom', __name__)

# 재시도 횟수를 세션에 저장
def get_retry_count():
    return session.get('retry_count', 0)


@bp.route('/symptom', methods=['POST'])
def recommend_medicine_by_symptom():
    #사용자 입력
    data = request.get_json()
    symptom_input = data.get("input", "")

    #이전 라우트 확인
    session['name_to_select'] = False

    if not symptom_input:
        return jsonify({
            "error": translate_to_user_lang("입력이 필요합니다."),
            "next": "/symptom",
            "response_type": "symptom_fail"
        }), 400

    #증상 추출 모델
    prompt = f"""
    다음 문장에서 의학적 증상 키워드만 콤마로 나열해줘. 언어는 한국어, 영어 등 다양할 수 있는데 한국어가 아닐 경우 한국어로 번역해서 나열해줘. 예시나 설명 없이 키워드만 출력해. 
    정확한 증상명이 없으면 입력 문장을 바탕으로 적절한 일반적 증상명을 의학용어 명사로 출력해. 증상에 대해 의미가 같은 여러 단어로 출력해줘
    문장: "{symptom_input}"
    """
    try:
        extract_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "의학적 증상을 추출하는 도우미야."},
                {"role": "user", "content": "몸이 으슬으슬해"},
                {"role": "assistant", "content": "오한"},
            
                {"role": "user", "content": "숨쉬기가 좀 힘들고 기운이 없어"},
                {"role": "assistant", "content": "호흡곤란, 무기력감"},
                {"role": "user", "content": "기운이 하나도 없고 계속 누워 있고 싶어"},
                {"role": "assistant", "content": "무기력감, 피로감"},
    
                {"role": "user", "content": "머리가 멍하고 집중이 잘 안돼"},
                {"role": "assistant", "content": "두통, 인지장애"},
    
                {"role": "user", "content": "가슴이 답답하고 두근거려"},
                {"role": "assistant", "content": "가슴답답함, 심계항진"},
    
                {"role": "user", "content": "배가 꾸르륵거리고 아파"},
                {"role": "assistant", "content": "복통, 복부팽만감"},
    
                {"role": "user", "content": "열이 나는 것 같고 머리도 아파"},
                {"role": "assistant", "content": "발열, 두통"},
    
                {"role": "user", "content": "콧물이 나고 재채기가 자주 나와"},
                {"role": "assistant", "content": "콧물, 재채기"},
    
                {"role": "user", "content": "몸이 근질근질하고 간지러워"},
                {"role": "assistant", "content": "가려움증"},
    
                {"role": "user", "content": "속이 더부룩하고 소화가 잘 안돼"},
                {"role": "assistant", "content": "소화불량, 복부팽만감"},

                {"role": "user", "content": "입냄새가 심해요"},
                {"role": "assistant", "content": "입냄새, 구취"},

                {"role": "user", "content": "모기에 물려서 너무 간지러운데 바를만한 약이 있나요?"},
                {"role": "assistant", "content": "벌레물린데, 가려움"},

                {"role": "user", "content": prompt}
            ],
            temperature= 0.8
        )
        symptoms_text = extract_response.choices[0].message.content.strip()
        symptoms_ko = [s.strip() for s in symptoms_text.split(",") if s.strip()]
    except Exception as e:
        return jsonify({
            "error": translate_to_user_lang("증상 추출 중 오류 발생"),
            "details": str(e),
            "next": "/symptom",
            "response_type": "symptom_fail"
        }), 500

    if not symptoms_ko:
        return jsonify({
            "error": translate_to_user_lang("증상 키워드를 추출하지 못했습니다. 더 자세한 증상을 입력해주세요."),
            "next": "/symptom",
            "response_type": "symptom_fail"
        }), 400

    #DB에서 해당 증상에 효능이 있는 약 검색
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

    #약 검색에 실패할 경우 3회 재시도 가능. 초과할 경우 처음으로 돌아감.
    if not results:
        session['retry_count'] = get_retry_count() + 1
        if get_retry_count() >= 3:
            session['retry_count'] = 0
            return jsonify({
                "error": translate_to_user_lang("3회 시도에도 약을 찾지 못했습니다. 처음으로 돌아갑니다."),
                "next": "/start",
                "response_type": "symptom_all_fail"
            }), 404
        else:
            return jsonify({
                "error": translate_to_user_lang(f"해당 증상에 맞는 약을 찾지 못했습니다. 더 자세한 증상을 입력해주세요. ({get_retry_count()}/3)"),
                "extracted_symptoms": [translate_to_user_lang(s) for s in symptoms_ko], #디버깅용 삭제 예정
                "next": "/symptom"
            }), 404

    #추출에 성공하면 증상을 사용자 세션에 저장
    session['symptoms_ko'] = symptoms_ko

    #가중치를 고려하여 softmax알고리즘에 따라 약을 선택
    weights = [float(r.get("weight", 1.0)) for r in results]
    probabilities = softmax_with_temperature(weights, temperature=1.0)
    sampled = np.random.choice(results, size=min(5, len(results)), replace=False, p=probabilities)

    #약 후보 목록을 정리
    candidates = []
    for r in sampled:
        name_ko = r.get("itemName", "")
        name_en = r.get("engName", "")
        combined_name = f"{name_ko} ({name_en})" if name_en else name_ko
        efcy_raw = clean_text(r.get("efcyQesitm", ""))
        translated_efcy = translate_to_user_lang(efcy_raw)

        candidates.append({
            "itemName": combined_name,
            "name_ko": name_ko,
            "efcyQesitm": translated_efcy
        })

    #시도 횟수 초기화 및 정보 반환
    session['retry_count'] = 0
    return jsonify({
        "medicine_candidates": candidates,
        "message": translate_to_user_lang("다음 중 어떤 약이 궁금하신가요?"),
        "next": "/select",
        "extracted_symptoms": [translate_to_user_lang(s) for s in symptoms_ko], #디버깅용 삭제 예정
        "response_type": "symptom_success"
    })
