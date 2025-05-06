from openai import OpenAI
from config import OPENAI_API_KEY
from config import MONGODB_URI
from pymongo import MongoClient
from services.gpt_service import extract_medcine_name
import re

client = OpenAI(api_key=OPENAI_API_KEY)
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client['K_Medi_Guide']
collection = db['Api']

# DB에서 약물의 모든 정보 조회 함수
def get_medication_info(med_name):
    # 약물 이름을 바탕으로 모든 정보 찾기
    result = collection.find_one({"itemName": {"$regex": re.escape(med_name), "$options": "i"}})
    if result:
        return result  # 모든 필드 반환
    else:
        return None  # 정보가 없으면 None 반환

# 사용자의 질문에 맞게 정보를 동적으로 답변 생성하는 함수
def fallback_response(session_id, user_input):
    # 사용자가 묻는 약물명 추출 (예: "타이레놀")
    med_name = extract_medcine_name(session_id, user_input)

    # DB에서 해당 약물의 모든 정보 가져오기
    medication_info = get_medication_info(med_name)
    if medication_info:
        # DB에서 가져온 정보를 자연스럽게 하나의 문장으로 변환하여 GPT에게 제공
        context = f"약물명: {medication_info.get('engName', '정보 없음')}\n"
        context += f"효능: {medication_info.get('efcyQesitm', '정보 없음')}\n"
        context += f"복용법: {medication_info.get('useMethodQesitm', '정보 없음')}\n"
        context += f"주의사항: {medication_info.get('atpnQesitm', '정보 없음')}\n"

        # 이 문맥을 GPT 모델에 제공하여 답변을 도출하게 함
        context += f"\n위 정보를 바탕으로 아래 질문에 대답해주세요: {user_input}"

        # OpenAI API로 메시지 전송
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 한국에 거주하는 외국인을 도와주는 친절한 약국 상담 챗봇입니다. 일반의약품에 대한 정보만 답하세요. 주제에 벗어난 질문에는 잘 모르겠다고 답하세요."},
                {"role": "user", "content": context}
            ]
        )

        return response.choices[0].message.content.strip()
    else:
        return "해당 약물에 대한 정보가 없습니다."
