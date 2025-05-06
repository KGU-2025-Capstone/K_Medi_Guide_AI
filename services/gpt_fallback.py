from collections import deque
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

MAX_HISTORY = 5
chat_history = deque(maxlen=MAX_HISTORY)

# DB에서 약물의 모든 정보 조회 함수
def get_medication_info(med_name):
    # 약물 이름을 바탕으로 모든 정보 찾기
    result = collection.find_one({"itemName": {"$regex": re.escape(med_name), "$options": "i"}})
    if result:
        return result  # 모든 필드 반환
    else:
        return None  # 정보가 없으면 None 반환

# 사용자의 질문에 맞게 정보를 동적으로 답변 생성하는 함수
def fallback_response(user_input):
    # 사용자가 묻는 약물명 추출 (예: "타이레놀")
    med_name = extract_medcine_name(user_input)

    # DB에서 해당 약물의 모든 정보 가져오기
    medication_info = get_medication_info(med_name)
    if medication_info:
        # DB에서 가져온 정보를 자연스럽게 하나의 문장으로 변환하여 GPT에게 제공
        context = f"약물명: {medication_info.get('itemName', '정보 없음')}\n"
        context += f"효능: {medication_info.get('efcyQesitm', '정보 없음')}\n"
        context += f"복용법: {medication_info.get('useMethodQesitm', '정보 없음')}\n"
        context += f"주의사항: {medication_info.get('atpnQesitm', '정보 없음')}\n"

        # 이 문맥을 GPT 모델에 제공하여 답변을 도출하게 함
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
        context += f"\n이전 대화:\n{history_text}\n\n이전 대화를 바탕으로 다음 질문에 대답하세요. 질문: {user_input}"

        # OpenAI API로 메시지 전송
        messages = [
            {"role": "system", "content": "당신은 한국에 거주하는 외국인을 도와주는 친절한 약국 상담 챗봇입니다."},
            {"role": "user", "content": context}
        ]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        answer = response.choices[0].message.content.strip()

        # 현재 사용자 질문과 답변을 대화 기록에 추가
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": answer})

        return answer
    else:
        return "말씀하신 내용을 잘 이해하지 못했어요."
    
# 이전 대화 기록을 초기화하는 함수 (필요한 경우 호출)
def clear_chat_history():
    chat_history.clear()

# 현재 대화 기록 확인 (디버깅용)
def get_current_chat_history():
    return list(chat_history)
