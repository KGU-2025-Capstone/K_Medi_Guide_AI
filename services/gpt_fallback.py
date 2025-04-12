from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def fallback_response(session_id, user_input):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 한국에 거주하는 외국인을 도와주는 친절한 약국 상담 챗봇입니다. 일반의약품에 대한 정보만 답하세요. 주제에 벗어난 질문에는 잘 모르겠다고 답하세요. "},
                {"role": "user", "content": user_input}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "죄송합니다. 잠시 문제가 발생했어요."
