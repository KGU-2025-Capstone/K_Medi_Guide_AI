# K-Medi-Guide AI

**K-Medi-Guide AI**는 증상 기반으로 의약품을 추천하고, 사용법/주의사항을 GPT 기반으로 요약 제공하는 다국어 지원 메디컬 챗봇 API입니다.

## 🏗 프로젝트 구조

```
project/
├── app.py
├── config.py
├── requirements.txt
├── .env
├── routes/
│   ├── symptom.py
│   ├── select.py
│   ├── detail.py
│   └── name.py
├── services/
│   ├── gpt_service.py
│   └── utils.py
└── data/
    └── sample_data.json
```

## 🚀 실행 방법

1. `.env` 파일에 API 키 및 MongoDB URI 설정
2. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```
3. Flask 앱 실행
```bash
python app.py
```

## 🔑 환경 변수

```
OPENAI_API_KEY=
MONGODB_URI=
FINE_TUNE_SYMPTOM_MODEL=
FINE_TUNE_EFCY_MODEL=
FINE_TUNE_USEMETHOD_MODEL=
FINE_TUNE_ATPN_MODEL=
```

## 📌 주요 API

- `POST /medicine/symptom` : 증상 기반 약 추천
- `POST /medicine/select` : 선택한 약에 대한 설명 생성
- `POST /medicine/detail` : 복용법/주의사항 제공
- `POST /medicine/name` : 약 이름 추출 및 후보 제공

## 📄 Swagger 문서

Swagger 문서는 `/docs/swagger.yaml` 참고