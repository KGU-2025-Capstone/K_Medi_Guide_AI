from keybert import KeyBERT
import json
import os
import glob
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# 한국어 불용어 리스트 (필요시 확장 가능)
KOREAN_STOPWORDS = set([
    "이", "그", "저", "것", "수", "등", "들", "및", "의", "은", "는", "을", "를",
    "에", "에서", "과", "와", "하다", "되다", "되며", "했다", "합니다", "있습니다",
    "같습니다", "됩니다", "되므로", "하면", "있고", "있다", "라고", "하고",
    "대해", "대한", "하지만", "그러나", "즉", "또는", "때문에", "더", "좀", "더욱", "또한"
])

kw_model = KeyBERT(model='distilbert-base-nli-mean-tokens')

input_folder = "rag/data/paragraphs"
output_folder = "rag/data/summaries"
os.makedirs(output_folder, exist_ok=True)

def clean_text(text):
    # 한글, 영어, 숫자, 공백만 남기고 특수문자 제거
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def is_stopword(word):
    # 영어/한국어 불용어 모두 검사 + 길이 2 이하 단어 제외 (너무 짧은 단어)
    return (word in ENGLISH_STOP_WORDS) or (word in KOREAN_STOPWORDS) or (len(word) <= 2)

def clean_keywords(keywords):
    cleaned = set()
    for kw in keywords:
        kw_clean = clean_text(kw)
        words = kw_clean.split()
        if any(not is_stopword(w) for w in words):
            cleaned.add(kw_clean)
    return list(cleaned)

def extract_keywords(text, top_n=5):
    try:
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 4),  # n-gram 범위 확대
            top_n=top_n * 3,               # 후보 키워드 좀 더 많이 추출 후 필터링
            use_maxsum=False,              # maxsum 끔, 더 많은 키워드 후보 확보
            nr_candidates=30
        )
        # 후보 키워드 클린 및 중복 제거
        cleaned = clean_keywords([kw[0] for kw in keywords])
        # top_n 개로 자르기 (필터링 후에도 충분히 뽑힘)
        return cleaned[:top_n]
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []

def split_question_answer(text):
    if "?" in text:
        parts = text.split("?")
        question_part = parts[0] + "?"
        answer_part = "?".join(parts[1:]).strip()
    else:
        question_part = text
        answer_part = ""
    return question_part, answer_part

def extract_keywords_from_qa(text, top_n=5):
    question_part, answer_part = split_question_answer(text)
    question_keywords = extract_keywords(question_part, top_n=top_n)
    answer_keywords = extract_keywords(answer_part, top_n=top_n) if answer_part else []
    all_keywords = list(dict.fromkeys(question_keywords + answer_keywords))  # 순서 유지 중복 제거
    return all_keywords[:top_n]

for json_path in glob.glob(os.path.join(input_folder, "*.json")):
    with open(json_path, "r", encoding="utf-8") as f:
        paragraphs = json.load(f)

    summaries = []
    for p in paragraphs:
        text = p.get("text", "")
        if not text.strip():
            continue  # 빈 텍스트 스킵
        keywords = extract_keywords_from_qa(text, top_n=5)
        summaries.append({
            'text': text,
            'keywords': keywords
        })

    filename = os.path.basename(json_path)
    output_path = os.path.join(output_folder, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print(f"Saved keyword summary: {output_path}")
