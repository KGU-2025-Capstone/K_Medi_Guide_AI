from sentence_transformers import SentenceTransformer, util
import json
import os
from glob import glob
import re
import logging
from collections import Counter

# 로그 설정
logging.basicConfig(level=logging.INFO)

# 모델 로딩 (정확도 높은 검색 특화 모델)
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

cached_corpus = None

def preprocess_context(context):
    """
    너무 긴 문맥은 문단 단위로 쪼개고, 너무 짧은 문장 제거
    """
    paragraphs = [p.strip() for p in context.split('\n') if len(p.strip().split()) >= 10]
    return paragraphs

def load_all_corpus(corpus_dir='rag/data/corpus'):
    global cached_corpus
    if cached_corpus is not None:
        logging.info("코퍼스를 메모리에서 로드합니다.")
        return cached_corpus

    corpus = []
    file_paths = glob(os.path.join(corpus_dir, '*.json'))
    if not file_paths:
        raise FileNotFoundError(f"{corpus_dir} 폴더에 corpus 파일이 없습니다. 파이프라인을 먼저 실행해주세요.")

    logging.info(f"{len(file_paths)}개의 파일을 로드 중입니다.")
    for file_path in file_paths:
        with open(file_path, encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                filename = item.get("filename", os.path.basename(file_path))
                for para in preprocess_context(item['context']):
                    embedding = model.encode(f"passage: {para}", convert_to_tensor=True)
                    corpus.append({
                        "context": para,
                        "embedding": embedding,
                        "filename": filename
                    })

    cached_corpus = corpus
    logging.info(f"총 {len(corpus)}개의 문단(context)이 로드되었습니다.")
    return corpus

def extract_keywords(query, num_keywords=5):
    words = re.findall(r'\w+', query.lower())
    word_counts = Counter(words)
    most_common = word_counts.most_common(num_keywords)
    return [word for word, count in most_common]

def get_similar_contexts(query, top_k=3):
    corpus = load_all_corpus()
    query_keywords = extract_keywords(query)

    query_embedding = model.encode(f"query: {query}", convert_to_tensor=True)

    scored_contexts = []
    for c in corpus:
        context = c["context"]

        score = util.cos_sim(query_embedding, c["embedding"]).item()

        # 키워드가 몇 개 포함됐는지에 따른 점수 보정 (비율 기반)
        keyword_matches = sum(1 for kw in query_keywords if kw in context.lower())
        match_ratio = keyword_matches / len(query_keywords) if query_keywords else 0
        score += 0.05 * match_ratio  # 가중치 강화

        scored_contexts.append((score, context, c["filename"]))

    top_contexts = sorted(scored_contexts, key=lambda x: x[0], reverse=True)[:top_k]

    for idx, (score, context, filename) in enumerate(top_contexts):
        preview = context.strip().replace("\n", " ")[:100]
        logging.info(f"{idx + 1}. 파일: {filename} | 점수: {score:.4f} | 내용: {preview}...")

    return top_contexts
