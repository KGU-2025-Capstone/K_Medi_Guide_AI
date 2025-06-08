import os
import json
import re
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

# 문서별 전처리
def load_paragraphs_from_docx(filepath):
    doc = Document(filepath)
    cleaned_paragraphs = []
    for p in doc.paragraphs:
        raw = p.text.strip()
        if raw:
            cleaned = clean_paragraph(raw)
            if cleaned:
                cleaned_paragraphs.append(cleaned)
    return cleaned_paragraphs

# 패턴 기반 필터링하기
def is_irrelevant(text):
    patterns = [
        r"^안녕하세요.*",
        r"^오늘은.*",
        r"^포스팅.*",
        r"^읽어주셔서.*",
        r"^다음 시간.*",
        r".*공감.*댓글.*부탁.*",
        r".*링크.*참고.*",
        r".*광고.*포함.*",
    ]
    for pattern in patterns:
        if re.match(pattern, text):
            return True
    return False

# 불용문구
def contains_irrelevant_phrase(text):
    stop_phrases = ["안녕하세요", "읽어주셔서", "포스팅", "공감", "댓글", "팔로우", "협찬", "광고", "다음 시간에"]
    return any(phrase in text for phrase in stop_phrases)

def clean_paragraph(text):
    # 물음표는 보존
    text = re.sub(r"[^\w\s가-힣?]", "", text)
    text = re.sub(r"\s+", " ", text)
    if len(text) < 5:
        return None
    if is_irrelevant(text) or contains_irrelevant_phrase(text):
        return None
    return text

# 질문 패턴 검사
def is_question(text):
    return text.strip().endswith("?") or re.match(r"^\d+\.\s.*\?$", text)

def merge_similar_paragraphs(paragraphs, model, threshold=0.8):
    texts = [p["text"] for p in paragraphs]
    embeddings = model.encode(texts, convert_to_numpy=True)

    merged_paragraphs = []
    i = 0
    while i < len(texts):
        current_text = texts[i]
        current_embedding = embeddings[i]
        source = paragraphs[i]["source"]

        if is_question(current_text) and i + 1 < len(texts):
            # 질문 다음 문단은 무조건 병합
            next_text = texts[i + 1]
            next_embedding = embeddings[i + 1]
            merged_text = current_text + " " + next_text
            merged_embedding = (current_embedding + next_embedding) / 2
            merged_paragraphs.append({"source": source, "text": merged_text})
            i += 2
        else:
            # 일반 유사도 기준 병합
            buffer_text = current_text
            buffer_embedding = current_embedding
            j = i + 1
            while j < len(texts):
                sim = cosine_similarity(
                    buffer_embedding.reshape(1, -1),
                    embeddings[j].reshape(1, -1)
                )[0][0]
                if sim >= threshold:
                    buffer_text += " " + texts[j]
                    buffer_embedding = (buffer_embedding + embeddings[j]) / 2
                    j += 1
                else:
                    break
            merged_paragraphs.append({"source": source, "text": buffer_text})
            i = j
    return merged_paragraphs

def calculate_optimal_threshold(merged_counts, thresholds):
    diffs = np.diff(merged_counts)
    if len(diffs) > 0:
        candidates = [
            idx for idx in range(len(diffs)) if merged_counts[idx+1] > 3
        ]
        if candidates:
            max_drop_idx = candidates[np.argmin(diffs[candidates])]
            return thresholds[max_drop_idx]
    return 0.8

if __name__ == "__main__":
    folder = 'rag/docs'
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    all_results = []

    for filename in os.listdir(folder):
        if not filename.endswith('.docx') or filename.startswith("~$"):
            continue
        full_path = os.path.join(folder, filename)
        paragraphs = load_paragraphs_from_docx(full_path)
        paragraphs_dicts = [{"source": filename, "text": p} for p in paragraphs]
        if not paragraphs_dicts:
            print(f"{filename}: 내용 없음")
            continue

        print(f"{filename}: {len(paragraphs)} 문단")

        thresholds = np.arange(0.5, 0.96, 0.05)
        merged_counts = []

        for t in thresholds:
            merged = merge_similar_paragraphs(paragraphs_dicts, model, threshold=t)
            merged_counts.append(len(merged))
            print(f"  Threshold: {t:.2f} -> 병합 문단 수: {len(merged)}")

        # plt.figure(figsize=(8, 5))
        # plt.plot(thresholds, merged_counts, marker='o')
        # plt.xlabel("Cosine Similarity Threshold")
        # plt.ylabel("Number of Merged Paragraphs")
        # plt.title(f"Change in Number of Merged Paragraphs by Threshold\n({filename})")
        # plt.grid(True)
        # plt.show()

        best_threshold = calculate_optimal_threshold(merged_counts, thresholds)
        print(f"선택된 최적 임계값: {best_threshold:.2f}")

        final_merged = merge_similar_paragraphs(paragraphs_dicts, model, threshold=best_threshold)
        print(f"최종 병합 후 문단 수: {len(final_merged)}")

        output_dir = "rag/data/paragraphs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_merged, f, ensure_ascii=False, indent=2)

        print(f"병합된 문단 저장 완료: {output_path}")
        all_results.append({
            "filename": filename,
            "original_paragraphs": len(paragraphs),
            "final_paragraphs": len(final_merged),
            "best_threshold": best_threshold
        })

    print("\n전체 문서 처리 완료")
    for res in all_results:
        print(f"{res['filename']}: 원문 {res['original_paragraphs']} -> 병합 {res['final_paragraphs']} (임계값: {res['best_threshold']:.2f})")

    for f in os.listdir(output_dir):
        if f.endswith(".docx"):
            os.remove(os.path.join(output_dir, f))
