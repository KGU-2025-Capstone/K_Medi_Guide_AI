import json
import os


def build_corpus_for_all_documents(
    clusters_folder="rag/data/clusters",
    corpus_folder="rag/data/corpus"
):
    """
    클러스터링된 문서 요약 데이터를 기반으로 통합 코퍼스를 생성합니다.
    """
    os.makedirs(corpus_folder, exist_ok=True)

    all_corpus = []
    corpus_id = 1

    for filename in os.listdir(clusters_folder):
        if not filename.endswith(".json"):
            continue

        cluster_path = os.path.join(clusters_folder, filename)
        with open(cluster_path, encoding="utf-8") as f:
            clustered_data = json.load(f)

        for item in clustered_data:
            entry = {
                "id": corpus_id,
                "filename": filename,
                "cluster": item.get("cluster", -1),
                "summary": ', '.join(item.get("keywords", [])),
                "context": item.get("text", ""),
                "embedding": item.get("embedding", [])
            }
            all_corpus.append(entry)
            corpus_id += 1

    # 전체 코퍼스 통합 저장
    output_path = os.path.join(corpus_folder, "corpus.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_corpus, f, ensure_ascii=False, indent=2)

    print(f"[통합 코퍼스 생성 완료] 저장 위치: {output_path}")


if __name__ == "__main__":
    build_corpus_for_all_documents()
