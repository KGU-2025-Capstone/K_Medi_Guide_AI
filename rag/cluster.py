from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import json
import os
import numpy as np

def cluster_all_documents_summary(summaries_folder, output_path, method='kmeans'):
    print("\n[전체 문서 클러스터링 시작]")

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    all_summaries = []
    all_texts = []

    for filename in os.listdir(summaries_folder):
        if filename.endswith(".json"):
            with open(os.path.join(summaries_folder, filename), encoding="utf-8") as f:
                summaries = json.load(f)
                all_summaries.extend(summaries)
                all_texts.extend([s['text'] for s in summaries])

    if len(all_texts) < 2:
        print("  문단이 2개 미만이므로 클러스터링 생략")
        return

    embeddings = model.encode(all_texts)

    best_k = None
    best_score = -1
    cluster_range = range(2, min(11, len(all_texts)))
    sse = []

    if method == 'kmeans':
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            sse.append(kmeans.inertia_)
            print(f"  k={k}, Silhouette Score={score:.4f}")
            if score > best_score:
                best_score = score
                best_k = k

        kmeans = KMeans(n_clusters=best_k, random_state=42)
        labels = kmeans.fit_predict(embeddings)

    elif method == 'agglomerative':
        for k in cluster_range:
            agg = AgglomerativeClustering(n_clusters=k)
            labels = agg.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            print(f"  k={k}, Silhouette Score={score:.4f}")
            if score > best_score:
                best_score = score
                best_k = k

        agg = AgglomerativeClustering(n_clusters=best_k)
        labels = agg.fit_predict(embeddings)

    else:
        raise ValueError("지원하지 않는 클러스터링 방식입니다. (kmeans/agglomerative)")

    print(f"  최적 k = {best_k}, Silhouette Score = {best_score:.4f}")

    for i, s in enumerate(all_summaries):
        s['cluster'] = int(labels[i])
        s['embedding'] = embeddings[i].tolist()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    print(f"  저장 완료: {output_path}")

    plt.figure(figsize=(6, 4))
    plt.plot(list(cluster_range), sse, marker='o')
    plt.title(f"Elbow Method - 전체 문서")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Sum of Squared Errors (SSE)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    summaries_folder = "rag/data/summaries"
    output_path = "rag/data/clusters/all_documents_clustered.json"
    cluster_all_documents_summary(summaries_folder, output_path, method='kmeans')  # 또는 'agglomerative'
