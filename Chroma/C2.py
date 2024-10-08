import chromadb
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from time import perf_counter

client = chromadb.Client()

collection_name = "frases_collection"
collection = client.create_collection(name=collection_name)

def process_sentences(sentences, batch_size=5000):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    cosine_times = []
    euclidean_times = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]

        embeddings = model.encode(batch)

        start_time_cosine = perf_counter()
        cosine_sim_matrix = cosine_similarity(embeddings)
        end_time_cosine = perf_counter()
        cosine_elapsed_time = end_time_cosine - start_time_cosine

        max_cosine_similarity = 0
        most_similar_pair_cosine = (None, None)
        for j in range(len(embeddings)):
            for k in range(j + 1, len(embeddings)):
                cosine_sim = cosine_sim_matrix[j][k]
                if cosine_sim > max_cosine_similarity:
                    max_cosine_similarity = cosine_sim
                    most_similar_pair_cosine = (j, k)

        start_time_euclidean = perf_counter()
        euclidean_dist_matrix = euclidean_distances(embeddings)
        end_time_euclidean = perf_counter()
        euclidean_elapsed_time = end_time_euclidean - start_time_euclidean

        min_euclidean_distance = float('inf')
        most_similar_pair_euclidean = (None, None)
        for j in range(len(embeddings)):
            for k in range(j + 1, len(embeddings)):
                euclidean_dist = euclidean_dist_matrix[j][k]
                if euclidean_dist < min_euclidean_distance:
                    min_euclidean_distance = euclidean_dist
                    most_similar_pair_euclidean = (j, k)

        cosine_times.append(cosine_elapsed_time)
        euclidean_times.append(euclidean_elapsed_time)

        if most_similar_pair_cosine != (None, None):
            j, k = most_similar_pair_cosine
            print(f"\nCoseno - Frase {j + 1}: \"{batch[j]}\"")
            print(f" - Similar a frase {k + 1}: \"{batch[k]}\" con similitud coseno: {max_cosine_similarity:.4f}")

        if most_similar_pair_euclidean != (None, None):
            j, k = most_similar_pair_euclidean
            print(f"\nEuclidiana - Frase {j + 1}: \"{batch[j]}\"")
            print(f" - Similar a frase {k + 1}: \"{batch[k]}\" con distancia euclidiana: {min_euclidean_distance:.4f}")

        collection.add(
            ids=[str(i + j) for j in range(len(batch))],
            documents=batch
        )

        print(f"Tiempo de cálculo de similitud coseno para frases {i + 1} a {i + len(batch)}: {cosine_elapsed_time:.4f} segundos.")
        print(f"Tiempo de cálculo de distancia euclidiana para frases {i + 1} a {i + len(batch)}: {euclidean_elapsed_time:.4f} segundos.")

    if cosine_times:
        print(f"Tiempo promedio de cálculo de similitud coseno: {sum(cosine_times) / len(cosine_times):.4f} segundos")

    if euclidean_times:
        print(f"Tiempo promedio de cálculo de distancia euclidiana: {sum(euclidean_times) / len(euclidean_times):.4f} segundos")

if __name__ == '__main__':
    sentences = [
        "as the first grandchild , megan spent a lot of time with her grandparents , and that in turn , meant she spent a lot of time with aidan .",
        "he had devoted hours to holding her and spoiling her rotten .",
        "when it came time for her to talk , she just could n't seem to get `` uncle aidan '' out .",
        "instead , she called him `` ankle . ''",
        "it was a nickname that had stuck with him even now that he was thirty-four and married .",
        "while it had been no question that she wanted him as godfather for mason , she had been extremely honored when he and his wife , emma , had asked her to be their son , noah 's , godmother .",
        "she loved her newest cousin very much and planned to be the best godmother she could for him .",
        "as she stepped out of the bedroom , she found that mason had yet to move .",
        "`` okay buddy , time to go . ''",
        "when he started to whine , she shook her head ."
    ]

    process_sentences(sentences, batch_size=5003)
