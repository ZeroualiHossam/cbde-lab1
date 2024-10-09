import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from config import load_config
from connect import connect
import time
import statistics

def load_frases_and_embeddings(cursor):
    query = '''
        SELECT id, frase, embedding FROM frases WHERE embedding IS NOT NULL
    '''
    cursor.execute(query)
    return cursor.fetchall()

def get_new_phrases():
    return [
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

def generate_embeddings(model, frases):
    return model.encode(frases)

def compare_embeddings(new_embeddings, db_embeddings, new_phrases, metric="cosine"):
    comparison_results = []
    times = []

    for i, new_embedding in enumerate(new_embeddings):
        similarities = []

        start_time = time.time()

        for db_id, db_frase, db_embedding in db_embeddings:
            db_embedding = np.array(db_embedding)

            if db_frase.strip() != new_phrases[i].strip():
                if metric == "cosine":
                    similarity = cosine_similarity([new_embedding], [db_embedding])[0][0]
                elif metric == "euclidean":
                    similarity = euclidean(new_embedding, db_embedding)

                similarities.append((db_id, db_frase, similarity))

        similarities = sorted(similarities, key=lambda x: x[2], reverse=True if metric == "cosine" else False)
        top_2 = similarities[:2]
        comparison_results.append((i, top_2))

        end_time = time.time()
        elapsed_time = end_time - start_time
        times.append(elapsed_time)

    return comparison_results, times

def print_comparisons(new_phrases, comparison_results, metric_name):
    print(f"\nTop-2 más similares usando {metric_name}:")
    for i, top_similarities in comparison_results:
        print(f"\nNueva oración {i + 1}: \"{new_phrases[i]}\"")
        for db_id, db_frase, similarity in top_similarities:
            print(f" - Similar a frase en BD con ID {db_id}: \"{db_frase}\" con {metric_name}: {similarity:.4f}")

def calculate_time_statistics(times):
    min_time = min(times)
    max_time = max(times)
    avg_time = statistics.mean(times)
    std_dev_time = statistics.stdev(times) if len(times) > 1 else 0

    print("\nEstadísticas de tiempo:")
    print(f" - Tiempo mínimo: {min_time:.4f} segundos")
    print(f" - Tiempo máximo: {max_time:.4f} segundos")
    print(f" - Tiempo promedio: {avg_time:.4f} segundos")
    print(f" - Desviación estándar: {std_dev_time:.4f} segundos")

def main():
    config = load_config()
    connection = connect(config)

    if connection is None:
        print("Error: No se pudo conectar a la base de datos.")
        return

    try:
        with connection:
            with connection.cursor() as cursor:
                db_frases_embeddings = load_frases_and_embeddings(cursor)
                new_phrases = get_new_phrases()
                model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                new_embeddings = generate_embeddings(model, new_phrases)

                top_2_cosine, times_cosine = compare_embeddings(new_embeddings, db_frases_embeddings, new_phrases, metric="cosine")
                print_comparisons(new_phrases, top_2_cosine, "similitud coseno")
                calculate_time_statistics(times_cosine)

                top_2_euclidean, times_euclidean = compare_embeddings(new_embeddings, db_frases_embeddings, new_phrases, metric="euclidean")
                print_comparisons(new_phrases, top_2_euclidean, "distancia euclidiana (inversa)")
                calculate_time_statistics(times_euclidean)

    except (psycopg2.DatabaseError, Exception) as e:
        print(f"Error en la ejecución: {e}")
    finally:
        if connection is not None:
            connection.close()

if __name__ == '__main__':
    main()
