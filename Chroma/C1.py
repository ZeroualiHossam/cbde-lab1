import chromadb
import re
import os
import time

client = chromadb.Client()

collection_name = "frases_collection"
collection = client.create_collection(name=collection_name)

def load_sentences_in_batches(file_path, batch_size=5000):

    if not os.path.exists(file_path):
        print(f"El archivo {file_path} no existe.")
        return

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        sentences = content.splitlines()

        sentence_data = [{"id": str(i), "text": sentence.strip()} for i, sentence in enumerate(sentences) if sentence.strip()]

        times = []

        for i in range(0, len(sentence_data), batch_size):
            batch = sentence_data[i:i + batch_size]
            start_time = time.time()

            collection.add(
                ids=[item["id"] for item in batch],
                documents=[item["text"] for item in batch]
            )

            end_time = time.time()
            elapsed_time = end_time - start_time
            times.append(elapsed_time)

            print(f"Frases {i + 1} a {i + len(batch)} cargadas exitosamente en Chroma en {elapsed_time:.4f} segundos.")

        if times:
            max_time = max(times)
            min_time = min(times)
            total_time = sum(times)
            avg_time = sum(times) / len(times)
    
            print(f"Tiempo máximo: {max_time:.4f} segundos")
            print(f"Tiempo mínimo: {min_time:.4f} segundos")
            print(f"Tiempo total: {total_time:.4f} segundos")
            print(f"Tiempo promedio: {avg_time:.4f} segundos")

if __name__ == '__main__':
    file_path = '../BookCorpus/frases_extraidas.txt'
    load_sentences_in_batches(file_path, batch_size=5003)

