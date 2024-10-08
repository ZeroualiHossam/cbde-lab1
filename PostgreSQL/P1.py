import psycopg2
from config import load_config
from connect import connect
from sentence_transformers import SentenceTransformer
import time
import statistics

BATCH_SIZE = 1000

def fetch_frases(cursor, batch_size):
    query = 'SELECT id, frase FROM frases WHERE embedding IS NULL ORDER BY id'
    cursor.execute(query)
    while True:
        batch = cursor.fetchmany(batch_size)
        if not batch:
            break
        yield batch

def extract_frases(frases_batch):
    return [frase for _, frase in frases_batch]

def generar_embeddings(model, frases):
    return model.encode(frases)

def actualizar_embeddings(cursor, frases_batch, embeddings):
    query = 'UPDATE frases SET embedding = %s WHERE id = %s'
    try:
        data = [(embedding.tolist(), id) for (id, _), embedding in zip(frases_batch, embeddings)]
        cursor.executemany(query, data)
    except (psycopg2.DatabaseError, Exception) as e:
        print(f"Error actualizando embeddings: {e}")
        raise

def main():
    config = load_config()
    connection = connect(config)

    if connection is None:
        print('Error: No se pudo establecer la conexión a la base de datos.')
        return

    tiempos_insercion = []

    try:
        with connection:
            with connection.cursor() as cur_fetch, connection.cursor() as cur_update:
                model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

                for batch in fetch_frases(cur_fetch, BATCH_SIZE):
                    frases = extract_frases(batch)
                    embeddings = generar_embeddings(model, frases)

                    inicio = time.time()
                    actualizar_embeddings(cur_update, batch, embeddings)
                    fin = time.time()

                    tiempos_insercion.append(fin - inicio)

                connection.commit()
                print('Embeddings actualizados correctamente.')

        if tiempos_insercion:
            min_tiempo = min(tiempos_insercion)
            max_tiempo = max(tiempos_insercion)
            promedio_tiempo = sum(tiempos_insercion) / len(tiempos_insercion)
            desviacion_std = statistics.stdev(tiempos_insercion) if len(tiempos_insercion) > 1 else 0

            print("Tiempos de inserción de embeddings:")
            print(f" - Mínimo: {min_tiempo:.4f} s")
            print(f" - Máximo: {max_tiempo:.4f} s")
            print(f" - Promedio: {promedio_tiempo:.4f} s")
            print(f" - Desviación estándar: {desviacion_std:.4f} s")

    except (psycopg2.DatabaseError, Exception) as e:
        connection.rollback()
        print(f"Error durante la ejecución: {e}")
    finally:
        if connection:
            connection.close()
            print('Conexión a la base de datos cerrada.')

if __name__ == '__main__':
    main()
