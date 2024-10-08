import psycopg2
from config import load_config
from connect import connect
import time
import statistics

BATCH_SIZE = 1000

def initialize_table(cursor):
    create_table_query = '''
        CREATE TABLE IF NOT EXISTS frases (
            id SERIAL PRIMARY KEY,
            frase TEXT NOT NULL,
            embedding FLOAT[] DEFAULT NULL
        )
    '''
    try:
        cursor.execute(create_table_query)

    except (psycopg2.DatabaseError, Exception) as e:
        print(f"Error al crear la tabla: {e}")
        raise

def batch_insert_frases(cursor, frases, batch_size=BATCH_SIZE):
    insert_query = '''
        INSERT INTO frases (frase) VALUES (%s)
    '''
    frases_batch = []
    insertion_times = []

    try:
        for frase in frases:
            frases_batch.append((frase,))
            if len(frases_batch) == batch_size:
                start = time.time()
                cursor.executemany(insert_query, frases_batch)
                end = time.time()
                insertion_times.append(end - start)
                frases_batch = []

        if frases_batch:
            start = time.time()
            cursor.executemany(insert_query, frases_batch)
            end = time.time()
            insertion_times.append(end - start)

        if insertion_times:
            min_time = min(insertion_times)
            max_time = max(insertion_times)
            avg_time = statistics.mean(insertion_times)
            std_dev_time = statistics.stdev(insertion_times) if len(insertion_times) > 1 else 0

            print("Estadísticas de inserción de frases:")
            print(f" - Tiempo mínimo: {min_time:.4f} segundos")
            print(f" - Tiempo máximo: {max_time:.4f} segundos")
            print(f" - Tiempo promedio: {avg_time:.4f} segundos")
            print(f" - Desviación estándar: {std_dev_time:.4f} segundos")

    except (psycopg2.DatabaseError, Exception) as e:
        print(f"Error al insertar frases: {e}")
        raise

def read_frases_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            return content.splitlines()

    except FileNotFoundError as e:
        print(f"Archivo no encontrado: {e}")
        raise

def main():
    config = load_config()
    connection = connect(config)

    if not connection:
        print("Error: No se pudo conectar a la base de datos.")
        return

    try:
        with connection:
            with connection.cursor() as cursor:
                initialize_table(cursor)
                frases = read_frases_from_file('../BookCorpus/frases_extraidas.txt')
                batch_insert_frases(cursor, frases, batch_size=BATCH_SIZE)
                connection.commit()

    except (psycopg2.DatabaseError, Exception) as e:
        connection.rollback()
        print(f"Error: {e}")
    finally:
        if connection:
            connection.close()

if __name__ == '__main__':
    main()
