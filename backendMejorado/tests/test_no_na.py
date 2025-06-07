import psycopg2

# Configuración de la conexión
conn = psycopg2.connect(
    dbname='animes',
    user='anime_db',
    password='anime_db',
    host='localhost',
    port=5432
)

cur = conn.cursor()

# Obtener los nombres de las columnas
cur.execute('SELECT column_name FROM information_schema.columns WHERE table_name = %s', ('anime',))
columns = [row[0] for row in cur.fetchall()]

# Obtener el total de registros
cur.execute('SELECT COUNT(*) FROM anime')
total = cur.fetchone()[0]

# Imprimir los primeros 100 registros
print('Primeros 100 registros:')
cur.execute('SELECT * FROM anime ORDER BY anime_id ASC LIMIT 100')
rows = cur.fetchall()
for row in rows:
    print(dict(zip(columns, row)))

# Imprimir los últimos 100 registros
print('\nÚltimos 100 registros:')
if total > 100:
    cur.execute('SELECT * FROM anime ORDER BY anime_id DESC LIMIT 100')
    rows = cur.fetchall()[::-1]  # Invertir para mostrar en orden ascendente
else:
    cur.execute('SELECT * FROM anime ORDER BY anime_id ASC')
    rows = cur.fetchall()
for row in rows:
    print(dict(zip(columns, row)))

cur.close()
conn.close()
