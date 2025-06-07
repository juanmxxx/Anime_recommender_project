import pandas as pd
import sqlalchemy

# Conexión a la base de datos
engine = sqlalchemy.create_engine('postgresql://anime_db:anime_db@localhost:5432/animes')

# Consulta SQL
query = "SELECT * FROM anime"

# Cargar datos en DataFrame de pandas
df = pd.read_sql_query(query, engine)

# Mostrar las primeras filas del DataFrame
pd.set_option('display.max_columns', None)
print(df.head())




# También puedes guardar en otros formatos
df.to_excel('anime_dataset.xlsx', index=False)
df.to_json('anime_dataset.json', orient='records')
df.to_pickle('anime_dataset.pkl')