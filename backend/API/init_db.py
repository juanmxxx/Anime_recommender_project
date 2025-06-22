# Initialize the database and create tables
import sys
import os
import traceback

# Add proper paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import configuration and models
try:
    from config.database import engine, Base, DB_CONFIG
    from models.anime import MetricEntry
    print("✓ Importaciones correctas desde el módulo API")
except ImportError:
    # Intentar con rutas absolutas si las importaciones relativas fallan
    print("⚠️ Intentando importaciones absolutas...")
    from backend.API.config.database import engine, Base, DB_CONFIG
    from backend.API.models.anime import MetricEntry
    print("✓ Importaciones correctas usando rutas absolutas")

# Asegúrate de que MetricEntry esté en los metadatos de Base
print(f"ℹ️ Tablas registradas en los metadatos: {list(Base.metadata.tables.keys())}")

def check_table_exists(table_name):
    """Check if a table exists in the database"""
    try:
        # Intenta usar PostgreSQL
        import psycopg2
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{table_name}')")
        exists = cursor.fetchone()[0]
        conn.close()
        return exists
    except Exception as e:
        # Si falla PostgreSQL, podemos estar usando SQLite u otra base de datos
        print(f"No se pudo verificar tabla con PostgreSQL: {e}")
        # Con SQLAlchemy podemos hacer esto de manera independiente del motor de base de datos
        from sqlalchemy import inspect
        inspector = inspect(engine)
        return table_name in inspector.get_table_names()

def init_db():
    try:
        # Verificar si las tablas ya existen
        print("Verificando existencia de tablas...")

        # Verificar si MetricEntry está registrado correctamente
        if "user_metrics" not in Base.metadata.tables:
            print("❌ La tabla 'user_metrics' no está registrada en los metadatos. Verificando modelo...")
            # Forzar el registro del modelo
            from models.anime import MetricEntry
            print(f"✓ Modelo MetricEntry referenciado. Tablas disponibles: {list(Base.metadata.tables.keys())}")
        
        # Verificar si la tabla existe en la base de datos
        user_metrics_exists = check_table_exists("user_metrics")
        
        if user_metrics_exists:
            print("✓ Tabla 'user_metrics' ya existe, omitiendo creación.")
        else:
            # Crear tablas
            print("Creando tablas en la base de datos...")
            # Imprimir tablas que se van a crear
            tables_to_create = Base.metadata.tables
            print(f"ℹ️ Creando las siguientes tablas: {list(tables_to_create.keys())}")
            
            # Crear todas las tablas
            Base.metadata.create_all(bind=engine)
            
            # Verificar que las tablas se crearon
            from sqlalchemy import inspect
            inspector = inspect(engine)
            created_tables = inspector.get_table_names()
            print(f"✓ Tablas creadas: {created_tables}")
            
            if "user_metrics" in created_tables:
                print("✓ Tabla 'user_metrics' creada correctamente.")
            else:
                print("❌ La tabla 'user_metrics' NO se creó correctamente.")
                
    except Exception as e:
        print(f"❌ Error al inicializar la base de datos: {e}")
        traceback.print_exc()
        sys.exit(1)

def recreate_db():
    """Recrear la base de datos eliminando y volviendo a crear tablas"""
    try:
        print("🔄 Eliminando y recreando todas las tablas...")
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        print("✓ Base de datos recreada correctamente.")
    except Exception as e:
        print(f"❌ Error al recrear la base de datos: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Inicializar o recrear la base de datos')
    parser.add_argument('--recreate', action='store_true', help='Eliminar y recrear todas las tablas')
    
    args = parser.parse_args()
    
    if args.recreate:
        recreate_db()
    else:
        init_db()
