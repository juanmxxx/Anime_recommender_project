"""
Script de diagnóstico para identificar problemas en el motor de búsqueda de anime
"""
import sys
import os
import subprocess
import json

def test_search_engine(query, top_n=5):
    """
    Prueba el motor de búsqueda con una consulta específica y muestra información de diagnóstico
    """
    print(f"\n{'='*80}")
    print(f"DIAGNÓSTICO DE CONSULTA: {query}")
    print(f"{'='*80}")
      # Configuración de comandos y rutas
    python_executable = r"c:\proyectoIA\.venv\Scripts\python.exe"
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "anime_search_engine_fixed.py")
    
    print(f"1. Ejecutando consulta con comando:")
    cmd = [python_executable, script_path, query, "-n", str(top_n), "-f", "json"]
    print(f"   {' '.join(cmd)}")
    
    # Ejecutar la consulta
    result = subprocess.run(
        cmd,
        capture_output=True, 
        text=True, 
        encoding="utf-8"
    )
    
    # Mostrar información del proceso
    print(f"\n2. Información del proceso:")
    print(f"   - Código de salida: {result.returncode}")
    print(f"   - ¿Tiene salida stdout?: {'Sí (' + str(len(result.stdout)) + ' bytes)' if result.stdout else 'No'}")
    print(f"   - ¿Tiene salida stderr?: {'Sí (' + str(len(result.stderr)) + ' bytes)' if result.stderr else 'No'}")
    
    # Analizar la salida estándar
    if result.stdout:
        print("\n3. Análisis de la salida estándar:")
        output = result.stdout.strip()
        
        if not output:
            print("   La salida está vacía después de quitar espacios en blanco.")
        else:
            # Buscar el inicio del JSON
            json_start = output.find('{')
            json_output = output[json_start:] if json_start >= 0 else output
            
            print(f"   - Primeros 200 caracteres: {output[:200]}...")
            
            try:
                # Intentar cargar el JSON
                parsed_data = json.loads(json_output)
                print("   - ✅ JSON válido detectado")
                print(f"   - Estructura: {list(parsed_data.keys()) if isinstance(parsed_data, dict) else 'No es un diccionario'}")
                
                if isinstance(parsed_data, dict) and "success" in parsed_data:
                    print(f"   - Éxito: {parsed_data['success']}")
                    
                    if not parsed_data.get("success"):
                        print(f"   - Error reportado: {parsed_data.get('error', 'No se especificó error')}")
                        
                    if "results" in parsed_data:
                        print(f"   - Número de resultados: {len(parsed_data['results'])}")
            except json.JSONDecodeError as e:
                print(f"   - ❌ Error al parsear JSON: {str(e)}")
                print(f"   - Posición del error: {e.pos}")
                print(f"   - Cerca de: {json_output[max(0, e.pos-20):min(len(json_output), e.pos+20)]}")
    
    # Analizar la salida de error
    if result.stderr:
        print("\n4. Análisis de stderr:")
        print("   " + result.stderr.replace('\n', '\n   '))
        
    return result

def main():
    # Consulta problemática
    problematic_query = "a female protagonist with a red band"
    
    if len(sys.argv) > 1:
        problematic_query = sys.argv[1]
    
    # Ejecutar diagnóstico
    test_search_engine(problematic_query)
    
    print("\n🔍 Diagnóstico completo")

if __name__ == "__main__":
    main()
