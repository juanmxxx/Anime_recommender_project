
Como arrancar el servidor, esta todo en localhost, proceso:

## Iniciar parte back end

Importante: Antes de comenzar primero hay que activar el entorno que esta en la carpeta raiz


```
.venv\Scripts\activate
```

Ingresar siguientes comandos

```
cd backend
uvicorn api:app --reload
```

## Inicializar parte front end

Primero ingresar a la carpeta de frontend por comandos y escribir lo siguiente

```
cd frontend
npm run dev
```



## Errores

Si no funciona la parte de web cuando se hace una query o un request a la API, el problema puede ser en como se este comunicando la parte de la API con la parte del backend, para poder acceder al ENDPOINT y hacer un debug de lo que va sucediento se puede probar con esta direccion web y ejecutar ENDPOINTS a modo de request

```
http://localhost:8000/recommend?keywords=love%20war%20&top_n=34
```