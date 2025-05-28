# Proyecto de Recomendador de Anime con IA

## Descripción General del Proyecto

Este sistema de recomendación de anime basado en inteligencia artificial representa una solución avanzada para descubrir contenido anime personalizado a partir de descripciones textuales proporcionadas por los usuarios. El proyecto implementa técnicas de vanguardia en procesamiento de lenguaje natural (NLP) y aprendizaje profundo para comprender las preferencias implícitas en las consultas de los usuarios y encontrar coincidencias relevantes en una extensa base de datos de anime.

La motivación detrás de este proyecto surge de la creciente dificultad para descubrir nuevos títulos de anime en un catálogo cada vez más extenso y diverso. Los sistemas de recomendación tradicionales basados en filtros colaborativos requieren un historial de visualizaciones, mientras que nuestro enfoque permite obtener recomendaciones inmediatas basadas en descripciones textuales de lo que el usuario busca, sin necesidad de un historial previo o cuentas de usuario.

El sistema aprovecha modelos transformers pre-entrenados para analizar semánticamente las consultas del usuario, y las compara con las características de los animes almacenados utilizando técnicas avanzadas de similitud vectorial. La arquitectura del proyecto está diseñada para ser escalable, con una clara separación entre el backend de procesamiento de IA y el frontend de interfaz de usuario, permitiendo mejoras incrementales en ambos componentes de manera independiente.

## Tecnologías Utilizadas

### Backend
- **Python:** Lenguaje de programación principal para el backend, seleccionado por su extensa biblioteca de herramientas para ciencia de datos y procesamiento de lenguaje natural. Python 3.8+ proporciona un equilibrio óptimo entre rendimiento y facilidad de desarrollo.

- **FastAPI:** Framework moderno y de alto rendimiento para la construcción de APIs con Python. Seleccionado por su velocidad (comparable a Node.js y Go), tipado estático, documentación automática (Swagger UI) y soporte nativo para operaciones asíncronas, lo que permite manejar múltiples solicitudes de recomendación simultáneamente.

- **PyTorch:** Framework de deep learning desarrollado por Facebook AI Research que proporciona flexibilidad y facilidad de uso. Se utiliza para la creación, entrenamiento e inferencia de nuestros modelos de procesamiento de lenguaje natural. La naturaleza dinámica del grafo computacional de PyTorch facilita la experimentación con diferentes arquitecturas de modelos.

- **Transformers (Hugging Face):** Biblioteca que proporciona implementaciones state-of-the-art de arquitecturas transformer como BERT, GPT, T5, etc. Específicamente, utilizamos modelos pre-entrenados DistilBERT, que ofrecen un equilibrio entre precisión y eficiencia computacional (40% más ligero que BERT manteniendo el 97% de su capacidad). Esta biblioteca nos permite aprovechar el transfer learning para comprender consultas textuales complejas.

- **Pandas:** Biblioteca fundamental para la manipulación y análisis de datos tabulares. Utilizada para procesar los conjuntos de datos de anime, realizar limpieza, filtrado, y preparación de features para nuestros modelos de machine learning.

- **Sentence-Transformers:** Extensión especializada de la biblioteca Transformers que optimiza la creación de embeddings (representaciones vectoriales) para oraciones completas. Esta biblioteca nos permite transformar tanto las consultas de los usuarios como las descripciones de anime en vectores de alta dimensionalidad que preservan el significado semántico, facilitando la búsqueda por similitud.

- **KeyBERT:** Biblioteca para la extracción automatizada de palabras clave de textos utilizando embeddings BERT. Empleada para identificar los términos más relevantes tanto en las consultas de usuario como en las descripciones de anime, mejorando así la precisión de las recomendaciones.

- **NumPy:** Biblioteca para computación científica que proporciona soporte para arrays y matrices multidimensionales, junto con funciones matemáticas de alto nivel. Fundamental para las operaciones de álgebra lineal utilizadas en el cálculo de similitudes coseno entre embeddings.

### Frontend
- **React:** Biblioteca de JavaScript para la construcción de interfaces de usuario basada en componentes. Seleccionada por su eficiencia en la renderización mediante el Virtual DOM, su arquitectura de componentes reutilizables, y su amplio ecosistema. React nos permite crear una experiencia de usuario fluida y reactiva.

- **Vite:** Herramienta de construcción y servidor de desarrollo de nueva generación que ofrece tiempos de inicio instantáneos mediante Hot Module Replacement (HMR) y un empaquetado optimizado para producción. Significativamente más rápido que alternativas como Create React App.

- **JavaScript/JSX:** Lenguaje de programación principal para el frontend. JSX como extensión sintáctica permite escribir HTML en React, facilitando la creación de componentes que encapsulan tanto la lógica como la presentación.

- **CSS moderno:** Utilización de características avanzadas de CSS para crear una interfaz visualmente atractiva, con efectos de transición, animaciones y diseño responsive.

### Gestión de Datos
- **Datos CSV:** Almacenamiento y gestión de conjuntos de datos de anime estructurados en formato CSV, incluyendo metadatos como títulos, sinopsis, puntuaciones, géneros, años de emisión y otros atributos relevantes para la caracterización del contenido. El dataset principal contiene más de 17,000 entradas de anime con sus correspondientes metadatos.

- **Pickle:** Formato de serialización de Python utilizado para almacenar modelos entrenados, embeddings pre-calculados y estructuras de datos procesadas, lo que permite una carga rápida durante la inicialización del sistema y elimina la necesidad de reprocesar datos.

## Estructura del Proyecto

```
proyectoIA/
├── backend/                  # Código Python del backend
│   ├── api.py                # Endpoints de FastAPI
│   ├── modelFormer.py        # Implementación principal del modelo
│   ├── datasetProcessor.py   # Preprocesamiento del conjunto de datos
│   ├── tokenizer.py          # Tokenización de texto
│   ├── requirements.txt      # Dependencias de Python
│   ├── dataset/              # Conjuntos de datos de anime
│   │   ├── anime-dataset-2023-cleaned.csv
│   │   ├── anime-dataset-2023.csv
│   │   └── small2.csv
│   └── model/               # Archivos guardados del modelo y embeddings
│       ├── anime_data.pkl
│       ├── anime_embeddings.npy
│       ├── anime_recommender.pt
│       └── scaler.pkl
├── frontend/                # Frontend en React
│   ├── src/                 # Código fuente
│   │   ├── App.jsx          # Componente principal de la aplicación
│   │   ├── App.css          # Estilos
│   │   └── main.jsx         # Punto de entrada
│   ├── public/              # Activos estáticos
│   └── images/              # Recursos de imágenes
└── README.md                # Esta documentación
```

## Fases de Desarrollo

El desarrollo de este sistema de recomendación de anime se realizó siguiendo un proceso iterativo e incremental, dividido en varias fases bien definidas que permitieron abordar la complejidad del proyecto de manera estructurada.

### Fase 1: Recolección y Preparación de Datos

Esta fase inicial fue fundamental para establecer los cimientos del sistema de recomendación, centrándose en la adquisición y preparación de datos de alta calidad.

#### Adquisición de Datos
- **Identificación de Fuentes**: Investigación exhaustiva para identificar fuentes confiables y completas de datos de anime, evaluando la calidad y cobertura de los datasets disponibles públicamente.
- **Obtención del Dataset Principal**: Adquisición del dataset "Anime Dataset 2023" con más de 17,000 títulos de anime, incluyendo información detallada como sinopsis, géneros, puntuaciones, años de emisión y otros metadatos relevantes.
- **Datos Complementarios**: Recopilación de información adicional para enriquecer el dataset principal, como imágenes de portada y URLs de referencia.

#### Limpieza y Preprocesamiento
- **Detección y Manejo de Valores Faltantes**: Implementación de estrategias para tratar entradas con información incompleta, como imputación para campos numéricos y valores predeterminados para campos categóricos.
- **Normalización de Textos**: Aplicación de técnicas de procesamiento de lenguaje natural para normalizar sinopsis y descripciones, incluyendo tokenización, eliminación de stopwords, lematización y normalización de caracteres especiales.
- **Estandarización de Géneros**: Unificación del formato de etiquetas de género y categorías, eliminando redundancias y asegurando consistencia.
- **Filtrado de Calidad**: Eliminación de entradas con información insuficiente o de baja calidad que podrían afectar negativamente al rendimiento del sistema.

#### Ingeniería de Características
- **Extracción de Características Textuales**: Análisis de sinopsis para extraer entidades, temas recurrentes y tonos narrativos que pudieran servir como indicadores para las recomendaciones.
- **Vectorización de Textos**: Conversión de datos textuales en representaciones numéricas mediante técnicas como TF-IDF y Bag-of-Words como análisis preliminar.
- **Desarrollo de Pipeline de Datos**: Creación de un flujo de trabajo automatizado para la limpieza y preprocesamiento de datos, implementado en `datasetProcessor.py`, garantizando la reproducibilidad del proceso y facilitando actualizaciones futuras del dataset.
- **Estratificación de Dataset**: División de datos en conjuntos de entrenamiento, validación y prueba, asegurando una representación equilibrada de los diferentes géneros y tipos de anime.

#### Resultados de la Fase
- Dataset limpio y estructurado (`anime-dataset-2023-cleaned.csv`) con más de 15,000 títulos de anime con información completa.
- Pipeline de preprocesamiento reutilizable para futuras actualizaciones o ampliaciones del dataset.
- Análisis exploratorio detallado de los datos, identificando patrones, distribuciones y correlaciones relevantes para el diseño del sistema de recomendación.

### Fase 2: Desarrollo del Modelo

Esta fase se centró en el diseño, implementación y optimización de los modelos de aprendizaje automático que conforman el núcleo del sistema de recomendación.

#### Investigación y Selección de Arquitectura
- **Evaluación de Modelos**: Análisis comparativo de diferentes arquitecturas para procesamiento de lenguaje natural, incluyendo modelos tradicionales (TF-IDF + cosine similarity), Word2Vec, y arquitecturas transformer.
- **Selección de Base**: Decisión fundamentada de utilizar modelos transformer pre-entrenados (específicamente DistilBERT) por su capacidad superior para capturar relaciones semánticas complejas en texto.
- **Diseño Arquitectónico**: Definición de la arquitectura general del sistema, optando por un enfoque de recuperación basado en similitud semántica entre consultas y descripciones de anime.

#### Implementación del Modelo Base
- **Integración con Hugging Face**: Configuración y adaptación de modelos pre-entrenados de la biblioteca Transformers, específicamente DistilBERT, para el dominio específico de contenido anime.
- **Tokenización Especializada**: Implementación de un tokenizador adaptado (`tokenizer.py`) para procesar eficientemente tanto las consultas de usuario como las descripciones de anime, manejando adecuadamente términos específicos del dominio.
- **Generación de Embeddings**: Desarrollo de funciones para transformar descripciones textuales en vectores de alta dimensionalidad (768 dimensiones) que preservan las relaciones semánticas entre conceptos.

#### Optimización y Refinamiento
- **Fine-tuning del Modelo**: Ajuste de hiperparámetros para adaptar el modelo pre-entrenado al dominio específico del anime, mejorando su capacidad para captar matices relevantes en este contexto.
- **Reducción de Dimensionalidad**: Exploración de técnicas como PCA para reducir la dimensionalidad de los embeddings, mejorando la eficiencia computacional mientras se mantiene la capacidad representativa.
- **Implementación de Extracción de Keywords**: Integración de KeyBERT para identificar automáticamente palabras clave relevantes en las consultas y en las descripciones de anime, proporcionando un nivel adicional de contexto para las recomendaciones.

#### Algoritmo de Recomendación
- **Diseño del Sistema de Ranking**: Desarrollo de algoritmos para clasificar animes según su relevancia respecto a la consulta del usuario, basados principalmente en similitud coseno entre embeddings.
- **Ponderación Contextual**: Implementación de un sistema de ponderación que considera múltiples factores además de la similitud textual, como popularidad, calificación media y relevancia de géneros.
- **Estrategias de Diversificación**: Incorporación de mecanismos para asegurar diversidad en las recomendaciones, evitando resultados excesivamente homogéneos.

#### Evaluación y Validación
- **Métricas de Rendimiento**: Definición e implementación de métricas de evaluación como precisión, recall, f1-score y NDCG (Normalized Discounted Cumulative Gain) para cuantificar la calidad de las recomendaciones.
- **Validación Cruzada**: Aplicación de técnicas de validación cruzada para evaluar la robustez del modelo frente a diferentes conjuntos de datos.
- **Análisis de Casos**: Estudio detallado de casos particulares para identificar patrones de éxito y fallo, guiando mejoras posteriores.

#### Optimización para Producción
- **Serialización de Modelos**: Almacenamiento eficiente de modelos entrenados y embeddings pre-calculados utilizando Pickle para minimizar tiempos de carga.
- **Batch Processing**: Implementación de procesamiento por lotes para optimizar el uso de recursos computacionales durante la generación de embeddings.
- **Compresión de Modelos**: Investigación y aplicación de técnicas de compresión para reducir el tamaño de los modelos sin comprometer significativamente su capacidad predictiva.

#### Resultados de la Fase
- Modelo principal implementado en `modelFormer.py` capaz de entender consultas en lenguaje natural y encontrar animes semánticamente similares.
- Embeddings pre-calculados almacenados en `model/anime_embeddings.npy` para todas las entradas del dataset, permitiendo búsquedas rápidas.
- Modelo serializado en `model/anime_recommender.pt` listo para producción con tiempos de inferencia optimizados.
- Documentación técnica detallada del funcionamiento interno del modelo y sus componentes.

### Fase 3: Desarrollo de la API

Esta fase se centró en la creación de una capa de servicio robusta que expone la funcionalidad del modelo de recomendación a través de endpoints HTTP bien definidos y documentados.

#### Diseño de la API
- **Definición de Endpoints**: Diseño cuidadoso de la interfaz de la API, estableciendo endpoints intuitivos y semánticamente significativos siguiendo principios RESTful.
- **Estructuración de Parámetros**: Especificación de parámetros obligatorios y opcionales para cada endpoint, con valores predeterminados adecuados y validación rigurosa.
- **Formato de Respuestas**: Definición de esquemas JSON estandarizados para las respuestas, asegurando consistencia y facilitando la integración frontend.
- **Documentación OpenAPI**: Generación automatizada de documentación interactiva mediante Swagger UI, proporcionando ejemplos de uso y descripciones detalladas.

#### Implementación con FastAPI
- **Configuración del Framework**: Instalación y configuración de FastAPI, aprovechando su velocidad, tipado estático y generación automática de documentación.
- **Desarrollo del Punto de Entrada**: Implementación del archivo `api.py` como punto de entrada principal, definiendo la aplicación FastAPI y sus routers.
- **Endpoint de Recomendación**: Creación del endpoint `/recommend` que recibe palabras clave o descripciones y devuelve una lista ordenada de recomendaciones de anime.
- **Optimización de Carga**: Implementación de un sistema de inicialización lazy-loading que carga el modelo y los embeddings solo cuando es necesario, reduciendo el tiempo de inicio del servidor.

#### Seguridad y Control de Acceso
- **Implementación de CORS**: Configuración de Cross-Origin Resource Sharing (CORS) para permitir solicitudes desde dominios específicos, particularmente el frontend de la aplicación.
- **Rate Limiting**: Implementación de límites de tasa para prevenir abusos y garantizar la disponibilidad del servicio.
- **Validación de Entradas**: Implementación de validación rigurosa para todas las entradas de usuario, previniendo inyecciones y otros vectores de ataque.

#### Optimización de Rendimiento
- **Caché de Respuestas**: Implementación de un sistema de caché para almacenar temporalmente los resultados de consultas frecuentes, reduciendo la carga computacional.
- **Procesamiento Asíncrono**: Utilización de las capacidades asíncronas de FastAPI para manejar múltiples solicitudes concurrentes sin bloqueos.
- **Compresión de Respuestas**: Configuración de compresión Gzip para reducir el tamaño de las respuestas JSON, mejorando los tiempos de carga.

#### Pruebas y Depuración
- **Tests Unitarios**: Desarrollo de tests automatizados para validar el funcionamiento correcto de cada endpoint.
- **Tests de Integración**: Verificación de la interacción adecuada entre la API y el modelo subyacente.
- **Monitoreo de Rendimiento**: Implementación de registros detallados para tiempos de respuesta y uso de recursos, facilitando la identificación de cuellos de botella.

#### Resultados de la Fase
- API completamente funcional implementada en `api.py`, capaz de servir recomendaciones de anime basadas en consultas textuales.
- Documentación interactiva accesible a través de Swagger UI en la ruta `/docs`.
- Sistema robusto con manejo adecuado de errores, validación de entradas y optimizaciones de rendimiento.
- Capacidad para procesar múltiples solicitudes concurrentes manteniendo tiempos de respuesta óptimos.

### Fase 4: Desarrollo del Frontend

Esta fase se centró en la creación de una interfaz de usuario intuitiva, atractiva y funcional que permite a los usuarios interactuar eficientemente con el sistema de recomendación.

#### Diseño de Interfaz
- **Conceptualización Visual**: Creación de mockups y wireframes para definir la estructura general de la interfaz, priorizando la usabilidad y la experiencia del usuario.
- **Identidad Visual**: Desarrollo de un sistema de diseño coherente con una paleta de colores, tipografía y elementos visuales inspirados en la estética anime.
- **Componentes UI**: Identificación y diseño de los componentes principales como el campo de búsqueda, tarjetas de anime, selectores de filtrado y modales informativos.
- **Experiencia de Usuario**: Planificación cuidadosa del flujo de interacción, minimizando la fricción y maximizando la satisfacción del usuario.

#### Implementación con React
- **Configuración del Entorno**: Inicialización del proyecto React utilizando Vite como herramienta de construcción para un desarrollo rápido y eficiente.
- **Estructura de Componentes**: Organización del código siguiendo una arquitectura modular basada en componentes reutilizables con responsabilidades bien definidas.
- **Componente Principal**: Implementación del componente `App.jsx` como contenedor principal de la aplicación, gestionando el estado global y la lógica de alto nivel.
- **Componentes de Presentación**: Desarrollo de componentes específicos para la visualización de resultados, como tarjetas de anime y modales de detalles.

#### Gestión de Estado
- **Estado Local con Hooks**: Utilización de React Hooks (`useState`, `useEffect`) para gestionar eficientemente el estado local de los componentes.
- **Manejo de Formularios**: Implementación de un sistema robusto para gestionar entradas de usuario en el campo de búsqueda y selectores de filtrado.
- **Estados de Carga**: Gestión adecuada de los estados de carga, mostrando indicadores visuales (spinners) durante las operaciones asíncronas.
- **Persistencia de Estado**: Implementación de almacenamiento de preferencias del usuario en localStorage para mantener configuraciones entre sesiones.

#### Integración con API
- **Servicio de Comunicación**: Desarrollo de funciones para comunicarse con el backend a través de fetch API, encapsulando la lógica de solicitudes HTTP.
- **Gestión de Respuestas**: Implementación de handlers para procesar adecuadamente las respuestas del backend, incluyendo manejo de errores y timeouts.
- **Configuración de CORS**: Asegurar la correcta configuración del cliente para realizar solicitudes cross-origin al backend.
- **Optimizaciones de Rendimiento**: Implementación de debouncing en las solicitudes de búsqueda para reducir la carga en el servidor.

#### Estilización y Responsividad
- **CSS Moderno**: Aplicación de estilos utilizando propiedades CSS avanzadas como flexbox, grid, variables CSS y media queries.
- **Diseño Responsive**: Adaptación de la interfaz para funcionar óptimamente en diferentes tamaños de pantalla, desde móviles hasta monitores de alta resolución.
- **Animaciones y Transiciones**: Implementación de efectos visuales sutiles para mejorar la experiencia del usuario, como transiciones suaves entre estados.
- **Tema Anime**: Incorporación de elementos decorativos y referencias visuales al mundo del anime para enriquecer la identidad del proyecto.

#### Pruebas de Usabilidad
- **Feedback de Usuarios**: Recopilación y análisis de opiniones de usuarios reales durante pruebas preliminares.
- **Ajustes de Accesibilidad**: Implementación de mejoras para asegurar que la interfaz sea accesible para usuarios con diferentes capacidades.
- **Optimización de Flujos**: Refinamiento de los flujos de interacción basado en observaciones de comportamiento de usuarios.

#### Resultados de la Fase
- Interfaz de usuario completamente funcional implementada en React, con componentes bien estructurados y estilizados.
- Experiencia de usuario fluida con estados de carga apropiados, manejo de errores y transiciones suaves.
- Diseño visual atractivo con elementos temáticos de anime integrados sutilmente.
- Interfaz adaptativa que funciona correctamente en diferentes dispositivos y tamaños de pantalla.

### Fase 5: Integración y Pruebas

La fase final se centró en la integración de todos los componentes, pruebas exhaustivas del sistema completo, y optimizaciones basadas en datos reales de uso.

#### Integración de Componentes
- **Conexión Frontend-Backend**: Configuración definitiva de la comunicación entre la interfaz de usuario y la API de recomendación.
- **Gestión de Entornos**: Establecimiento de configuraciones específicas para entornos de desarrollo, prueba y producción.
- **Flujo de Datos End-to-End**: Verificación del correcto flujo de información desde la entrada del usuario hasta la visualización de recomendaciones.
- **Autenticación y Autorización**: Implementación de mecanismos básicos de seguridad para proteger los recursos del sistema.

#### Testing Integral
- **Pruebas de Sistema**: Verificación del funcionamiento correcto del sistema completo bajo diferentes escenarios de uso.
- **Pruebas de Regresión**: Asegurar que nuevas implementaciones no afecten negativamente a la funcionalidad existente.
- **Pruebas de Carga**: Evaluación del comportamiento del sistema bajo condiciones de alta demanda, simulando múltiples usuarios concurrentes.
- **Pruebas de Usabilidad**: Sesiones estructuradas con usuarios reales para evaluar la intuitividad y facilidad de uso de la interfaz.

#### Optimización de Rendimiento
- **Análisis de Cuellos de Botella**: Identificación de puntos críticos que afectan la velocidad o eficiencia del sistema.
- **Mejoras de Latencia**: Implementación de estrategias para reducir los tiempos de respuesta, como optimización de consultas y caché.
- **Optimización de Recursos**: Ajustes para minimizar el consumo de memoria y CPU, especialmente importante para el modelo de IA.
- **Lazy Loading**: Implementación de carga diferida para ciertos componentes y recursos, mejorando los tiempos de carga inicial.

#### Gestión de Errores
- **Sistema de Logging**: Configuración de un sistema robusto de registro de eventos y errores para facilitar el diagnóstico y solución de problemas.
- **Recuperación Graceful**: Implementación de mecanismos para manejar fallos de manera elegante, minimizando el impacto en la experiencia del usuario.
- **Feedback al Usuario**: Desarrollo de mensajes de error informativos y acciones sugeridas cuando ocurren problemas.
- **Monitoreo Proactivo**: Configuración de alertas para detectar problemas potenciales antes de que afecten significativamente al sistema.

#### Documentación Final
- **Manual de Usuario**: Creación de guías detalladas sobre el uso del sistema y sus funcionalidades.
- **Documentación Técnica**: Actualización de la documentación del código y arquitectura del sistema para facilitar mantenimiento futuro.
- **Guías de Despliegue**: Instrucciones paso a paso para la instalación y configuración del sistema en diferentes entornos.
- **Registro de Problemas Conocidos**: Documentación de limitaciones actuales y planes para abordarlas en futuras actualizaciones.

#### Resultados de la Fase
- Sistema completo y funcional con todos los componentes perfectamente integrados.
- Documentación exhaustiva tanto a nivel técnico como de usuario final.
- Pruebas que verifican la robustez, rendimiento y usabilidad del sistema.
- Plan de mejoras futuras basado en observaciones durante el proceso de integración y pruebas.

## Características

El sistema de recomendación de anime ofrece un conjunto completo de funcionalidades diseñadas para proporcionar una experiencia personalizada y enriquecedora:

- **Búsqueda Avanzada en Lenguaje Natural:** El sistema permite a los usuarios expresar sus preferencias utilizando lenguaje natural cotidiano. Los usuarios pueden ingresar descripciones elaboradas ("anime con protagonista femenina fuerte en un mundo post-apocalíptico"), conceptos abstractos ("anime melancólico con hermosos paisajes"), o simplemente mencionar temas, géneros o características específicas. Esta flexibilidad elimina la necesidad de conocer terminología especializada o títulos específicos para encontrar contenido relevante.

- **Filtrado Personalizado con Top-N:** El usuario tiene control total sobre la cantidad de resultados que desea visualizar, pudiendo seleccionar entre diferentes rangos (top 5, top 10, top 20, top 50 o top 100). Esta característica permite adaptar la experiencia según las necesidades específicas: desde una lista concisa de las mejores coincidencias hasta un catálogo extenso para explorar múltiples opciones.

- **Visualización Enriquecida de Información:** Cada recomendación incluye información comprehensiva sobre el anime, presentada de manera estructurada y fácil de asimilar. Los usuarios pueden conocer instantáneamente puntuaciones promedio, ranking global, sinopsis, estado de emisión, número de episodios, tipo de contenido (serie, película, OVA) y géneros asociados, facilitando la toma de decisiones informadas.

- **Presentación Visual Optimizada:** El sistema presenta cada recomendación en tarjetas visualmente atractivas que incluyen imágenes de portada oficiales, optimizando el reconocimiento visual. Las tarjetas están diseñadas para presentar la información de manera jerárquica, destacando primero los datos más relevantes para la decisión del usuario.

- **Interacción Modal Intuitiva:** Al hacer clic en cualquier recomendación, se despliega un modal interactivo que ofrece opciones adicionales, como la posibilidad de iniciar la visualización del anime seleccionado. Esta característica proporciona una transición fluida entre la fase de descubrimiento y la fase de consumo del contenido.

- **Interfaz Temática Inmersiva:** Todo el sistema está envuelto en una interfaz visualmente coherente con la temática anime, incluyendo elementos decorativos sutiles, animaciones contextuales, y un diseño que resonará con los aficionados del género. Esta atención al detalle estético contribuye a una experiencia de usuario más satisfactoria y envolvente.

- **Indicadores Visuales de Estado:** El sistema implementa indicadores visuales claros durante los procesos de carga, como animaciones temáticas y spinners, proporcionando feedback constante al usuario sobre el estado de sus solicitudes.

- **Codificación Visual de Información:** Utilización de esquemas de color semánticos para transmitir información adicional de manera intuitiva. Por ejemplo, diferentes colores para distinguir entre animes en emisión y finalizados, o para categorizar géneros específicos.

## Cómo Funciona

El sistema de recomendación de anime opera mediante un sofisticado proceso de análisis semántico y recuperación de información que se puede explicar en las siguientes etapas:

### 1. Procesamiento de la Consulta del Usuario

Cuando un usuario ingresa una descripción o conjunto de palabras clave:

- **Tokenización:** El texto es dividido en unidades lingüísticas significativas (tokens) utilizando técnicas avanzadas de procesamiento de lenguaje natural.
- **Normalización:** Se aplican transformaciones como conversión a minúsculas, eliminación de caracteres especiales y normalización de acentos para estandarizar el texto.
- **Extracción de Keywords:** Utilizando la biblioteca KeyBERT, el sistema identifica automáticamente los términos más relevantes y significativos dentro de la consulta.
- **Generación de Embedding:** La consulta procesada se transforma en un vector de alta dimensionalidad (768 dimensiones) utilizando el modelo transformer DistilBERT de Hugging Face. Este embedding captura la esencia semántica del texto en un espacio vectorial multidimensional donde consultas conceptualmente similares se representan como puntos cercanos.

### 2. Búsqueda y Comparación Semántica

Una vez que la consulta está representada como un vector:

- **Acceso a Embeddings Pre-calculados:** El sistema recupera los embeddings previamente generados para cada anime en la base de datos, evitando la necesidad de calcularlos en tiempo real.
- **Cálculo de Similitud:** Se computa la similitud coseno entre el embedding de la consulta y cada embedding de anime. Esta medida matemática cuantifica la similitud semántica entre dos vectores basándose en el ángulo que forman, independientemente de su magnitud.
- **Consideración de Factores Adicionales:** Además de la similitud textual directa, el algoritmo incorpora factores adicionales como la popularidad del anime, su puntuación media, y la relevancia de sus géneros respecto a preferencias detectadas en la consulta.
- **Ranking Contextual:** Se genera una lista ordenada de animes según su relevancia total combinada, considerando tanto la similitud semántica como los factores adicionales ponderados.

### 3. Presentación de Resultados

Los resultados del proceso de recomendación se presentan al usuario de manera estructurada:

- **Filtrado Top-N:** El sistema aplica el filtro seleccionado por el usuario (top 5, top 10, etc.) para mostrar sólo el número deseado de resultados.
- **Renderización de Interfaz:** Los animes seleccionados se presentan en tarjetas informativas ordenadas según su relevancia, mostrando información clave como título, imagen, sinopsis, puntuación y géneros.
- **Actualización Dinámica:** La interfaz se actualiza de manera fluida al cambiar el filtro Top-N, sin necesidad de realizar nuevas solicitudes al backend, ya que los resultados completos se mantienen en el estado de la aplicación.
- **Interactividad:** Cada tarjeta es interactiva, permitiendo al usuario obtener información adicional o tomar acciones sobre las recomendaciones presentadas.

### 4. Optimización Continua

El sistema incorpora mecanismos para mejorar constantemente la calidad de las recomendaciones:

- **Análisis de Patrones de Consulta:** El sistema registra y analiza patrones comunes en las consultas para identificar áreas de mejora en el procesamiento del lenguaje natural.
- **Refinamiento de Embeddings:** Periódicamente, los embeddings de los animes pueden ser recalculados utilizando modelos actualizados o técnicas mejoradas para incrementar la precisión semántica.
- **Evaluación de Relevancia:** Se implementan métricas para evaluar continuamente la calidad de las recomendaciones basándose en interacciones implícitas y explícitas de los usuarios.

Este flujo de trabajo integral permite al sistema proporcionar recomendaciones altamente personalizadas basadas exclusivamente en descripciones textuales, sin requerir un historial previo de visualizaciones o preferencias explícitas del usuario.

## Primeros Pasos

### Requisitos Previos
- Python 3.8 o superior
- Node.js y npm/yarn
- Entorno virtual (recomendado)

### Configuración del Backend
```bash
cd backend
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
# source venv/bin/activate
pip install -r requirements.txt
uvicorn api:app --reload
```

### Configuración del Frontend
```bash
cd frontend
npm install
npm run dev
```

La aplicación debería estar ejecutándose con el backend en http://localhost:8000 y el frontend en http://localhost:5173.

## Mejoras Futuras

- Cuentas de usuario y recomendaciones personalizadas
- Seguimiento del historial de recomendaciones
- Opciones de filtrado avanzadas por género, año, etc.
- Conjunto de datos ampliado con más títulos de anime
- Versión de aplicación móvil
- Integración con MyAnimeList u otros servicios de seguimiento de anime

---

Creado con ❤️ para fans del anime en todas partes
