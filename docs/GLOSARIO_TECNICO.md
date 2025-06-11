# Glosario Técnico - S.A.R. (Smart Anime Recommender)

Este glosario contiene definiciones de los términos técnicos y conceptos utilizados en el desarrollo y documentación del proyecto Smart Anime Recommender.

## A

### API (Application Programming Interface)
Conjunto de reglas y especificaciones que permite que diferentes aplicaciones se comuniquen entre sí. En S.A.R., la API es el puente entre el frontend y los sistemas de backend, permitiendo realizar búsquedas y obtener recomendaciones.

### Algoritmo de recomendación
Sistema lógico que analiza datos para ofrecer sugerencias personalizadas a los usuarios. S.A.R. utiliza algoritmos de recomendación basados en similitud de contenido y procesamiento de lenguaje natural.

## B

### Backend
Parte de un sistema que procesa la lógica de la aplicación, interactúa con bases de datos y realiza cálculos complejos. En S.A.R., el backend incluye la API y los modelos de IA que generan recomendaciones.

### Base de datos
Sistema organizado para almacenar, gestionar y recuperar información. S.A.R. utiliza PostgreSQL para almacenar datos sobre animes y métricas de uso.

## C

### Componente (React)
Piezas reutilizables e independientes de código en React que encapsulan una parte de la interfaz de usuario. En S.A.R., ejemplos de componentes son `AnimeCard`, `AnimeModal` y `MetricsPanel`.

### Conversión (tasa de)
Porcentaje de usuarios que realizan una acción deseada (como hacer clic en un anime) después de realizar una búsqueda. En S.A.R., esta métrica ayuda a medir la efectividad de las recomendaciones.

### CUDA
Plataforma de computación paralela desarrollada por NVIDIA que permite usar GPUs (tarjetas gráficas) para acelerar cálculos complejos. En S.A.R., CUDA se utiliza para optimizar el procesamiento de modelos de machine learning.

## D

### Deep Learning
Subcampo del machine learning basado en redes neuronales con múltiples capas de procesamiento. S.A.R. utiliza técnicas de deep learning para comprender mejor las preferencias de los usuarios y el contenido de los animes.

### Docker
Plataforma que permite desarrollar, enviar y ejecutar aplicaciones en contenedores, facilitando la consistencia en diferentes entornos. S.A.R. utiliza Docker para facilitar la configuración de la base de datos y otros componentes.

## E

### Embedding
Representación de datos de alta dimensión (como palabras o documentos) en un espacio vectorial de menor dimensión, preservando relaciones semánticas. En S.A.R., los embeddings se usan para convertir descripciones de anime y consultas de usuario en vectores que pueden ser comparados matemáticamente.

### Endpoints
URLs específicas en una API a las que se pueden hacer solicitudes para interactuar con el servicio. S.A.R. tiene endpoints para búsqueda de animes, registro de eventos y métricas.

## F

### Frontend
Parte de una aplicación con la que interactúan directamente los usuarios. En S.A.R., el frontend está desarrollado con React y proporciona la interfaz visual para buscar y ver recomendaciones de anime.

## H

### Hook (React)
Funciones que permiten usar estado y otras características de React sin escribir clases. En S.A.R., `useAnimeSearch` es un hook personalizado que maneja la lógica de búsqueda de animes.

## I

### IA (Inteligencia Artificial)
Campo de la informática que se centra en crear sistemas capaces de realizar tareas que requieren inteligencia humana. S.A.R. utiliza IA para comprender consultas en lenguaje natural y recomendar animes relevantes.

## K

### KPI (Key Performance Indicator)
Medidas cuantificables utilizadas para evaluar el rendimiento de un sistema en relación con sus objetivos. En S.A.R., algunos KPIs incluyen tiempos de carga, tasas de conversión y números de búsquedas diarias.

## M

### Machine Learning
Rama de la IA que permite a las computadoras aprender de datos sin ser programadas explícitamente. S.A.R. utiliza modelos de machine learning para generar recomendaciones personalizadas.

### Métricas
Medidas cuantitativas utilizadas para evaluar el rendimiento de un sistema. S.A.R. rastrea métricas como tiempo de carga, búsquedas y clics para mejorar la experiencia del usuario.

## N

### NLP (Procesamiento de Lenguaje Natural)
Rama de la IA centrada en la interacción entre computadoras y lenguaje humano. S.A.R. utiliza NLP para entender las consultas de los usuarios y extraer significado de las descripciones de anime.

## P

### PostgreSQL
Sistema de gestión de bases de datos relacional de código abierto. S.A.R. utiliza PostgreSQL para almacenar información sobre animes y métricas de uso del sistema.

### Prompts
Consultas o instrucciones dadas a un sistema de IA. En S.A.R., los prompts son las descripciones o palabras clave que los usuarios ingresan para buscar animes.

## R

### React
Biblioteca de JavaScript para construir interfaces de usuario, especialmente aplicaciones de una sola página. S.A.R. utiliza React para crear una interfaz dinámica y responsiva para los usuarios.

### Refactorización
Proceso de reestructurar código existente sin cambiar su comportamiento externo. En S.A.R., la refactorización permitió modularizar el código para hacerlo más mantenible y escalable.

## S

### Sesión
Periodo de interacción entre un usuario y una aplicación. S.A.R. genera IDs de sesión únicos para rastrear la actividad del usuario y proporcionar análisis de uso.

### Similitud coseno
Medida matemática de similitud entre dos vectores. En S.A.R., se utiliza para determinar qué tan similar es una consulta de usuario a las descripciones de animes en la base de datos.

## T

### Tokenizer
Herramienta que divide texto en unidades más pequeñas llamadas tokens (palabras, subpalabras o caracteres). En S.A.R., el tokenizer procesa tanto las consultas de los usuarios como las descripciones de los animes.

## U

### UI (User Interface)
Conjunto de elementos visuales y interactivos que permiten a los usuarios interactuar con un sistema. La UI de S.A.R. incluye campos de búsqueda, tarjetas de anime y modales de confirmación.

### UX (User Experience)
Experiencia general que tiene un usuario al interactuar con un producto. S.A.R. está diseñado para ofrecer una UX intuitiva y agradable en la búsqueda y descubrimiento de animes.

## V

### Vector
Representación matemática de datos en un espacio multidimensional. En S.A.R., las descripciones de anime y las consultas de los usuarios se convierten en vectores para poder comparar su similitud.

### Vite
Herramienta de construcción de frontend que proporciona una experiencia de desarrollo más rápida. S.A.R. utiliza Vite para optimizar el proceso de desarrollo del frontend.

---

Este glosario puede ampliarse a medida que evolucione el proyecto S.A.R. con nuevos términos y tecnologías.
