# Gatovision: Sistema de Visión Artificial

Gatovision es un sistema de visión artificial desarrollado en Python, diseñado para detectar, reconocer y analizar objetos en imágenes y videos. Utiliza técnicas de procesamiento y aprendizaje automático para ofrecer soluciones de análisis visual, integrando tanto soluciones en línea de comandos como en modo servidor.

---

## Características principales

- **Detección de objetos:** Identificación precisa de objetos en imágenes y videos.
- **Extracción de embeddings:** Generación de vectores representativos para facilitar el reconocimiento y clasificación de objetos.
- **Memoria y seguimiento:** Almacenamiento y actualización del historial de detección para el seguimiento continuo de objetos.
- **Soporte multimodal:** Operación en modo CLI para comandos directos y modo servidor para integración en aplicaciones.
- **Configuración adaptable:** Parámetros ajustables a través de archivos de configuración simples y seguros.

---

## Estructura del proyecto

```
Gatovision/
├── main.py                 # Punto de entrada principal
├── requirements.txt        # Dependencias del proyecto
├── README.md               # Documentación del proyecto
├── app
│   ├── cli.py              # Módulo de interacción en línea de comandos
│   ├── config.py           # Configuración del sistema
│   ├── detector.py         # Detección de objetos
│   ├── embeddings.py       # Manejo y procesamiento de embeddings
│   ├── memory.py           # Gestión de la memoria y el seguimiento
│   ├── recognizer.py       # Reconocimiento y clasificación de objetos
│   ├── server.py           # Modo servidor para procesamiento en tiempo real
│   ├── stream.py           # Procesamiento de secuencias de video
│   └── utils.py            # Funciones auxiliares
└── data
    ├── cats.db            # Base de datos de ejemplo
    └── ...                # Otros recursos
```

---

## Instalación y requisitos

1. **Python 3.8+**
   Instala las dependencias:
   ```sh
   pip install -r requirements.txt
   ```

2. **Configuración del entorno**
   Crea un archivo `.env` si es necesario para almacenar variables sensibles de configuración.

---

## Uso básico

1. **Modo CLI**
   Ejecuta el sistema desde la línea de comandos:
   ```sh
   python main.py --opción valor
   ```

2. **Modo Servidor**
   Inicia el servidor para procesamiento de datos en tiempo real:
   ```sh
   python app/server.py
   ```

---

## Detalles técnicos

- **app/**: Contiene la lógica principal del sistema dividida en módulos:
  - `cli.py`: Interacción vía línea de comandos.
  - `config.py`: Configuración y parámetros globales.
  - `detector.py`: Algoritmos de detección de objetos.
  - `embeddings.py`: Generación de embeddings de imagen.
  - `memory.py`: Gestión de memoria y seguimiento de objetos.
  - `recognizer.py`: Reconocimiento y clasificación.
  - `server.py`: Orquestador del modo servidor.
  - `stream.py`: Procesamiento de flujo de video.
  - `utils.py`: Funciones complementarias.
- **data/**: Almacena recursos y bases de datos para pruebas y despliegue.

---

## Personalización

- **Ajuste de parámetros:** Edita `app/config.py` para modificar la configuración del sistema.
- **Extensión de funcionalidades:** Puedes agregar nuevos módulos o mejorar los existentes según tus requerimientos.