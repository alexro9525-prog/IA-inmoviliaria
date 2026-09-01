# IA Inmobiliaria con Random Forest, Ollama, Mistral y Telegram

## Objetivo

Desarrollar un sistema de Inteligencia Artificial capaz de estimar el precio de una vivienda a partir de un mensaje escrito en lenguaje natural enviado mediante Telegram.

El sistema combina:

- Random Forest para realizar la predicción del precio.
- Ollama + Mistral para interpretar el mensaje del usuario.
- Telegram como interfaz de comunicación.
- Python para integrar todos los componentes.

## Arquitectura

Usuario
↓
Telegram
↓
Python
↓
Ollama + Mistral
↓
Extracción de sector, metros cuadrados y antigüedad
↓
Random Forest
↓
Predicción del precio
↓
Telegram

## Dataset

Se utiliza el dataset Ames Housing disponible mediante OpenML.

Variables utilizadas:

- Neighborhood: sector de la vivienda.
- GrLivArea: área habitable.
- YearBuilt: año de construcción.
- YrSold: año de venta.
- SalePrice: precio de venta.

Para el modelo se utilizan finalmente:

- Neighborhood
- Area_m2
- Antiguedad

La variable objetivo es:

- SalePrice

## Ejemplo de consulta

Tengo una casa en Old Town de 120 metros cuadrados y 25 años.

Respuesta esperada:

$171,393

El valor puede variar dependiendo del modelo entrenado.

## Ejecución

1. Instalar las dependencias:

```bash
pip install -r requirements_local.txt
