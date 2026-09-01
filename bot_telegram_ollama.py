import getpass
import json
import time
from pathlib import Path

import joblib
import pandas as pd
import requests


# ==================================================
# RUTAS PORTABLES
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

RUTA_MODELO = BASE_DIR / "modelo_casas.pkl"
RUTA_SECTORES = BASE_DIR / "sectores_ames.json"


# ==================================================
# CARGAR RANDOM FOREST Y SECTORES
# ==================================================

print("Cargando modelo Random Forest...")

modelo = joblib.load(RUTA_MODELO)

with open(RUTA_SECTORES, "r", encoding="utf-8") as archivo:
    sectores = json.load(archivo)

print("Modelo cargado correctamente.")


# ==================================================
# MISTRAL
# ==================================================

def interpretar_mensaje(mensaje):

    prompt = f"""
Tu tarea es extraer información inmobiliaria de un mensaje.

Sectores válidos:
{sectores}

Mensaje del usuario:
"{mensaje}"

Extrae únicamente:

- sector
- metros cuadrados
- años de antigüedad

Reglas:

1. El sector debe coincidir exactamente con uno de los sectores válidos.
2. No elijas un sector solamente porque su nombre se parece.
3. Usa estas equivalencias cuando aparezcan:
   - "Old Town" = "OldTown"
   - "North Ames" = "NAmes"
   - "Northridge" = "NoRidge"
4. Si no puedes identificar con seguridad el sector, usa null.
5. Si falta algún dato, usa null.
6. No calcules ni inventes ningún precio.
7. Responde únicamente en formato JSON.

Formato esperado:

{{
    "sector": "OldTown",
    "metros": 120.0,
    "anios": 25
}}
"""

    payload = {
        "model": "mistral",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "format": "json",
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0
        }
    }

    respuesta = requests.post(
        "http://127.0.0.1:11434/api/chat",
        json=payload,
        timeout=600
    )

    respuesta.raise_for_status()

    contenido = respuesta.json()["message"]["content"]

    return json.loads(contenido)


# ==================================================
# VALIDACIÓN
# ==================================================

def validar_datos(datos):

    sector = datos.get("sector")
    metros = datos.get("metros")
    anios = datos.get("anios")

    faltantes = []

    if sector is None:
        faltantes.append("sector")

    if metros is None:
        faltantes.append("metros cuadrados")

    if anios is None:
        faltantes.append("antigüedad")

    if faltantes:
        return False, "Faltan los siguientes datos: " + ", ".join(faltantes)

    if sector not in sectores:
        return False, f"El sector '{sector}' no es válido."

    try:
        metros = float(metros)
        anios = float(anios)

    except (TypeError, ValueError):
        return False, "Los metros y la antigüedad deben ser valores numéricos."

    if metros <= 0:
        return False, "Los metros cuadrados deben ser mayores que cero."

    if anios < 0:
        return False, "La antigüedad no puede ser negativa."

    return True, {
        "sector": sector,
        "metros": metros,
        "anios": anios
    }


# ==================================================
# RANDOM FOREST
# ==================================================

def predecir_precio(datos):

    entrada = pd.DataFrame([{
        "Neighborhood": datos["sector"],
        "Area_m2": datos["metros"],
        "Antiguedad": datos["anios"]
    }])

    return modelo.predict(entrada)[0]


# ==================================================
# PROCESAR MENSAJE
# ==================================================

def procesar_mensaje(mensaje):

    datos = interpretar_mensaje(mensaje)

    print("Datos interpretados:", datos)

    valido, resultado = validar_datos(datos)

    if not valido:
        return resultado

    precio = predecir_precio(resultado)

    # La guía solicita solamente el precio cuando los datos son válidos
    return f"${precio:,.0f}"


# ==================================================
# TELEGRAM
# ==================================================

token = getpass.getpass("Token del bot de Telegram: ")

url_updates = f"https://api.telegram.org/bot{token}/getUpdates"
url_send = f"https://api.telegram.org/bot{token}/sendMessage"

offset = None

print("\n====================================")
print(" IA INMOBILIARIA INICIADA")
print("====================================")
print("Esperando mensajes de Telegram...")
print("Ctrl+C para finalizar.\n")


while True:

    try:

        parametros = {
            "timeout": 30
        }

        if offset is not None:
            parametros["offset"] = offset

        respuesta = requests.get(
            url_updates,
            params=parametros,
            timeout=40
        )

        respuesta.raise_for_status()

        actualizaciones = respuesta.json().get("result", [])

        for actualizacion in actualizaciones:

            offset = actualizacion["update_id"] + 1

            mensaje = actualizacion.get("message")

            if mensaje is None:
                continue

            chat_id = mensaje["chat"]["id"]
            texto = mensaje.get("text")

            if not texto:
                continue

            print("Mensaje recibido:", texto)

            # Comandos básicos
            if texto.lower() in ["/start", "/help"]:

                respuesta_usuario = (
                    "Envía los datos de una vivienda indicando "
                    "sector, metros cuadrados y antigüedad."
                )

            else:

                try:
                    respuesta_usuario = procesar_mensaje(texto)

                except requests.exceptions.RequestException:
                    respuesta_usuario = "Error al conectar con Ollama."

                except json.JSONDecodeError:
                    respuesta_usuario = "No se pudo interpretar el mensaje."

                except Exception as error:
                    print("Error procesando mensaje:", error)
                    respuesta_usuario = "Ocurrió un error al procesar la consulta."

            requests.post(
                url_send,
                json={
                    "chat_id": chat_id,
                    "text": respuesta_usuario
                },
                timeout=30
            )

            print("Respuesta enviada:", respuesta_usuario)
            print("------------------------------------")

    except KeyboardInterrupt:

        print("\nBot detenido.")
        break

    except requests.exceptions.RequestException as error:

        print("Problema de conexión con Telegram:", error)
        print("Reintentando en 3 segundos...")
        time.sleep(3)

    except Exception as error:

        print("Error general:", error)
        time.sleep(3)