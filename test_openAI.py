from dotenv import load_dotenv
import os
from openai import OpenAI

# 1. Cargar variables de entorno desde .env
load_dotenv()

# 2. Obtener la API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ No se encontró la API Key. Revisá tu archivo .env")

print("✅ API KEY DETECTADA")

# 3. Inicializar cliente
client = OpenAI(api_key=api_key)

# 4. Probar un request simple
try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hola! ¿Podés responder con un emoji?"}],
        max_tokens=20
    )
    print("🤖 Respuesta del modelo:", response.choices[0].message["content"])
except Exception as e:
    print("❌ Error al llamar a la API:", e)
