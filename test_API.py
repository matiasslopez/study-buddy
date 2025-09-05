import os
from dotenv import load_dotenv
from openai import OpenAI

# Cargar .env
print("📂 Cargando .env...")
load_dotenv()

# Leer variable
api_key = os.getenv("OPENAI_API_KEY")
print("🔑 Valor leído de OPENAI_API_KEY:", api_key)

if not api_key:
    raise ValueError("❌ No se encontró la API key. Revisá que el archivo .env esté en la misma carpeta y bien escrito.")

# Crear cliente
client = OpenAI(api_key=api_key)

# Probar llamada
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Decime un dato curioso sobre el mate"}],
    max_tokens=50
)

print("✅ Respuesta:", resp.choices[0].message.content)
