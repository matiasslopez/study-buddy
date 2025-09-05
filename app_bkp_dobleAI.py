import streamlit as st
from dotenv import load_dotenv
import os
from openai import OpenAI
import requests

# -------------------------------
# 1. Cargar variables de entorno
# -------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # Plan B

# -------------------------------
# 2. Configuración inicial
# -------------------------------
st.set_page_config(page_title="📘 Study Bot", page_icon="🤖")

st.title("📘 Study Bot - Tu entrenador de parciales")
st.write("Subí un archivo de texto y hacé preguntas para practicar antes del examen.")

uploaded_file = st.file_uploader("📂 Subí tu archivo de estudio (.txt)", type="txt")
user_question = st.text_input("✍️ Escribí tu pregunta")

# -------------------------------
# 3. Función para llamar a OpenAI
# -------------------------------
def query_openai(prompt):
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400
        )
        return response.choices[0].message["content"]
    except Exception as e:
        raise RuntimeError(f"❌ OpenAI Error: {e}")

# -------------------------------
# 4. Función para llamar a OpenRouter
# -------------------------------
def query_openrouter(prompt):
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "openai/gpt-4o-mini",  # También podés usar "anthropic/claude-3.5-sonnet" u otros
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 400
        }
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"❌ OpenRouter Error: {e}")

# -------------------------------
# 5. Lógica principal
# -------------------------------
if st.button("🤔 Generar respuesta"):
    if uploaded_file is None or not user_question.strip():
        st.warning("⚠️ Tenés que subir un archivo y escribir una pregunta.")
    else:
        try:
            # Leer archivo en UTF-8
            file_text = uploaded_file.read().decode("utf-8", errors="ignore")
            prompt = f"""
            Tenés el siguiente material de estudio:

            {file_text}

            El estudiante pregunta: {user_question}

            Generá una respuesta clara, en español, como si fueras un profesor preparando para el parcial.
            """

            # 1. Intentar con OpenAI
            if OPENAI_API_KEY:
                try:
                    answer = query_openai(prompt)
                    st.success("✅ Respuesta generada con OpenAI")
                    st.write(answer)
                except Exception as e:
                    st.error(e)
                    if OPENROUTER_API_KEY:
                        st.info("🔄 Probando con OpenRouter...")
                        answer = query_openrouter(prompt)
                        st.success("✅ Respuesta generada con OpenRouter")
                        st.write(answer)
                    else:
                        st.error("❌ No se pudo generar respuesta. Revisá tus API Keys.")
            # 2. Directo a OpenRouter si no hay API Key de OpenAI
            elif OPENROUTER_API_KEY:
                answer = query_openrouter(prompt)
                st.success("✅ Respuesta generada con OpenRouter")
                st.write(answer)
            else:
                st.error("⚠️ No configuraste ninguna API Key (ni OpenAI ni OpenRouter).")
        except Exception as e:
            st.error(f"Error al generar respuesta: {e}")
