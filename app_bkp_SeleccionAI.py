# ====== Forzar UTF-8 en todo el entorno (Windows-safe) ======
import sys, os
try:
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys, "stdout"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys, "stderr"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
# ============================================================

import json
import time
import re
import unicodedata
import string
import httpx
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict

# Similitud difusa (opcional)
try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None

load_dotenv()

# ====== API Keys desde Secrets/.env ======
OPENAI_API_KEY       = st.secrets.get("OPENAI_API_KEY")       or os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY   = st.secrets.get("OPENROUTER_API_KEY")   or os.getenv("OPENROUTER_API_KEY")
GROQ_API_KEY         = st.secrets.get("GROQ_API_KEY")         or os.getenv("GROQ_API_KEY")
GEMINI_API_KEY       = st.secrets.get("GEMINI_API_KEY")       or os.getenv("GEMINI_API_KEY")
COHERE_API_KEY       = st.secrets.get("COHERE_API_KEY")       or os.getenv("COHERE_API_KEY")
HUGGINGFACE_API_KEY  = st.secrets.get("HUGGINGFACE_API_KEY")  or os.getenv("HUGGINGFACE_API_KEY")

APP_URL = (st.secrets.get("APP_URL") or os.getenv("APP_URL") or "").strip()

# ====== Endpoints / Modelos por defecto ======
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENAI_MODEL   = "gpt-4o-mini"
DEFAULT_OR_MODEL       = "anthropic/claude-3-haiku"
DEFAULT_GROQ_MODEL     = "llama3-8b-8192"          # alias en Groq
DEFAULT_GEMINI_MODEL   = "gemini-1.5-flash"
DEFAULT_COHERE_MODEL   = "command-r"
DEFAULT_HF_MODEL       = "meta-llama/Meta-Llama-3-8B-Instruct"  # requiere token HF

# ====== Layout ======
st.set_page_config(
    page_title="📘 Study Buddy - Entrenador de parciales",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("📘 Study Buddy — Tu compañero de estudio")
st.caption(
    "Te ayuda a validar tus conocimientos, desafiarte con preguntas y acompañarte para lograr tu mejor performance "
    "en una presentación o examen. Subí material (TXT/PDF/DOCX), generá preguntas y recibí feedback con corrección flexible."
)

# ====== Helpers ======
def _get_int(value, default):
    try:
        return int(str(value).strip())
    except Exception:
        return default

MAX_QUESTIONS_PER_DAY = _get_int(
    st.secrets.get("MAX_QUESTIONS_PER_DAY", os.getenv("MAX_QUESTIONS_PER_DAY", 60)),
    60
)

def get_openrouter_headers():
    h = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}" if OPENROUTER_API_KEY else "",
        "Content-Type": "application/json",
        "X-Title": "Study Buddy",
    }
    if APP_URL:
        h["HTTP-Referer"] = APP_URL
        h["Origin"] = APP_URL
    return h

# ====== Estado global ======
if "refs" not in st.session_state:
    st.session_state.refs = []
if "ref_title" not in st.session_state:
    st.session_state.ref_title = ""
if "ref_authors" not in st.session_state:
    st.session_state.ref_authors = ""
if "questions" not in st.session_state:
    st.session_state.questions = None
    st.session_state.answers = {}
    st.session_state.checked = {}
if "usage_date" not in st.session_state:
    st.session_state.usage_date = datetime.utcnow().strftime("%Y-%m-%d")
if "questions_used" not in st.session_state:
    st.session_state.questions_used = 0

# Reset diario
today_str = datetime.utcnow().strftime("%Y-%m-%d")
if st.session_state.usage_date != today_str:
    st.session_state.usage_date = today_str
    st.session_state.questions_used = 0

# ====== Health & guía ======
with st.expander("⚙️ Estado de configuración (debug)", expanded=False):
    st.write("🔑 OPENAI_API_KEY:", "✅ Sí" if OPENAI_API_KEY else "❌ No")
    st.write("🔑 OPENROUTER_API_KEY:", "✅ Sí" if OPENROUTER_API_KEY else "❌ No")
    st.write("🔑 GROQ_API_KEY:", "✅ Sí" if GROQ_API_KEY else "❌ No")
    st.write("🔑 GEMINI_API_KEY:", "✅ Sí" if GEMINI_API_KEY else "❌ No")
    st.write("🔑 COHERE_API_KEY:", "✅ Sí" if COHERE_API_KEY else "❌ No")
    st.write("🔑 HUGGINGFACE_API_KEY:", "✅ Sí" if HUGGINGFACE_API_KEY else "❌ No")
    st.write("📊 Límite diario de preguntas:", MAX_QUESTIONS_PER_DAY)

with st.expander("❓ ¿Cómo lo uso? (guía rápida)", expanded=False):
    st.markdown(
        """
**Objetivo:** practicar con preguntas generadas a partir de tu material para medir tu preparación y recibir feedback.

**Pasos:**
1. **Cargá tu material:** clic en *Cargar archivo/s* y subí uno o varios documentos **TXT / PDF / DOCX** (máx. 5).  
2. **(Opcional) Sumá bibliografía:** activá *Sumar bibliografía de Wikipedia* y/o agregá *Libros de referencia* indicando **título** y **autor/es**.  
3. **Configurá el cuestionario:** en la barra lateral elegí **Cantidad de preguntas**, **Tipo** (opción múltiple o desarrollo) y **Dificultad** (fácil / media / difícil).  
4. **Generá preguntas:** presioná **“🧪 Generar preguntas”**.  
5. **Respondé y verificá:** escribí tu respuesta (o elegí una opción) y tocá **“Verificar N”** para cada pregunta.  
6. **Mirá tu resultado:** al final verás tu **puntaje**, **porcentaje** y **sugerencias** sobre qué reforzar.  
7. **Repasá errores:** si hubo temas flojos, usá **“🔄 Repasar errores”** para generar un nuevo mini-cuestionario centrado en esos puntos.  
8. **Dejanos tu feedback:** al pie hay un link para contarnos tu experiencia y ayudarnos a mejorar.

> Tip: si cambiás opciones o bibliografía, podés volver a generar preguntas para una nueva ronda de práctica.
"""
    )

# ====== Cliente LLM multi-proveedor ======
def call_llm(messages: List[Dict[str, str]], max_tokens=800, temperature=0.6, provider="OpenAI"):
    """
    Proveedores soportados:
      - OpenAI
      - OpenRouter
      - Groq
      - Gemini
      - Cohere
      - Hugging Face
    """
    # --- OpenAI ---
    if provider == "OpenAI":
        if not OPENAI_API_KEY:
            raise RuntimeError("OpenAI error: falta OPENAI_API_KEY.")
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=OPENAI_API_KEY,
                default_headers={
                    "User-Agent": "StudyBuddy/1.0",
                    "X-Client-Name": "StudyBuddy",
                    "X-Client-Version": "1.0",
                },
            )
            resp = client.chat.completions.create(
                model=DEFAULT_OPENAI_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI error: {e}")

    # --- OpenRouter ---
    if provider == "OpenRouter":
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OpenRouter error: falta OPENROUTER_API_KEY.")
        headers = get_openrouter_headers()
        body = {
            "model": DEFAULT_OR_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            with httpx.Client(http2=False, timeout=60.0, verify=True) as c:
                r = c.post(OPENROUTER_URL, headers=headers, json=body)
            if r.status_code == 401:
                raise RuntimeError("OpenRouter error: 401 Unauthorized (modelo/clave/origen).")
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"OpenRouter error: {e}")

    # --- Groq ---
    if provider == "Groq":
        if not GROQ_API_KEY:
            raise RuntimeError("Groq error: falta GROQ_API_KEY.")
        try:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            r = client.chat.completions.create(
                model=DEFAULT_GROQ_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return r.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Groq error: {e}")

    # --- Gemini (forzar JSON) ---
    if provider == "Gemini":
        if not GEMINI_API_KEY:
            raise RuntimeError("Gemini error: falta GEMINI_API_KEY.")
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)

            model = genai.GenerativeModel(DEFAULT_GEMINI_MODEL)
            sys_txt = "\n".join([m["content"] for m in messages if m["role"] == "system"])
            usr_txt = "\n".join([m["content"] for m in messages if m["role"] == "user"])
            prompt = (
                (sys_txt + "\n" if sys_txt else "") +
                usr_txt +
                "\n\nIMPORTANTE: devolvé SOLO JSON válido (un array) sin texto extra."
            )

            gen_config = {
                "temperature": temperature,
                "response_mime_type": "application/json",
                "max_output_tokens": max(256, min(2048, int(max_tokens))),
            }

            out = model.generate_content(prompt, generation_config=gen_config)

            # Extraer texto aunque .text venga vacío
            content = None
            if hasattr(out, "text") and out.text:
                content = out.text
            elif getattr(out, "candidates", None):
                for c in out.candidates:
                    if getattr(c, "content", None) and getattr(c.content, "parts", None):
                        parts_txt = "".join([getattr(p, "text", "") for p in c.content.parts if hasattr(p, "text")])
                        if parts_txt:
                            content = parts_txt
                            break

            if not content or not content.strip():
                raise RuntimeError("Gemini devolvió vacío (posible filtro de seguridad). Probá menos dificultad/otro modelo.")

            return content
        except Exception as e:
            raise RuntimeError(f"Gemini error: {e}")

    # --- Cohere ---
    if provider == "Cohere":
        if not COHERE_API_KEY:
            raise RuntimeError("Cohere error: falta COHERE_API_KEY.")
        try:
            import cohere
            co = cohere.Client(api_key=COHERE_API_KEY)
            sys_txt = "\n".join([m["content"] for m in messages if m["role"] == "system"])
            usr_txt = "\n".join([m["content"] for m in messages if m["role"] == "user"])
            prompt = (sys_txt + "\n" if sys_txt else "") + usr_txt
            resp = co.chat(
                model=DEFAULT_COHERE_MODEL,
                message=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return getattr(resp, "text", None) or getattr(resp, "message", "") or ""
        except Exception as e:
            raise RuntimeError(f"Cohere error: {e}")

    # --- Hugging Face (Inference API) ---
    if provider == "Hugging Face":
        if not HUGGINGFACE_API_KEY:
            raise RuntimeError("HF error: falta HUGGINGFACE_API_KEY.")
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=HUGGINGFACE_API_KEY)
            sys_txt = "\n".join([m["content"] for m in messages if m["role"] == "system"])
            usr_txt = "\n".join([m["content"] for m in messages if m["role"] == "user"])
            prompt = (sys_txt + "\n" if sys_txt else "") + usr_txt
            out = client.text_generation(
                DEFAULT_HF_MODEL,
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                repetition_penalty=1.1,
                do_sample=True,
                stream=False,
            )
            return out
        except Exception as e:
            raise RuntimeError(f"Hugging Face error: {e}")

    raise RuntimeError(f"Proveedor no soportado: {provider}")

# -------- File loaders --------
@st.cache_data(show_spinner=False)
def load_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")

@st.cache_data(show_spinner=False)
def load_pdf(file) -> str:
    try:
        from PyPDF2 import PdfReader
    except Exception:
        st.error("Falta PyPDF2. Instalá: pip install PyPDF2")
        return ""
    reader = PdfReader(file)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)

@st.cache_data(show_spinner=False)
def load_docx(file) -> str:
    try:
        import docx
    except Exception:
        st.error("Falta python-docx. Instalá: pip install python-docx")
        return ""
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs if p.text])

# ====== Normalización para evaluación ======
def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    table = str.maketrans({ch: " " for ch in string.punctuation})
    s = s.translate(table)
    s = " ".join(s.split())
    return s

def best_snippet_similarity(haystack: str, needle: str):
    if not haystack or not needle:
        return 0, ""
    if fuzz is None:
        return (100, needle) if needle in haystack else (0, "")
    sim = fuzz.token_set_ratio(haystack, needle)
    snippet = (haystack[:140] + "…") if len(haystack) > 140 else haystack
    return sim, snippet

def normalize_for_eval(text: str) -> str:
    return normalize_text(text)

# ====== Parser robusto de JSON ======
def parse_questions_strict(s: str):
    s = (s or "").strip()
    if not s:
        raise ValueError("Respuesta vacía del modelo.")
    # intento directo
    try:
        return json.loads(s)
    except Exception:
        pass
    # bloque ```json ... ```
    m = re.search(r"```json\s*(\[.*?\])\s*```", s, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # primer array top-level
    m2 = re.search(r"(\[.*\])", s, flags=re.S)
    if m2:
        try:
            return json.loads(m2.group(1))
        except Exception:
            pass
    raise ValueError("La respuesta del modelo no es JSON válido.")

# ====== Sidebar ======
with st.sidebar:
    st.header("⚙️ Opciones")

    # Proveedor de IA
    st.subheader("Proveedor de IA")
    provider = st.radio(
        "Elegí proveedor",
        ["OpenAI", "OpenRouter", "Groq", "Gemini", "Cohere", "Hugging Face"],
        index=0,
        help="OpenAI recomendado; Groq/Gemini/Cohere/HF tienen opciones gratuitas/limitadas."
    )

    q_count = st.selectbox("Cantidad de preguntas", list(range(1, 21)), index=4)
    difficulty = st.selectbox("Dificultad", ["Fácil", "Media", "Difícil"], index=0)
    q_type = st.selectbox("Tipo de pregunta", ["Opción múltiple", "Desarrollo"])

    add_wiki = st.checkbox(
        "Sumar bibliografía de Wikipedia",
        value=False,
        help="Agrega resúmenes breves de Wikipedia de términos detectados",
    )

    with st.expander("📚 Libros de referencia (opcional)", expanded=False):
        st.caption("Ingresá **un libro por vez**. Indicá *título* y *autores* (como aparecen en librerías).")
        ref_title = st.text_input("Título del libro", key="ref_title")
        ref_authors = st.text_input("Autor/es", key="ref_authors")

        if st.button("➕ Agregar referencia", use_container_width=True):
            if ref_title.strip():
                st.session_state.refs.append({"titulo": ref_title.strip(), "autores": ref_authors.strip()})
                st.session_state.ref_title = ""
                st.session_state.ref_authors = ""
                st.rerun()

        if st.button("🗑️ Vaciar lista", use_container_width=True):
            st.session_state.refs = []
            st.rerun()

        if st.session_state.refs:
            st.markdown("**Referencias cargadas:**")
            for idx, r in enumerate(list(st.session_state.refs)):
                c1, c2 = st.columns([8, 1])
                with c1:
                    st.write(f"• *{r['titulo']}* — {r['autores']}")
                with c2:
                    if st.button("❌", key=f"del_ref_{idx}", help="Eliminar esta referencia"):
                        st.session_state.refs.pop(idx)
                        st.rerun()

    st.divider()
    lenient = st.checkbox("Corrección flexible (texto libre)", value=True)
    threshold = st.slider("Umbral de aceptación (%)", 50, 95, 70) if lenient else 0
    show_diag = st.checkbox("Mostrar diagnóstico detallado", value=True)

    st.caption(f"📊 Uso diario: {st.session_state.questions_used}/{MAX_QUESTIONS_PER_DAY} preguntas")
    st.caption(f"🧠 Proveedor activo: {provider}")

# ====== Uploader ======
st.markdown(
    """
    <style>
    div[data-testid="stFileUploader"] {max-width: 520px; margin-left:auto; margin-right:auto;}
    section[data-testid="stFileUploaderDropzone"] {padding: 0.25rem 0.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    uploaded_files = st.file_uploader(
        "📂 Cargar archivo/s (TXT, PDF, DOCX)",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
    )

# Cargar hasta 5 archivos
corpus = ""
if uploaded_files:
    texts = []
    for uploaded in uploaded_files[:5]:
        name = uploaded.name.lower()
        if uploaded.type == "text/plain" or name.endswith(".txt"):
            texts.append(load_txt(uploaded.read()))
        elif uploaded.type == "application/pdf" or name.endswith(".pdf"):
            texts.append(load_pdf(uploaded))
        else:
            texts.append(load_docx(uploaded))
    corpus = "\n\n".join(texts)
    if corpus.strip():
        st.success(f"✅ {len(uploaded_files[:5])} archivo(s) cargado(s)")
    else:
        st.warning("No se pudo extraer texto de los archivos.")

# ====== Wikipedia opcional ======
def wikipedia_summary(term: str, lang: str = "es") -> str:
    try:
        import wikipedia
        wikipedia.set_lang(lang)
        return wikipedia.summary(term, sentences=2)
    except Exception:
        return ""

@st.cache_data(show_spinner=False)
def enrich_with_wikipedia(text: str, max_terms: int = 3) -> str:
    candidates = re.findall(r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñA-ZÁÉÍÓÚÑ]{3,}\b", text)
    freq = {}
    for c in candidates:
        freq[c] = freq.get(c, 0) + 1
    top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:max_terms]
    extras = []
    for term, _ in top:
        s = wikipedia_summary(term)
        if s:
            extras.append(f"### {term}\n{s}")
    return "\n\n".join(extras)

# ====== Generación de preguntas ======
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🧪 Generar preguntas"):
        if not corpus.strip():
            st.warning("Subí al menos un archivo con contenido primero.")
        else:
            if st.session_state.questions_used + int(q_count) > MAX_QUESTIONS_PER_DAY:
                st.warning(
                    f"⚠️ Alcanzaste el límite diario de {MAX_QUESTIONS_PER_DAY} preguntas en esta sesión. "
                    "Reducí la cantidad o intentá mañana."
                )
            else:
                extra = ""
                if add_wiki:
                    with st.spinner("Buscando bibliografía de apoyo en Wikipedia..."):
                        extra = enrich_with_wikipedia(corpus)
                        if extra:
                            st.info("Se agregó bibliografía extra de Wikipedia.")

                sys_prompt = """
Sos un generador de preguntas para preparar exámenes.
Devolvé SIEMPRE un JSON válido con esta forma:

[
  {
    "pregunta": str,
    "opciones": [str],
    "respuesta": str,
    "explicacion": str,
    "puntos_clave": [str]
  }
]

Para "Opción múltiple": incluí 3–5 "opciones" y la correcta en "respuesta".
Para "Desarrollo": "opciones" debe ser [], y completá "puntos_clave" con 3–6 ideas que permitan evaluar sin literalidad.
"""

                refs_txt = ""
                if st.session_state.refs:
                    lines = [f"- {r['titulo']} — {r['autores']}".strip() for r in st.session_state.refs]
                    refs_txt = "Referencias adicionales:\n" + "\n".join(lines)

                user_prompt = f"""
Texto base del alumno:
---
{corpus[:15000]}
---

{('Bibliografía extra:\n' + extra) if extra else ''}

{refs_txt}

Generá {q_count} preguntas de tipo {q_type} con dificultad {difficulty}.
En las preguntas de desarrollo, incluí "puntos_clave".
Procurá incluir también preguntas que vinculen conceptos del material con las referencias adicionales cuando sea pertinente.
Devolvé JSON puro (sin comentarios ni texto extra).
"""
                content = None
                with st.spinner("Generando preguntas..."):
                    try:
                        content = call_llm(
                            [
                                {"role": "system", "content": sys_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            max_tokens=2000,
                            provider=provider,
                        )
                        questions = parse_questions_strict(content)
                        assert isinstance(questions, list) and all("pregunta" in q for q in questions)
                        st.session_state.questions = questions
                        st.session_state.answers = {}
                        st.session_state.checked = {}
                        st.session_state.questions_used += int(q_count)
                        st.success("✅ Preguntas listas")
                    except Exception as e:
                        msg = str(e)
                        st.error(f"No se pudieron generar preguntas. Detalle: {msg}")
                        if isinstance(content, str):
                            st.caption("Respuesta del modelo (primeros 1000 chars):")
                            st.code(content[:1000])

with col2:
    if st.button("🧹 Borrar todo"):
        st.session_state.questions = None
        st.session_state.answers = {}
        st.session_state.checked = {}
        st.rerun()

# ====== Render Quiz ======
if st.session_state.questions:
    st.subheader("📚 Preguntas")
    questions = st.session_state.questions
    total = len(questions)
    score = 0.0
    wrong_count = 0
    partial_count = 0
    missed_points = []
    checked_count = 0

    for i, q in enumerate(questions):
        st.markdown(f"**{i+1}. {q.get('pregunta','')}**")
        options = q.get("opciones", []) or []
        if options:
            choice = st.radio("Elegí una opción:", options, key=f"opt_{i}")
            if st.button(f"Verificar {i+1}", key=f"check_{i}"):
                st.session_state.checked[i] = True
                st.session_state.answers[i] = choice
        else:
            typed = st.text_area("Tu respuesta:", key=f"txt_{i}")
            if st.button(f"Verificar {i+1}", key=f"check_{i}"):
                st.session_state.checked[i] = True
                st.session_state.answers[i] = typed

        if st.session_state.checked.get(i):
            checked_count += 1
            is_mc = bool(options)
            if is_mc:
                ok = st.session_state.answers.get(i) == q.get("respuesta", "")
                if ok:
                    st.success(f"✅ Correcto. {q.get('explicacion','')}")
                    score += 1
                else:
                    st.error(f"❌ Incorrecto. Correcta: {q.get('respuesta','')}. {q.get('explicacion','')}")
                    wrong_count += 1
            else:
                # ====== Evaluación de desarrollo ======
                gold = (q.get('respuesta','') or '').strip()
                puntos = q.get('puntos_clave', []) or []
                user = (st.session_state.answers.get(i) or '').strip()

                gold_norm = normalize_for_eval(gold)
                user_norm = normalize_for_eval(user)

                thr = int(threshold) if lenient else 85

                passed = False
                partial = False
                details = []
                diag_rows = []
                missing = []

                # 1) Igualdad/contención
                if gold_norm and user_norm:
                    if gold_norm == user_norm:
                        passed = True
                        details.append("Igualdad tras normalización.")
                    elif user_norm in gold_norm or gold_norm in user_norm:
                        passed = True
                        details.append("Contención directa entre respuesta y solución.")

                # 2) Cobertura de puntos clave
                coverage = 0
                covered = 0
                if not passed and puntos:
                    for p in puntos:
                        p_norm = normalize_for_eval(p)
                        hit = False
                        sim = 0
                        snippet = ""
                        if p_norm and p_norm in user_norm:
                            hit = True
                            sim = 100
                            snippet = user[:140] + ("…" if len(user) > 140 else "")
                        elif fuzz is not None and p_norm:
                            sim, snippet = best_snippet_similarity(user_norm, p_norm)
                            if sim >= thr:
                                hit = True
                        diag_rows.append((p, sim, hit, snippet))
                        if hit:
                            covered += 1
                        else:
                            missing.append(p)

                    coverage = 100 * covered / max(1, len(puntos))
                    details.append(f"Cobertura de puntos clave: {coverage:.0f}% ({covered}/{len(puntos)})")
                    if coverage >= thr:
                        passed = True
                    elif coverage >= max(40, thr - 20):
                        partial = True
                    else:
                        if missing:
                            details.append("Puntos a reforzar: " + "; ".join(missing[:5]))

                # 3) Similitud global
                if not passed and fuzz is not None and gold_norm and user_norm and lenient:
                    sim_global = fuzz.token_set_ratio(user_norm, gold_norm)
                    details.append(f"Similitud global (token-set): {sim_global}%")
                    if sim_global >= thr:
                        passed = True
                    elif sim_global >= max(40, thr - 10):
                        partial = True

                # 4) Jaccard de contenido
                if not passed and gold_norm and user_norm:
                    def content_words(s):
                        return {w for w in s.split() if len(w) >= 3}
                    gw = content_words(gold_norm)
                    uw = content_words(user_norm)
                    inter = len(gw & uw)
                    union = max(1, len(gw | uw))
                    jacc = 100 * inter / union
                    details.append(f"Superposición de contenido (Jaccard): {jacc:.0f}%")
                    if jacc >= thr:
                        passed = True
                    elif jacc >= max(40, thr - 10):
                        partial = True

                if passed:
                    st.success("✅ Correcto (corrección flexible)")
                    score += 1
                elif partial:
                    st.warning("🟡 Parcialmente correcto")
                    score += 0.5
                    partial_count += 1
                else:
                    st.error("❌ A revisar (no coincide suficiente con la respuesta esperada)")
                    wrong_count += 1
                    if missing:
                        missed_points.extend(missing)

                with st.expander("Ver criterios / solución"):
                    st.markdown("**Respuesta esperada**\n\n" + gold)
                    if puntos:
                        st.markdown("**Puntos clave**\n\n- " + "\n- ".join(puntos))
                    if details:
                        st.markdown("**Criterios**\n\n- " + "\n- ".join(details))
                    if show_diag and puntos:
                        st.markdown("**Diagnóstico por punto**")
                        try:
                            import pandas as pd
                            df = pd.DataFrame([
                                {"Punto clave": p, "Similitud (%)": sim, "Cubierto": "Sí" if hit else "No", "Ejemplo": snippet}
                                for (p, sim, hit, snippet) in diag_rows
                            ])
                            st.dataframe(df, use_container_width=True)
                        except Exception:
                            for (p, sim, hit, snippet) in diag_rows:
                                st.write(f"- **{p}** → similitud: {sim}%, cubierto: {'Sí' if hit else 'No'}")
                                if snippet:
                                    st.caption(f"Ejemplo: {snippet}")

    # ---- Score final + feedback y repaso ----
    if any(st.session_state.checked.values()):
        st.markdown("---")
        st.subheader("📈 Resultado")
        st.write(f"Puntaje: **{score:.1f}** / **{total}**")
        st.write(f"Porcentaje: **{(score/total)*100:.1f}%**")
        st.caption("Regla: correcto = 1 punto; parcialmente correcto = 0.5 puntos.")

        # Feedback optimista si completó todas
        if checked_count == total:
            ratio = score / total if total else 0
            if ratio >= 0.85:
                st.success("🌟 ¡Excelente! Estás listo/a para rendir. Mantené un repaso ligero y dormí bien antes del examen.")
            elif ratio >= 0.7:
                st.info("💪 Buen trabajo. Reforzá los conceptos con menor cobertura y hacé un repaso focalizado.")
            else:
                st.warning("🚀 Vas en camino. Repasá los puntos clave marcados como faltantes y practicá 3–5 preguntas adicionales.")
            if 'missed_points' in locals() and missed_points:
                uniq = {}
                for p in missed_points:
                    if p and p.strip():
                        uniq[p.strip()] = uniq.get(p.strip(), 0) + 1
                if uniq:
                    st.markdown("**Temas a reforzar (según corrección):**")
                    for k, _ in sorted(uniq.items(), key=lambda x: x[1], reverse=True)[:8]:
                        st.write(f"• {k}")

        # Repasar errores
        if 'missed_points' in locals() and missed_points:
            if st.button("🔄 Repasar errores"):
                missed_text = "\n".join(f"- {p}" for p in sorted(set(missed_points)))
                sys_prompt_repaso = """
Sos un generador de preguntas para preparar exámenes.
Devolvé SIEMPRE un JSON válido con esta forma:

[
  {
    "pregunta": str,
    "opciones": [str],
    "respuesta": str,
    "explicacion": str,
    "puntos_clave": [str]
  }
]

Para "Opción múltiple": incluí 3–5 "opciones" y la correcta en "respuesta".
Para "Desarrollo": "opciones" debe ser [], y completá "puntos_clave" con 3–6 ideas evaluables.
"""
                repaso_prompt = f"""
Los siguientes temas fueron identificados como débiles en las respuestas del alumno:
{missed_text}

Generá {q_count} preguntas de tipo {q_type}
con dificultad {difficulty}, enfocadas específicamente en estos puntos débiles.

Devolvé JSON puro (sin texto extra).
"""
                content = None
                try:
                    content = call_llm(
                        [
                            {"role": "system", "content": sys_prompt_repaso},
                            {"role": "user", "content": repaso_prompt},
                        ],
                        max_tokens=1500,
                        provider=provider,
                    )
                    new_questions = parse_questions_strict(content)
                    assert isinstance(new_questions, list) and all("pregunta" in q for q in new_questions)
                    st.session_state.questions = new_questions
                    st.session_state.answers = {}
                    st.session_state.checked = {}
                    st.rerun()
                except Exception as e:
                    st.error(f"No se pudieron generar preguntas de repaso. Detalle: {e}")
                    if isinstance(content, str):
                        st.caption("Respuesta del modelo (primeros 1000 chars):")
                        st.code(content[:1000])

# ---- Sección de feedback (siempre visible) ----
st.markdown(
    """
    ---
    **Tu feedback nos ayuda a mejorar** 🙌  
    Por favor, dejanos tu opinión en el siguiente formulario:  
    [➡️ Abrir formulario de feedback](https://docs.google.com/forms/d/e/1FAIpQLSfe4qQG9jYyKVW1EU5zgq5c6_sFd-yiujj8gvylmxfZ9-fnAA/viewform?usp=sharing&ouid=107299619610374037951)
    """
)

# ---- Footer ----
st.markdown("<hr/>", unsafe_allow_html=True)
st.caption("Motores soportados: OpenAI · OpenRouter · Groq · Gemini · Cohere · Hugging Face")

# Footer con versión dinámica
version_str = datetime.now().strftime("%Y%m%d%H%M")
st.markdown(
    f"<p style='text-align: center; font-size: 12px; color: gray;'>"
    f"Proyecto creado por MSL, versión {version_str}"
    f"</p>",
    unsafe_allow_html=True
)
