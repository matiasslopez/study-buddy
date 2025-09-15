# ====== Forzar UTF-8 (Windows-safe) ======
import sys, os
try:
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys, "stdout"): sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys, "stderr"): sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
# =========================================

# ====== Imports ======
import json
import requests  # CounterAPI (counterapi.dev)
import streamlit as st
from dotenv import load_dotenv

# similitud opcional (para corrección flexible)
try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None

# ====== Config ======
load_dotenv()
GEMINI_API_KEY = (st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY"))
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"

# ====== Layout ======
st.set_page_config(page_title="📘 Study Buddy", page_icon="🤖", layout="wide")
st.title("📘 Study Buddy — Tu compañero de estudio")
st.caption("Cargá tu material, practicá preguntas y preparate para rendir al máximo 🚀")

# ====== CounterAPI (counterapi.dev) - visitas + generaciones ======
# Cambiá el namespace por algo único tuyo (evitá espacios y raros)
COUNTERAPI_NAMESPACE = "study-buddy-mati-20250915"
COUNTER_KEY_VISITS = "visits"
COUNTER_KEY_GENERAR = "click_generar_preguntas"

COUNTERAPI_BASE = "https://api.counterapi.dev/v1"  # V1 pública (sin auth)

def counter_up(namespace: str, key: str):
    """Incrementa y retorna el valor actual; si falla devuelve (None, error_str)."""
    url = f"{COUNTERAPI_BASE}/{namespace}/{key}/up"
    try:
        r = requests.get(url, timeout=6)
        r.raise_for_status()
        return r.json().get("count"), None
    except Exception as e:
        return None, f"UP {url} -> {e}"

def counter_get(namespace: str, key: str):
    """Obtiene el valor sin modificarlo; si falla devuelve (None, error_str)."""
    url = f"{COUNTERAPI_BASE}/{namespace}/{key}"
    try:
        r = requests.get(url, timeout=6)
        r.raise_for_status()
        return r.json().get("count"), None
    except Exception as e:
        return None, f"GET {url} -> {e}"

# Último error para debug
if "counter_last_error" not in st.session_state:
    st.session_state.counter_last_error = None

# Inicializar/leer visitas (si no existe, el primer 'up' la crea)
total_views, err_v = counter_get(COUNTERAPI_NAMESPACE, COUNTER_KEY_VISITS)
if err_v:
    st.session_state.counter_last_error = err_v
if total_views is None:
    _, err_u = counter_up(COUNTERAPI_NAMESPACE, COUNTER_KEY_VISITS)  # crea e incrementa
    if err_u:
        st.session_state.counter_last_error = err_u
    total_views, err_v2 = counter_get(COUNTERAPI_NAMESPACE, COUNTER_KEY_VISITS)
    if err_v2:
        st.session_state.counter_last_error = err_v2

# Leer total de generaciones (si no existe, quedará None hasta el primer clic)
total_generations, err_g = counter_get(COUNTERAPI_NAMESPACE, COUNTER_KEY_GENERAR)
if err_g:
    st.session_state.counter_last_error = err_g

# ====== Guía rápida (expandible) ======
with st.expander("❓ ¿Cómo lo uso? (guía rápida)", expanded=False):
    st.markdown(
        """
**Objetivo:** practicar con preguntas generadas a partir de tu material para medir tu preparación y recibir feedback.

**Pasos:**
1. **Cargá tu material:** clic en *Cargar archivo/s* y subí uno o varios documentos **TXT / PDF / DOCX** (máx. 5).  
2. **Configurá el cuestionario:** en la barra lateral elegí **Cantidad de preguntas**, **Tipo** (opción múltiple o desarrollo) y **Dificultad** (fácil / media / difícil).  
3. **(Opcional) Corrección flexible:** activá la casilla y ajustá el **umbral** para que sea más o menos exigente con respuestas de desarrollo.  
4. **Generá preguntas:** presioná **“🧪 Generar preguntas”**.  
5. **Respondé y verificá:** escribí tu respuesta (o elegí una opción) y tocá **“Verificar N”** para cada pregunta.  
6. **Mirá tu resultado:** al final verás tu **puntaje** y **porcentaje**, con indicaciones sobre qué reforzar.

> Tip: podés volver a generar otro set cambiando dificultad o cantidad de preguntas para entrenar distintos niveles.
"""
    )

# ====== Helpers (Gemini) ======
def call_gemini(messages, max_tokens=800, temperature=0.6, model=DEFAULT_GEMINI_MODEL):
    """
    Llama a Gemini forzando salida JSON.
    messages: [{"role":"system"/"user","content":"..."}]
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("Gemini error: falta GEMINI_API_KEY. Configurala en Secrets o .env.")
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)

        sys_txt = "\n".join([m["content"] for m in messages if m["role"] == "system"])
        usr_txt = "\n".join([m["content"] for m in messages if m["role"] == "user"])
        prompt = (
            (sys_txt + "\n" if sys_txt else "")
            + usr_txt
            + "\n\nIMPORTANTE: devolvé SOLO JSON válido (un array) sin texto extra."
        )

        gen_config = {
            "temperature": temperature,
            "response_mime_type": "application/json",
            "max_output_tokens": max(256, min(2048, int(max_tokens))),
        }

        model = genai.GenerativeModel(model)
        out = model.generate_content(prompt, generation_config=gen_config)

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
            raise RuntimeError("Gemini devolvió vacío (posible filtro de seguridad). Probá menos dificultad/menos preguntas.")
        return content
    except Exception as e:
        raise RuntimeError(f"Gemini error: {e}")

def parse_questions_strict(s: str):
    """Parsea la salida del modelo a JSON array de forma robusta."""
    s = (s or "").strip()
    if not s:
        raise ValueError("Respuesta vacía del modelo.")
    try:
        return json.loads(s)
    except Exception:
        pass
    import re
    m = re.search(r"```json\s*(\[.*?\])\s*```", s, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m2 = re.search(r"(\[.*\])", s, flags=re.S)
    if m2:
        try:
            return json.loads(m2.group(1))
        except Exception:
            pass
    raise ValueError("La respuesta del modelo no es JSON válido.")

# ====== Sidebar ======
with st.sidebar:
    st.header("⚙️ Opciones de configuración")
    q_count = st.selectbox("Cantidad de preguntas", list(range(1, 21)), index=4)
    difficulty = st.selectbox("Dificultad", ["Fácil", "Media", "Difícil"], index=0)
    q_type = st.selectbox("Tipo de pregunta", ["Opción múltiple", "Desarrollo"])
    lenient = st.toggle("Corrección flexible (texto libre)", value=True)
    threshold = st.slider("Umbral de aceptación (%)", 50, 95, 70) if lenient else 0

    # Debug opcional de CounterAPI
 #   with st.expander("🔎 Analytics (debug opcional)", expanded=False):
 #       st.write("Namespace:", COUNTERAPI_NAMESPACE)
 #       st.write("Visitas (valor leído):", total_views)
 #       st.write("Generaciones (valor leído):", total_generations)
 #       if st.session_state.counter_last_error:
 #           st.error(st.session_state.counter_last_error)

# ====== Carga de archivos ======
st.markdown(
    """
    <style>
    div[data-testid="stFileUploader"] {max-width: 520px; margin-left:auto; margin-right:auto;}
    section[data-testid="stFileUploaderDropzone"] {padding: 0.25rem 0.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
_, c2, _ = st.columns([1,2,1])
with c2:
    uploaded = st.file_uploader("📂 Cargar archivo/s (TXT, PDF, DOCX)", type=["txt","pdf","docx"], accept_multiple_files=True)

# Extraer texto
corpus = ""
if uploaded:
    texts = []
    for file in uploaded[:5]:
        name = file.name.lower()
        if file.type == "text/plain" or name.endswith(".txt"):
            texts.append(file.read().decode("utf-8", errors="ignore"))
        elif file.type == "application/pdf" or name.endswith(".pdf"):
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(file)
                pages = []
                for p in reader.pages:
                    try:
                        pages.append(p.extract_text() or "")
                    except Exception:
                        pages.append("")
                texts.append("\n".join(pages))
            except Exception:
                st.error("Falta PyPDF2. Instalá: pip install PyPDF2")
        else:
            try:
                import docx
                doc = docx.Document(file)
                texts.append("\n".join([p.text for p in doc.paragraphs if p.text]))
            except Exception:
                st.error("Falta python-docx. Instalá: pip install python-docx")
    corpus = "\n\n".join(texts)
    if corpus.strip():
        st.success(f"✅ {len(uploaded[:5])} archivo(s) cargado(s)")
    else:
        st.warning("No se pudo extraer texto de los archivos.")

# ====== Estado ======
if "questions" not in st.session_state:
    st.session_state.questions = None
    st.session_state.answers = {}
    st.session_state.checked = {}

# ====== Generar preguntas ======
if st.button("🧪 Generar preguntas"):
    # Contar click (crea el contador si no existe)
    _, err_click = counter_up(COUNTERAPI_NAMESPACE, COUNTER_KEY_GENERAR)
    if err_click:
        st.session_state.counter_last_error = err_click
    # refrescar total de generaciones
    total_generations, err_g2 = counter_get(COUNTERAPI_NAMESPACE, COUNTER_KEY_GENERAR)
    if err_g2:
        st.session_state.counter_last_error = err_g2

    if not corpus.strip():
        st.warning("Subí al menos un archivo con contenido primero.")
    else:
        sys_prompt = """
Sos un generador de preguntas para preparar exámenes.
Devolvé SIEMPRE un JSON válido con esta forma:

[
  {
    "pregunta": "texto de la consigna",
    "opciones": ["A","B","C"],     // vacío [] si es desarrollo
    "respuesta": "texto de la respuesta correcta",
    "explicacion": "explicación breve",
    "puntos_clave": ["idea1","idea2","idea3"] // para desarrollo, 3–6 ideas evaluables
  }
]
"""
        user_prompt = f"""
Texto base:
---
{corpus[:15000]}
---

Generá {q_count} preguntas de tipo {q_type} con dificultad {difficulty}.
- Para "Opción múltiple": 3–5 opciones y una correcta en "respuesta".
- Para "Desarrollo": "opciones" debe ser [], y agregá "puntos_clave" (3–6) para evaluar sin exigir literalidad.
Devolvé JSON puro (sin comentarios ni texto extra).
"""
        content = None
        try:
            content = call_gemini(
                [
                    {"role":"system","content": sys_prompt},
                    {"role":"user","content": user_prompt},
                ],
                max_tokens=2000,
                temperature=0.6,
            )
            questions = parse_questions_strict(content)
            assert isinstance(questions, list) and all("pregunta" in q for q in questions)
            st.session_state.questions = questions
            st.session_state.answers = {}
            st.session_state.checked = {}
            st.success("✅ Preguntas listas")
        except Exception as e:
            st.error(f"No se pudieron generar preguntas. Detalle: {e}")
            if isinstance(content, str):
                st.caption("Respuesta del modelo (primeros 1000 chars):")
                st.code(content[:1000])

# ====== Render del cuestionario ======
def norm_text(s: str) -> str:
    import unicodedata, string
    if not s: return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    table = str.maketrans({ch: " " for ch in string.punctuation})
    s = s.translate(table)
    s = " ".join(s.split())
    return s

def best_snippet_similarity(haystack: str, needle: str):
    if not haystack or not needle:
        return 0
    if fuzz is None:
        return 100 if needle in haystack else 0
    return fuzz.token_set_ratio(haystack, needle)

if st.session_state.questions:
    st.subheader("📚 Preguntas")
    questions = st.session_state.questions
    total = len(questions)
    score = 0.0

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
            if options:  # múltiple choice
                ok = st.session_state.answers.get(i) == q.get("respuesta", "")
                if ok:
                    st.success(f"✅ Correcto. {q.get('explicacion','')}")
                    score += 1
                else:
                    st.error(f"❌ Incorrecto. Correcta: {q.get('respuesta','')}. {q.get('explicacion','')}")
            else:  # desarrollo
                gold = (q.get('respuesta','') or '').strip()
                puntos = q.get('puntos_clave', []) or []
                user = (st.session_state.answers.get(i) or '').strip()

                gold_norm = norm_text(gold)
                user_norm = norm_text(user)

                thr = int(threshold) if lenient else 85
                passed = False
                partial = False
                details = []

                # Igualdad/contención tras normalización
                if gold_norm and user_norm:
                    if gold_norm == user_norm:
                        passed = True
                        details.append("Igualdad tras normalización.")
                    elif user_norm in gold_norm or gold_norm in user_norm:
                        passed = True
                        details.append("Contención directa entre respuesta y solución.")

                # Cobertura de puntos clave
                if not passed and puntos:
                    covered = 0
                    for p in puntos:
                        p_norm = norm_text(p)
                        hit = False
                        if p_norm and p_norm in user_norm:
                            hit = True
                        elif fuzz is not None and p_norm:
                            sim = best_snippet_similarity(user_norm, p_norm)
                            if sim >= thr:
                                hit = True
                        if hit:
                            covered += 1
                    coverage = 100 * covered / max(1, len(puntos))
                    details.append(f"Cobertura de puntos clave: {coverage:.0f}% ({covered}/{len(puntos)})")
                    if coverage >= thr:
                        passed = True
                    elif coverage >= max(40, thr - 20):
                        partial = True

                # Similitud global como respaldo
                if not passed and fuzz is not None and gold_norm and user_norm and lenient:
                    sim_global = fuzz.token_set_ratio(user_norm, gold_norm)
                    details.append(f"Similitud global (token-set): {sim_global}%")
                    if sim_global >= thr:
                        passed = True
                    elif sim_global >= max(40, thr - 10):
                        partial = True

                if passed:
                    st.success("✅ Correcto (corrección flexible)")
                    score += 1
                elif partial:
                    st.warning("🟡 Parcialmente correcto")
                    score += 0.5
                else:
                    st.error("❌ A revisar (no coincide suficiente con la respuesta esperada)")

                with st.expander("Ver criterios / solución"):
                    st.markdown("**Respuesta esperada**\n\n" + gold)
                    if puntos:
                        st.markdown("**Puntos clave**\n\n- " + "\n- ".join(puntos))
                    if details:
                        st.markdown("**Criterios**\n\n- " + "\n- ".join(details))

    if any(st.session_state.checked.values()):
        st.markdown("---")
        st.subheader("📈 Resultado")
        st.write(f"Puntaje: **{score:.1f}** / **{total}**")
        st.write(f"Porcentaje: **{(score/total)*100:.1f}%**")
        st.caption("Correcto = 1 punto; Parcial = 0.5 puntos.")

# ====== Footer ======
st.markdown("<hr/>", unsafe_allow_html=True)
from datetime import datetime
version_str = datetime.now().strftime("%Y%m%d%H%M")
views_txt = f" · 👀 Visitas totales: {total_views}" if isinstance(total_views, int) else ""
gens_txt = f" · ⚡️ Generaciones: {total_generations}" if isinstance(total_generations, int) else ""
st.markdown(
    f"<p style='text-align:center; font-size:12px; color:gray;'>"
    f"Proyecto creado por MSL · Motor de IA: Google Gemini · versión {version_str}{views_txt}{gens_txt}"
    f"</p>",
    unsafe_allow_html=True
)
