import os
import json
import time
import re
import unicodedata
import string
import requests  # opcional
import httpx
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

# Similitud difusa (opcional)
try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None

# ====== Config ======
load_dotenv()

# Leer API key de Secrets (Cloud) o .env (local)
OPENROUTER_API_KEY = (
    st.secrets.get("OPENROUTER_API_KEY")
    or os.getenv("OPENROUTER_API_KEY")
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-4o-mini"

# Headers extra recomendados (no obligatorios) por OpenRouter
OPENROUTER_HEADERS_EXTRA = {
    # Si querés, seteá tu URL pública en Secrets como APP_URL
    "HTTP-Referer": st.secrets.get("APP_URL", ""),
    "X-Title": "Study Buddy",
}

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

# ==== Guía breve de uso (plegada por defecto) ====
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

# ====== Límite diario configurable por Secrets/entorno ======
def _get_int(value, default):
    try:
        return int(str(value).strip())
    except Exception:
        return default

MAX_QUESTIONS_PER_DAY = _get_int(
    st.secrets.get("MAX_QUESTIONS_PER_DAY", os.getenv("MAX_QUESTIONS_PER_DAY", 60)),
    60
)

# ====== Control de uso diario ======
today_str = datetime.utcnow().strftime("%Y-%m-%d")
if "usage_date" not in st.session_state:
    st.session_state.usage_date = today_str
if "questions_used" not in st.session_state:
    st.session_state.questions_used = 0
if st.session_state.usage_date != today_str:
    st.session_state.usage_date = today_str
    st.session_state.questions_used = 0

# ====== Helpers ======
def call_openrouter(messages, max_tokens=800, temperature=0.6, model=DEFAULT_MODEL):
    """Cliente robusto con httpx, HTTP/1.1 y reintentos para evitar errores TLS."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "No se encontró OPENROUTER_API_KEY. "
            "Cargá el secreto en Streamlit Cloud (⋮ → Settings → Secrets) o definilo en tu .env local."
        )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        **OPENROUTER_HEADERS_EXTRA,
    }
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    backoff = [0.5, 1.0, 2.0, 4.0]
    last_err = None
    for sleep_s in backoff + [0]:
        try:
            with httpx.Client(http2=False, timeout=60.0, verify=True) as client:
                r = client.post(OPENROUTER_URL, headers=headers, json=body)
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            if sleep_s:
                time.sleep(sleep_s)
            else:
                break

    raise RuntimeError(f"Fallo al llamar a OpenRouter tras reintentos: {last_err}")

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

# ====== Normalización ======
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

# ====== State ======
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

# ====== Sidebar ======
with st.sidebar:
    st.header("⚙️ Opciones")

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

    # Mostrar uso diario en la sidebar
    st.caption(f"📊 Uso diario: {st.session_state.questions_used}/{MAX_QUESTIONS_PER_DAY} preguntas")

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

# ====== Generate Questions ======
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🧪 Generar preguntas"):
        if not corpus.strip():
            st.warning("Subí al menos un archivo con contenido primero.")
        else:
            # Chequeo de cuota ANTES de invocar a la IA
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
                with st.spinner("Generando preguntas..."):
                    try:
                        content = call_openrouter(
                            [
                                {"role": "system", "content": sys_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            max_tokens=2000,
                        )
                        questions = json.loads(content)
                        assert isinstance(questions, list) and all("pregunta" in q for q in questions)
                        st.session_state.questions = questions
                        st.session_state.answers = {}
                        st.session_state.checked = {}
                        st.session_state.questions_used += int(q_count)  # sumar uso solo si salió ok
                        st.success("✅ Preguntas listas")
                    except Exception as e:
                        msg = str(e)
                        if "401" in msg:
                            st.error("🔐 401 Unauthorized: revisá tu OPENROUTER_API_KEY en Secrets.")
                        elif "429" in msg:
                            st.error("⚠️ Límite de uso de la API (429). Reducí la cantidad o intentá más tarde.")
                        else:
                            st.error(f"No se pudieron generar preguntas. Detalle: {e}")

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

                gold_norm = normalize_text(gold)
                user_norm = normalize_text(user)

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
                        p_norm = normalize_text(p)
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

        # Repasar errores (respeta tipo y usa q_count)
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
                try:
                    content = call_openrouter(
                        [
                            {"role": "system", "content": sys_prompt_repaso},
                            {"role": "user", "content": repaso_prompt},
                        ],
                        max_tokens=1500,
                    )
                    new_questions = json.loads(content)
                    assert isinstance(new_questions, list) and all("pregunta" in q for q in new_questions)
                    st.session_state.questions = new_questions
                    st.session_state.answers = {}
                    st.session_state.checked = {}
                    st.rerun()
                except Exception as e:
                    st.error(f"No se pudieron generar preguntas de repaso. Detalle: {e}")

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
st.caption("Motor de IA: OpenRouter (podés cambiar el modelo en el código)")
