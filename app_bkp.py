
    import os
    import json
    import streamlit as st
    from openai import OpenAI

    # --- OpenAI client via secrets or env ---
    # In Streamlit Community Cloud, set the secret OPENAI_API_KEY in the app settings.
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

    client = OpenAI()

    st.set_page_config(page_title="Bot de Práctica para Exámenes", page_icon="🤖")
    st.title("🤖 Bot de Práctica para Exámenes")
    st.write("Subí tus apuntes en texto y generá preguntas para practicar.")

    # Subida de archivo
    uploaded_file = st.file_uploader("Subí un archivo de texto (.txt)", type=["txt"])

    if uploaded_file:
        text = uploaded_file.read().decode("utf-8", errors="ignore")
        st.success("Archivo cargado correctamente ✅")

        # Selección de modo
        mode = st.selectbox(
            "Seleccioná el tipo de preguntas:",
            ["Preguntas de opción múltiple", "Preguntas de desarrollo", "Definiciones cortas"],
        )

        if st.button("Generar preguntas"):
            # Prompt al modelo
            prompt = f"""
            A partir del siguiente texto:

            {text}

            Generá 5 {mode.lower()} para practicar un examen.
            Formato JSON con esta estructura:
            [
              {{"pregunta": "...", "opciones": ["..."], "respuesta": "...", "explicacion": "..."}}
            ]
            Si no aplica opciones (ej. desarrollo o definiciones), dejá "opciones": []
            """

            with st.spinner("Generando preguntas..."):
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Sos un generador de preguntas para estudio."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=1100,
                )

            content = response.choices[0].message.content

            try:
                questions = json.loads(content)
            except Exception:
                st.error("No se pudieron parsear las preguntas. Intentá de nuevo.")
                st.stop()

            st.session_state["questions"] = questions
            st.session_state["answers"] = {}
            st.session_state["checked"] = {}

    # Render de preguntas (si existen)
    if "questions" in st.session_state:
        st.subheader("📚 Preguntas:")
        questions = st.session_state["questions"]

        correct_count = 0
        total = len(questions)

        for i, q in enumerate(questions):
            st.markdown(f"**{i+1}. {q['pregunta']}**")

            # Opción múltiple
            if q.get("opciones"):
                selected = st.radio(
                    "Elegí una opción:",
                    q["opciones"],
                    key=f"radio_{i}",
                    index=None,
                )
                if st.button(f"Verificar respuesta {i+1}", key=f"check_{i}"):
                    st.session_state["answers"][i] = selected
                    st.session_state["checked"][i] = True

                if st.session_state["checked"].get(i):
                    correct = selected == q["respuesta"]
                    if correct:
                        st.success(f"✅ Correcto. {q['explicacion']}")
                    else:
                        st.error(f"❌ Incorrecto. Respuesta correcta: {q['respuesta']}. {q['explicacion']}")

            # Desarrollo / definiciones
            else:
                typed = st.text_area("Tu respuesta:", key=f"text_{i}")
                if st.button(f"Verificar respuesta {i+1}", key=f"check_text_{i}"):
                    st.session_state["answers"][i] = typed
                    st.session_state["checked"][i] = True

                if st.session_state["checked"].get(i):
                    st.info(f"Respuesta esperada: {q['respuesta']}\n\nExplicación: {q['explicacion']}")

        # Mostrar score si todas las de opción múltiple fueron verificadas
        # (simplemente cuenta las correctas que ya fueron chequeadas)
        for i, q in enumerate(questions):
            if q.get("opciones") and st.session_state["checked"].get(i):
                correct_count += int(st.session_state["answers"].get(i) == q["respuesta"])

        if any(st.session_state["checked"].values()):
            st.markdown("---")
            st.subheader("📈 Resultado parcial")
            st.write(f"Aciertos: **{correct_count}** sobre **{total}** (incluye solo las verificadas).")
            if total > 0:
                st.write(f"Porcentaje: **{(correct_count / total) * 100:.1f}%**")

    st.markdown("""

    ---
    **Cómo configurar tu API Key:** En Streamlit Cloud, abrí `Settings → Secrets` y agregá:


```
OPENAI_API_KEY = "sk-..."
```

    """)
