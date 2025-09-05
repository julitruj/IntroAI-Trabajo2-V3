import streamlit as st
import json
import pandas as pd
import time
from datetime import datetime
import uuid
import zipfile
import io
import os
from typing import List, Dict, Any, Optional
import PyPDF2
from llm_backend import LLMBackend, ModelConfig

# Configuración de la página
st.set_page_config(
    page_title="VoC Analyst - Análisis de Voz del Cliente con LLM",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar el estado de la sesión
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'run_id' not in st.session_state:
    st.session_state.run_id = None
if 'uploaded_files_data' not in st.session_state:
    st.session_state.uploaded_files_data = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

def extract_text_from_pdf(pdf_file) -> str:
    """Extraer texto de archivo PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error al extraer texto de PDF: {str(e)}")
        return ""

def validate_file_size(file) -> bool:
    """Validar que el tamaño del archivo sea menor a 100MB"""
    file.seek(0, 2)  # Mover al final del archivo
    size = file.tell()
    file.seek(0)  # Reiniciar al inicio
    return size <= 100 * 1024 * 1024  # 100MB

def process_uploaded_files(uploaded_files) -> List[Dict[str, Any]]:
    """Procesar archivos subidos y extraer contenido de texto"""
    processed_files = []
    
    for file in uploaded_files:
        if not validate_file_size(file):
            st.error(f"El archivo {file.name} excede el límite de 100MB")
            continue
            
        try:
            if file.type == "text/plain":
                content = str(file.read(), "utf-8")
            elif file.type == "application/pdf":
                content = extract_text_from_pdf(file)
            else:
                st.error(f"Tipo de archivo no soportado: {file.type}")
                continue
                
            if content.strip():
                processed_files.append({
                    "filename": file.name,
                    "content": content,
                    "size": len(content),
                    "type": file.type
                })
            else:
                st.error(f"No se encontró contenido de texto en {file.name}")
                
        except Exception as e:
            st.error(f"Error al procesar {file.name}: {str(e)}")
            
    return processed_files

def display_kpis(kpis: Dict[str, Any]):
    """Mostrar KPIs en formato de panel"""
    st.subheader("📊 Indicadores Clave de Desempeño")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        nps_data = kpis.get('nps', {})
        nps_value = nps_data.get('value', 0)
        st.metric(
            label="Net Promoter Score (NPS)",
            value=f"{nps_value}",
            help=f"Promotores: {nps_data.get('promoters', 0)}%, Detractores: {nps_data.get('detractors', 0)}%, Pasivos: {nps_data.get('passives', 0)}%"
        )
        if nps_data.get('simulated'):
            st.caption("*Simulado por LLM")
    
    with col2:
        csat_data = kpis.get('csat', {})
        csat_value = csat_data.get('mean', 0)
        st.metric(
            label="Satisfacción del Cliente (CSAT)",
            value=f"{csat_value:.1f}/5.0",
        )
        if csat_data.get('simulated'):
            st.caption("*Simulado por LLM")
    
    with col3:
        sentiment_data = kpis.get('sentiment', {})
        pos_avg = sentiment_data.get('pos', 0)
        st.metric(
            label="Sentimiento Positivo",
            value=f"{pos_avg:.2f}",
            help=f"Puntajes promedio de sentimiento - Negativo: {sentiment_data.get('neg', 0):.2f}, Neutral: {sentiment_data.get('neu', 0):.2f}"
        )

def display_topics(topics: List[Dict[str, Any]]):
    """Mostrar análisis de temas"""
    st.subheader("🏷️ Temas Descubiertos")
    
    if not topics:
        st.info("No se descubrieron temas en el análisis.")
        return
    
    # Crear DataFrame para los temas
    topics_df = pd.DataFrame([
        {
            "ID de Tema": topic['topic_id'],
            "Etiqueta": topic['label'],
            "Palabras Clave": ", ".join(topic.get('keywords', [])),
            "Descripción": topic['description'][:100] + "..." if len(topic['description']) > 100 else topic['description']
        }
        for topic in topics
    ])
    
    st.dataframe(topics_df, width='stretch')
    
    # Detalles de temas en secciones expandibles
    st.subheader("📝 Resúmenes de Temas")
    for topic in topics:
        with st.expander(f"Tema {topic['topic_id']}: {topic['label']}"):
            st.write(f"**Descripción:** {topic['description']}")
            st.write(f"**Palabras Clave:** {', '.join(topic.get('keywords', []))}")
            st.write(f"**Resumen:** {topic.get('summary', 'No hay resumen disponible')}")
            
            bullets = topic.get('bullets', [])
            if bullets:
                st.write("**Puntos Clave:**")
                for bullet in bullets:
                    st.write(bullet)

def display_recommendations(recommendations: List[Dict[str, Any]]):
    """Mostrar recomendaciones SMART"""
    st.subheader("💡 Recomendaciones SMART")
    
    if not recommendations:
        st.info("No se generaron recomendaciones.")
        return
    
    # Agrupar por tema
    topic_recs = {}
    for rec in recommendations:
        topic_id = rec.get('topic_id', 'Desconocido')
        if topic_id not in topic_recs:
            topic_recs[topic_id] = []
        topic_recs[topic_id].append(rec)
    
    for topic_id, recs in topic_recs.items():
        st.write(f"**Recomendaciones del Tema {topic_id}:**")
        
        for i, rec in enumerate(recs, 1):
            with st.expander(f"Recomendación {i}: {rec.get('what', 'Sin título')[:50]}..."):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Qué:** {rec.get('what', 'No especificado')}")
                    st.write(f"**Quién:** {rec.get('who', 'No especificado')}")
                    st.write(f"**Cuándo:** {rec.get('when', 'No especificado')}")
                
                with col2:
                    st.write(f"**Métrica:** {rec.get('metric', 'No especificado')}")
                    st.write(f"**Impacto:** {rec.get('impact', 'No especificado')}")
                    
                    tag = rec.get('tag', 'Desconocido')
                    tag_colors = {
                        'quick win': '🟢',
                        'proceso': '🔵', 
                        'producto': '🟠',
                        'formación': '🟡',
                        'política': '🔴'
                    }
                    st.write(f"**Etiqueta:** {tag_colors.get(tag, '⚪')} {tag}")

def display_message_assignments(assignments: List[Dict[str, Any]], conversations: List[Dict[str, Any]]):
    """Mostrar muestra de asignaciones de mensajes"""
    st.subheader("💬 Muestra de Análisis de Mensajes")
    
    if not assignments:
        st.info("No hay asignaciones de mensajes disponibles.")
        return
    
    # Mostrar primeras 10 asignaciones como muestra
    sample_assignments = assignments[:10]
    
    assignments_df = pd.DataFrame([
        {
            "ID de Conversación": assign.get('conversation_id', 'Desconocido'),
            "ID de Tema": assign.get('topic_id', 'Desconocido'),
            "Sentimiento": assign.get('sentiment_label', 'Desconocido'),
            "Puntaje de Sentimiento": f"{assign.get('sentiment_score', 0):.2f}",
            "Emoción (GEW)": assign.get('familia_gew', 'Desconocido'),
            "Intensidad de Emoción": f"{assign.get('intensidad', 0)}/5",
            "Valencia": f"{assign.get('valencia', 0):.2f}"
        }
        for assign in sample_assignments
    ])
    
    st.dataframe(assignments_df, width='stretch')
    
    if len(assignments) > 10:
        st.caption(f"Mostrando 10 de {len(assignments)} asignaciones de mensajes en total. Datos completos disponibles en la exportación.")

def export_results(results: Dict[str, Any], run_id: str):
    """Crear archivos de exportación"""
    st.subheader("📥 Exportar Resultados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Exportar JSON
        json_str = json.dumps(results, indent=2, ensure_ascii=False)
        st.download_button(
            label="📄 Descargar Resultados Completos (JSON)",
            data=json_str,
            file_name=f"voc_analysis_{run_id}.json",
            mime="application/json"
        )
    
    with col2:
        # Exportar CSV para temas
        if results.get('topics'):
            topics_df = pd.DataFrame([
                {
                    "ID de Tema": topic['topic_id'],
                    "Etiqueta": topic['label'],
                    "Descripción": topic['description'],
                    "Palabras Clave": ", ".join(topic.get('keywords', [])),
                    "Resumen": topic.get('summary', ''),
                }
                for topic in results['topics']
            ])
            
            csv_data = topics_df.to_csv(index=False)
            st.download_button(
                label="📊 Descargar Temas (CSV)",
                data=csv_data,
                file_name=f"voc_topics_{run_id}.csv",
                mime="text/csv"
            )

# Interfaz principal
st.title("📊 VoC Analyst - Análisis con LLM")
st.write("Transforma transcripciones de interacciones con clientes en información accionable usando análisis avanzado con LLM")

# Barra lateral de configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Selección de modelo
    st.subheader("🤖 Selecciona el Modelo LLM")
    
    provider = st.selectbox(
        "Proveedor",
        ["OpenAI", "Anthropic", "Gemini"],
        key="provider_select"
    )
    
    # Opciones de modelo según proveedor
    model_options = {
        "OpenAI": [
            "gpt-5",
            "gpt-4.1",
            "gpt-4.1-mini", 
            "gpt-4o",
            "o1-mini"
        ],
        "Anthropic": [
            "claude-sonnet-4-20250514",
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20241022"
        ],
        "Gemini": [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash"
        ]
    }
    
    model = st.selectbox(
        "Modelo",
        model_options[provider],
        key="model_select"
    )
    
    # Entrada de API Key (con variable de entorno como respaldo)
    api_key_labels = {
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY", 
        "Gemini": "GEMINI_API_KEY"
    }
    
    api_key_env = api_key_labels[provider]
    api_key = st.text_input(
        f"Clave API de {provider}",
        type="password",
        value=os.getenv(api_key_env, ""),
        help=f"Ingrese su clave API de {provider} o configure la variable de entorno {api_key_env}"
    )
    
    if not api_key:
        st.warning(f"Por favor proporcione la clave API de {provider} para continuar")

# Área de contenido principal
tab1, tab2, tab3 = st.tabs(["📁 Subir y Procesar", "📊 Panel", "📥 Exportar"])

with tab1:
    st.header("Subir Archivos de Interacciones con Clientes")
    st.write("Suba archivos TXT o PDF que contengan conversaciones cliente-agente. Cada archivo debe contener exactamente una interacción.")
    
    # Cargador de archivos
    uploaded_files = st.file_uploader(
        "Seleccionar archivos",
        type=['txt', 'pdf'],
        accept_multiple_files=True,
        help="Suba archivos .txt o .pdf (máx 100MB cada uno). Cada archivo debe contener una interacción de cliente."
    )
    
    if uploaded_files:
        st.write(f"📁 {len(uploaded_files)} archivo(s) cargado(s)")
        
        # Procesar archivos
        if st.button("🔍 Procesar Archivos", disabled=not api_key):
            with st.spinner("Procesando archivos cargados..."):
                processed_files = process_uploaded_files(uploaded_files)
                st.session_state.uploaded_files_data = processed_files
                
            if processed_files:
                st.success(f"✅ {len(processed_files)} archivos procesados exitosamente")
                
                # Mostrar resumen de archivos
                files_df = pd.DataFrame([
                    {
                        "Archivo": f['filename'],
                        "Tipo": f['type'],
                        "Tamaño (caracteres)": f['size']
                    }
                    for f in processed_files
                ])
                st.dataframe(files_df, width='stretch')

    # Mostrar botón de análisis si hay archivos procesados en estado de sesión
    if st.session_state.uploaded_files_data:
        st.write("📋 Archivos listos para análisis:")
        
        # Mostrar resumen de archivos procesados
        files_summary_df = pd.DataFrame([
            {
                "Archivo": f['filename'],
                "Tipo": f['type'],
                "Tamaño (caracteres)": f['size']
            }
            for f in st.session_state.uploaded_files_data
        ])
        st.dataframe(files_summary_df, width='stretch')
        
        # Botón de análisis - siempre disponible cuando hay archivos procesados
        if st.button("🚀 Analizar con LLM", disabled=not api_key):
            processed_files = st.session_state.uploaded_files_data
            if not api_key:
                st.error("Se requiere clave API para el análisis")
            else:
                # Crear configuración del modelo
                model_config = ModelConfig(
                    provider=provider.lower(),
                    model=model,
                    api_key=api_key
                )
                
                # Inicializar backend LLM
                backend = LLMBackend(model_config)
                
                # Generar ID de ejecución
                run_id = str(uuid.uuid4())[:8]
                st.session_state.run_id = run_id
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Optimizado: Llamada única al LLM para procesamiento en lote
                    status_text.text("🚀 Procesando todos los archivos con LLM en lote (más rápido)...")
                    progress_bar.progress(50)
                    
                    st.write(f"🔬 Analizando {len(processed_files)} archivos en lote...")
                    
                    # Usar procesamiento en lote para mejor rendimiento
                    analysis_results = backend.analyze_conversations_batch(processed_files)
                    
                    if analysis_results and isinstance(analysis_results, dict):
                        progress_bar.progress(100)
                        status_text.text("✅ ¡Análisis completo!")
                        
                        # Guardar resultados
                        st.session_state.analysis_results = analysis_results
                        st.session_state.processing_complete = True
                        
                        # Mostrar resumen
                        topics_count = len(analysis_results.get('topics', []))
                        recs_count = len(analysis_results.get('recommendations', []))
                        
                        st.success(f"🎉 ¡Análisis completado exitosamente! ID de ejecución: {run_id}")
                        st.write(f"📊 Se encontraron {topics_count} temas y {recs_count} recomendaciones")
                        st.info("👉 Revise la pestaña Panel para ver resultados")
                    else:
                        st.error("❌ Falló el análisis - No se devolvieron resultados válidos")
                            
                except Exception as e:
                    st.error(f"❌ Falló el análisis: {str(e)}")
                    import traceback
                    st.text("Información de depuración:")
                    st.code(traceback.format_exc())
                finally:
                    progress_bar.empty()
                    status_text.empty()

with tab2:
    st.header("Panel de Análisis")
    
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        # Mostrar KPIs
        if 'kpis' in results:
            display_kpis(results['kpis'])
        
        st.divider()
        
        # Mostrar Temas
        if 'topics' in results:
            display_topics(results['topics'])
        
        st.divider()
        
        # Mostrar Recomendaciones
        if 'recommendations' in results:
            display_recommendations(results['recommendations'])
        
        st.divider()
        
        # Mostrar Muestra de Asignaciones de Mensajes
        if 'message_assignments' in results:
            display_message_assignments(
                results['message_assignments'],
                results.get('conversations', [])
            )
    else:
        st.info("📈 No hay resultados de análisis disponibles. Por favor cargue y procese archivos primero.")

with tab3:
    st.header("Exportar Resultados")
    
    if st.session_state.analysis_results and st.session_state.run_id:
        export_results(st.session_state.analysis_results, st.session_state.run_id)
    else:
        st.info("📦 No hay resultados disponibles para exportar. Por favor complete un análisis primero.")

# Pie de página
st.divider()
st.caption("VoC Analyst - Impulsado por tecnología LLM para un análisis integral de la voz del cliente")


