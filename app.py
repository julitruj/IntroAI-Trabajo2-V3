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

# Page configuration
st.set_page_config(
    page_title="VoC Analyst - LLM-Powered Voice of Customer Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'run_id' not in st.session_state:
    st.session_state.run_id = None
if 'uploaded_files_data' not in st.session_state:
    st.session_state.uploaded_files_data = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def validate_file_size(file) -> bool:
    """Validate file size is under 100MB"""
    file.seek(0, 2)  # Move to end of file
    size = file.tell()
    file.seek(0)  # Reset to beginning
    return size <= 100 * 1024 * 1024  # 100MB

def process_uploaded_files(uploaded_files) -> List[Dict[str, Any]]:
    """Process uploaded files and extract text content"""
    processed_files = []
    
    for file in uploaded_files:
        if not validate_file_size(file):
            st.error(f"File {file.name} exceeds 100MB limit")
            continue
            
        try:
            if file.type == "text/plain":
                content = str(file.read(), "utf-8")
            elif file.type == "application/pdf":
                content = extract_text_from_pdf(file)
            else:
                st.error(f"Unsupported file type: {file.type}")
                continue
                
            if content.strip():
                processed_files.append({
                    "filename": file.name,
                    "content": content,
                    "size": len(content),
                    "type": file.type
                })
            else:
                st.error(f"No text content found in {file.name}")
                
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            
    return processed_files

def display_kpis(kpis: Dict[str, Any]):
    """Display KPIs in a dashboard format"""
    st.subheader("📊 Key Performance Indicators")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        nps_data = kpis.get('nps', {})
        nps_value = nps_data.get('value', 0)
        st.metric(
            label="Net Promoter Score (NPS)",
            value=f"{nps_value}",
            help=f"Promoters: {nps_data.get('promoters', 0)}%, Detractors: {nps_data.get('detractors', 0)}%, Passives: {nps_data.get('passives', 0)}%"
        )
        if nps_data.get('simulated'):
            st.caption("*Simulated by LLM")
    
    with col2:
        csat_data = kpis.get('csat', {})
        csat_value = csat_data.get('mean', 0)
        st.metric(
            label="Customer Satisfaction (CSAT)",
            value=f"{csat_value:.1f}/5.0",
        )
        if csat_data.get('simulated'):
            st.caption("*Simulated by LLM")
    
    with col3:
        sentiment_data = kpis.get('sentiment', {})
        pos_avg = sentiment_data.get('pos', 0)
        st.metric(
            label="Positive Sentiment",
            value=f"{pos_avg:.2f}",
            help=f"Average sentiment scores - Negative: {sentiment_data.get('neg', 0):.2f}, Neutral: {sentiment_data.get('neu', 0):.2f}"
        )

def display_topics(topics: List[Dict[str, Any]]):
    """Display topics analysis"""
    st.subheader("🏷️ Discovered Topics")
    
    if not topics:
        st.info("No topics discovered in the analysis.")
        return
    
    # Create DataFrame for topics
    topics_df = pd.DataFrame([
        {
            "Topic ID": topic['topic_id'],
            "Label": topic['label'],
            "Keywords": ", ".join(topic.get('keywords', [])),
            "Description": topic['description'][:100] + "..." if len(topic['description']) > 100 else topic['description']
        }
        for topic in topics
    ])
    
    st.dataframe(topics_df, width='stretch')
    
    # Topic details in expandable sections
    st.subheader("📝 Topic Summaries")
    for topic in topics:
        with st.expander(f"Topic {topic['topic_id']}: {topic['label']}"):
            st.write(f"**Description:** {topic['description']}")
            st.write(f"**Keywords:** {', '.join(topic.get('keywords', []))}")
            st.write(f"**Summary:** {topic.get('summary', 'No summary available')}")
            
            bullets = topic.get('bullets', [])
            if bullets:
                st.write("**Key Points:**")
                for bullet in bullets:
                    st.write(bullet)

def display_recommendations(recommendations: List[Dict[str, Any]]):
    """Display SMART recommendations"""
    st.subheader("💡 SMART Recommendations")
    
    if not recommendations:
        st.info("No recommendations generated.")
        return
    
    # Group by topic
    topic_recs = {}
    for rec in recommendations:
        topic_id = rec.get('topic_id', 'Unknown')
        if topic_id not in topic_recs:
            topic_recs[topic_id] = []
        topic_recs[topic_id].append(rec)
    
    for topic_id, recs in topic_recs.items():
        st.write(f"**Topic {topic_id} Recommendations:**")
        
        for i, rec in enumerate(recs, 1):
            with st.expander(f"Recommendation {i}: {rec.get('what', 'No title')[:50]}..."):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**What:** {rec.get('what', 'Not specified')}")
                    st.write(f"**Who:** {rec.get('who', 'Not specified')}")
                    st.write(f"**When:** {rec.get('when', 'Not specified')}")
                
                with col2:
                    st.write(f"**Metric:** {rec.get('metric', 'Not specified')}")
                    st.write(f"**Impact:** {rec.get('impact', 'Not specified')}")
                    
                    tag = rec.get('tag', 'Unknown')
                    tag_colors = {
                        'quick win': '🟢',
                        'proceso': '🔵', 
                        'producto': '🟠',
                        'formación': '🟡',
                        'política': '🔴'
                    }
                    st.write(f"**Tag:** {tag_colors.get(tag, '⚪')} {tag}")

def display_message_assignments(assignments: List[Dict[str, Any]], conversations: List[Dict[str, Any]]):
    """Display sample message assignments"""
    st.subheader("💬 Message Analysis Sample")
    
    if not assignments:
        st.info("No message assignments available.")
        return
    
    # Show first 10 assignments as sample
    sample_assignments = assignments[:10]
    
    assignments_df = pd.DataFrame([
        {
            "Conversation ID": assign.get('conversation_id', 'Unknown'),
            "Topic ID": assign.get('topic_id', 'Unknown'),
            "Sentiment": assign.get('sentiment_label', 'Unknown'),
            "Sentiment Score": f"{assign.get('sentiment_score', 0):.2f}",
            "Emotion (GEW)": assign.get('familia_gew', 'Unknown'),
            "Emotion Intensity": f"{assign.get('intensidad', 0)}/5",
            "Valence": f"{assign.get('valencia', 0):.2f}"
        }
        for assign in sample_assignments
    ])
    
    st.dataframe(assignments_df, width='stretch')
    
    if len(assignments) > 10:
        st.caption(f"Showing 10 of {len(assignments)} total message assignments. Full data available in export.")

def export_results(results: Dict[str, Any], run_id: str):
    """Create export files"""
    st.subheader("📥 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # JSON Export
        json_str = json.dumps(results, indent=2, ensure_ascii=False)
        st.download_button(
            label="📄 Download Complete Results (JSON)",
            data=json_str,
            file_name=f"voc_analysis_{run_id}.json",
            mime="application/json"
        )
    
    with col2:
        # CSV Export for topics
        if results.get('topics'):
            topics_df = pd.DataFrame([
                {
                    "Topic ID": topic['topic_id'],
                    "Label": topic['label'],
                    "Description": topic['description'],
                    "Keywords": ", ".join(topic.get('keywords', [])),
                    "Summary": topic.get('summary', ''),
                }
                for topic in results['topics']
            ])
            
            csv_data = topics_df.to_csv(index=False)
            st.download_button(
                label="📊 Download Topics (CSV)",
                data=csv_data,
                file_name=f"voc_topics_{run_id}.csv",
                mime="text/csv"
            )

# Main UI
st.title("📊 VoC Analyst - LLM-Powered Analysis")
st.write("Transform customer interaction transcripts into actionable insights using advanced LLM analysis")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model Selection
    st.subheader("🤖 Select LLM Model")
    
    provider = st.selectbox(
        "Provider",
        ["OpenAI", "Anthropic", "Gemini"],
        key="provider_select"
    )
    
    # Model options based on provider
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
        "Model",
        model_options[provider],
        key="model_select"
    )
    
    # API Key input (with environment variable fallback)
    api_key_labels = {
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY", 
        "Gemini": "GEMINI_API_KEY"
    }
    
    api_key_env = api_key_labels[provider]
    api_key = st.text_input(
        f"{provider} API Key",
        type="password",
        value=os.getenv(api_key_env, ""),
        help=f"Enter your {provider} API key or set {api_key_env} environment variable"
    )
    
    if not api_key:
        st.warning(f"Please provide {provider} API key to proceed")

# Main content area
tab1, tab2, tab3 = st.tabs(["📁 Upload & Process", "📊 Dashboard", "📥 Export"])

with tab1:
    st.header("Upload Customer Interaction Files")
    st.write("Upload TXT or PDF files containing customer-agent conversations. Each file should contain exactly one interaction.")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['txt', 'pdf'],
        accept_multiple_files=True,
        help="Upload .txt or .pdf files (max 100MB each). Each file should contain one customer interaction."
    )
    
    if uploaded_files:
        st.write(f"📁 {len(uploaded_files)} file(s) uploaded")
        
        # Process files
        if st.button("🔍 Process Files", disabled=not api_key):
            with st.spinner("Processing uploaded files..."):
                processed_files = process_uploaded_files(uploaded_files)
                st.session_state.uploaded_files_data = processed_files
                
            if processed_files:
                st.success(f"✅ Successfully processed {len(processed_files)} files")
                
                # Show file summary
                files_df = pd.DataFrame([
                    {
                        "Filename": f['filename'],
                        "Type": f['type'],
                        "Size (chars)": f['size']
                    }
                    for f in processed_files
                ])
                st.dataframe(files_df, width='stretch')

    # Show analysis button if files are processed and stored in session state
    if st.session_state.uploaded_files_data:
        st.write("📋 Files ready for analysis:")
        
        # Show summary of processed files
        files_summary_df = pd.DataFrame([
            {
                "Filename": f['filename'],
                "Type": f['type'],
                "Size (chars)": f['size']
            }
            for f in st.session_state.uploaded_files_data
        ])
        st.dataframe(files_summary_df, width='stretch')
        
        # Analysis button - now always available when files are processed
        if st.button("🚀 Analyze with LLM", disabled=not api_key):
            processed_files = st.session_state.uploaded_files_data
            if not api_key:
                st.error("API key is required for analysis")
            else:
                # Create model config
                model_config = ModelConfig(
                    provider=provider.lower(),
                    model=model,
                    api_key=api_key
                )
                
                # Initialize LLM backend
                backend = LLMBackend(model_config)
                
                # Generate run ID
                run_id = str(uuid.uuid4())[:8]
                st.session_state.run_id = run_id
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Optimized: Single LLM call for batch processing
                    status_text.text("🚀 Processing all files with LLM in batch (faster)...")
                    progress_bar.progress(50)
                    
                    st.write(f"🔬 Analyzing {len(processed_files)} files in batch...")
                    
                    # Use batch processing for better performance
                    analysis_results = backend.analyze_conversations_batch(processed_files)
                    
                    if analysis_results and isinstance(analysis_results, dict):
                        progress_bar.progress(100)
                        status_text.text("✅ Analysis complete!")
                        
                        # Store results
                        st.session_state.analysis_results = analysis_results
                        st.session_state.processing_complete = True
                        
                        # Show summary
                        topics_count = len(analysis_results.get('topics', []))
                        recs_count = len(analysis_results.get('recommendations', []))
                        
                        st.success(f"🎉 Analysis completed successfully! Run ID: {run_id}")
                        st.write(f"📊 Found {topics_count} topics and {recs_count} recommendations")
                        st.info("👉 Check the Dashboard tab to view results")
                    else:
                        st.error("❌ Analysis failed - No valid results returned")
                            
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    import traceback
                    st.text("Debug info:")
                    st.code(traceback.format_exc())
                finally:
                    progress_bar.empty()
                    status_text.empty()

with tab2:
    st.header("Analysis Dashboard")
    
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        # Display KPIs
        if 'kpis' in results:
            display_kpis(results['kpis'])
        
        st.divider()
        
        # Display Topics
        if 'topics' in results:
            display_topics(results['topics'])
        
        st.divider()
        
        # Display Recommendations
        if 'recommendations' in results:
            display_recommendations(results['recommendations'])
        
        st.divider()
        
        # Display Message Assignments Sample
        if 'message_assignments' in results:
            display_message_assignments(
                results['message_assignments'],
                results.get('conversations', [])
            )
    else:
        st.info("📈 No analysis results available. Please upload and process files first.")

with tab3:
    st.header("Export Results")
    
    if st.session_state.analysis_results and st.session_state.run_id:
        export_results(st.session_state.analysis_results, st.session_state.run_id)
    else:
        st.info("📦 No results available for export. Please complete an analysis first.")

# Footer
st.divider()
st.caption("VoC Analyst - Powered by LLM technology for comprehensive customer voice analysis")

