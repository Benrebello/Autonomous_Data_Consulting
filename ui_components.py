# ui_components.py
"""UI components for Streamlit interface, including rate limit timer."""

import streamlit as st
import time
import glob
import os
from datetime import datetime, timedelta
from typing import Optional
from io import BytesIO


def display_rate_limit_timer(wait_info: dict, placeholder: Optional[st.delta_generator.DeltaGenerator] = None):
    """Display a visual countdown timer when rate limit is hit.
    
    Args:
        wait_info: Dictionary with 'wait_seconds', 'reason', 'retry_at', and optional 'error'
        placeholder: Optional Streamlit placeholder to update
    """
    wait_seconds = wait_info['wait_seconds']
    reason = wait_info.get('reason', 'Rate limit')
    retry_at = wait_info.get('retry_at')
    error_msg = wait_info.get('error', '')
    
    # Create placeholder if not provided
    if placeholder is None:
        placeholder = st.empty()
    
    # Display initial message
    with placeholder.container():
        st.warning(f"**{reason}**")
        
        # Show error details if available
        if error_msg:
            with st.expander("Detalhes do erro"):
                st.code(error_msg, language=None)
        
        # Create progress bar and timer display
        progress_bar = st.progress(0.0)
        timer_text = st.empty()
        info_text = st.empty()
        
        if retry_at:
            info_text.info(f"Próxima tentativa em: {retry_at.strftime('%H:%M:%S')}")
        
        # Countdown loop
        start_time = time.time()
        total_wait = wait_seconds
        
        while True:
            elapsed = time.time() - start_time
            remaining = max(0, total_wait - elapsed)
            
            if remaining <= 0:
                break
            
            # Update progress bar (inverted - starts at 100% and goes to 0%)
            progress = 1.0 - (remaining / total_wait)
            progress_bar.progress(progress)
            
            # Format remaining time
            if remaining >= 60:
                time_str = f"{int(remaining // 60)}m {int(remaining % 60)}s"
            else:
                time_str = f"{int(remaining)}s"
        
            timer_text.markdown(f"### Aguardando: {time_str}")
        
            # Sleep for a short interval
            time.sleep(0.5)
        
        # Complete
        progress_bar.progress(1.0)
        timer_text.markdown("### Pronto! Retomando...")
        time.sleep(0.5)
    
    # Clear the placeholder
    placeholder.empty()


def display_rate_limit_info(rpm_limit: int, max_tokens: int = 8000):
    """Display rate limit information in sidebar.
    
    Args:
        rpm_limit: Requests per minute limit
        max_tokens: Maximum tokens per request
    """
    with st.sidebar.expander("Limites de API", expanded=False):
        st.markdown(f"""
        **Configuração atual:**
        - Requisições/minuto: `{rpm_limit}`
        - Tokens/requisição: `~{max_tokens}`
        
        **O que acontece se atingir o limite:**
        - Sistema aguarda automaticamente
        - Retry automático (até 3 tentativas)
        - Backoff exponencial em caso de erros
        
        **Dicas:**
        - Perguntas mais simples = menos tokens
        - Datasets menores = processamento mais rápido
        - O sistema gerencia tudo automaticamente
        """)


def display_token_estimate(text: str, max_tokens: int = 8000):
    """Display estimated token usage for a text.
    
    Args:
        text: Text to estimate
        max_tokens: Maximum allowed tokens
    """
    # Rough estimate: ~4 chars per token
    estimated_tokens = len(text) // 4
    percentage = (estimated_tokens / max_tokens) * 100
    
    if percentage > 90:
        st.error(f"Estimativa de tokens: {estimated_tokens} (~{percentage:.0f}% do limite)")
        st.warning("Considere reduzir o tamanho do dataset ou simplificar a pergunta.")
    elif percentage > 70:
        st.warning(f"Estimativa de tokens: {estimated_tokens} (~{percentage:.0f}% do limite)")
    else:
        st.info(f"Estimativa de tokens: {estimated_tokens} (~{percentage:.0f}% do limite)")


class RateLimitHandler:
    """Handler for rate limit events in Streamlit UI."""
    
    def __init__(self):
        self.placeholder = None
        self.active = False
    
    def on_wait(self, wait_info: dict):
        """Callback for rate limiter wait events.
        
        Args:
            wait_info: Wait information from rate limiter
        """
        # Only show timer in Streamlit context
        try:
            if not self.placeholder:
                self.placeholder = st.empty()
            
            self.active = True
            display_rate_limit_timer(wait_info, self.placeholder)
            self.active = False
            
        except Exception as e:
            # Fallback to simple sleep if not in Streamlit context
            print(f"Rate limit: waiting {wait_info['wait_seconds']:.1f}s - {wait_info['reason']}")
            time.sleep(wait_info['wait_seconds'])
    
    def reset(self):
        """Reset the handler state."""
        if self.placeholder:
            self.placeholder.empty()
        self.placeholder = None
        self.active = False


def stream_response_to_chat(stream) -> str:
    """Stream chunks to a single chat message, keeping text on screen.

    Aggregates chunk content and updates one placeholder to avoid
    per-chunk newlines and flicker. Returns the full concatenated text.
    """
    placeholder = st.empty()
    full_text = ""
    for chunk in stream:
        chunk_text = chunk.content if hasattr(chunk, 'content') else str(chunk)
        full_text += chunk_text
        # Render progressively in markdown to preserve formatting
        placeholder.markdown(full_text)
    return full_text


def cleanup_old_plot_files(pattern: str = "plot_*.png") -> int:
    """Remove legacy plot files matching the given pattern in the CWD.

    Returns the number of files removed.
    """
    removed = 0
    for path in glob.glob(pattern):
        try:
            os.remove(path)
            removed += 1
        except Exception:
            pass
    return removed


def generate_pdf_report(title: str, user_query: str, synthesis: str, full_response: str, plan: dict, charts: list[bytes]) -> bytes:
    """Generate a PDF report following ABNT-like formatting and Minto Pyramid structure.

    - Title page
    - Executive Summary (Minto: Situation, Complication, Question, Answer)
    - Development (Methods, Results with figures)
    - Conclusion and Recommendations
    - References (placeholder)
    """
    # Lazy import reportlab to avoid hard dependency during module import (useful for tests without reportlab)
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import enums
    except Exception as e:
        # If reportlab is missing, return a minimal PDF-like bytes as fallback
        # so the app does not crash; users can still run without PDF feature.
        return BytesIO(b"ReportLab not available. Install reportlab to generate PDFs.").getvalue()

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=3*cm, rightMargin=2*cm,
                            topMargin=3*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    # ABNT-like: Times 12, 1.5 spacing (approx using spaceBefore/After)
    normal = ParagraphStyle('ABNT_Normal', parent=styles['Normal'], fontName='Times-Roman', fontSize=12, leading=18)
    h1 = ParagraphStyle('ABNT_H1', parent=styles['Heading1'], fontName='Times-Bold', fontSize=14, spaceAfter=12, alignment=enums.TA_CENTER)
    h2 = ParagraphStyle('ABNT_H2', parent=styles['Heading2'], fontName='Times-Bold', fontSize=12, spaceAfter=8)

    elements = []
    # Title page
    elements.append(Paragraph(title or 'Relatório de Análise de Dados', h1))
    elements.append(Spacer(1, 18))
    elements.append(Paragraph(f"Pergunta do usuário: {user_query}", normal))
    elements.append(Spacer(1, 18))
    elements.append(Paragraph("Autores: Equipe Multiagente (Orchestrator, Team Leader, Data Architect, Data Analyst, Data Scientist)", normal))
    elements.append(PageBreak())

    # Executive Summary - Minto Pyramid
    elements.append(Paragraph("Resumo Executivo (Pirâmide de Minto)", h2))
    elements.append(Paragraph("Situação: Contexto do conjunto de dados e objetivo declarado pelo usuário.", normal))
    elements.append(Paragraph("Complicação: Limitações, qualidade dos dados, volume e restrições levantadas.", normal))
    elements.append(Paragraph("Questão-chave: Qual insight ou decisão a análise precisa apoiar?", normal))
    elements.append(Paragraph("Resposta: Síntese de alto nível dos resultados e implicações.", normal))
    if synthesis:
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Síntese do Time Leader:", h2))
        elements.append(Paragraph(full_response[:4000].replace('\n', '<br/>'), normal))

    # Development
    elements.append(PageBreak())
    elements.append(Paragraph("Desenvolvimento", h2))
    elements.append(Paragraph("Método: Plano de execução gerado e ferramentas utilizadas.", normal))
    if plan:
        try:
            plan_brief = str({k: plan[k] for k in plan.keys() if k != 'execution_plan'})
        except Exception:
            plan_brief = 'Plano não disponível.'
        elements.append(Paragraph(f"Plano: {plan_brief}", normal))
        if 'execution_plan' in plan:
            for t in plan['execution_plan'][:10]:
                desc = t.get('description', '')
                tool = t.get('tool_to_use', '')
                elements.append(Paragraph(f"Tarefa {t.get('task_id')}: {desc} (ferramenta: {tool})", normal))

    # Results with charts
    if charts:
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Resultados (Figuras)", h2))
        for ch in charts[:6]:  # limit pages
            try:
                img = Image(BytesIO(ch))
                img._restrictSize(15*cm, 12*cm)
                elements.append(img)
                elements.append(Spacer(1, 12))
            except Exception:
                continue

    # Conclusion
    elements.append(PageBreak())
    elements.append(Paragraph("Conclusões e Recomendações", h2))
    elements.append(Paragraph(full_response.replace('\n', '<br/>')[:8000], normal))

    # References
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Referências (quando aplicável)", h2))
    elements.append(Paragraph("Este relatório segue formatação semelhante às normas ABNT (margens e tipografia) e estrutura de comunicação da Pirâmide de Minto.", normal))

    doc.build(elements)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes
