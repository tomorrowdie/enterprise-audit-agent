"""
RAG-Based Semantic Document Comparison Module
Cross-Lingual Audit for Printing Companies

Phase 1: Cloud APIs (Claude/Gemini) for semantic comparison
Uses RAG to align sections and detect logical/semantic errors
(Not just textual diffs)
"""

import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

from model_config import get_llm

# Try to import EnsembleRetriever from different locations
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain_community.retrievers import EnsembleRetriever
    except ImportError:
        EnsembleRetriever = None

import os
import json
import re
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Try to import from new location first, fall back to old location
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


@dataclass
class RiskItem:
    """Structured risk audit item"""
    risk_level: str  # CRITICAL, WARNING, INFO
    category: str  # Financial, Entity, Translation, Logic
    source_section: str  # Where error was found in source
    source_text: str  # Original text from source
    target_text: str  # Mismatched text from target
    error_description: str  # What's wrong
    correction_needed: str  # What should it be
    confidence: str  # HIGH, MEDIUM, LOW

    def to_dict(self):
        return asdict(self)


def extract_pdf_text_with_sections(pdf_file) -> tuple:
    """Extract text from PDF with section markers and return as string + list"""
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        sections = []

        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text.strip():
                section_marker = f"[Page {page_num + 1}]"
                text += f"\n\n{section_marker}\n{page_text}"
                sections.append({
                    "page": page_num + 1,
                    "content": page_text
                })

        return text, sections
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return "", []


def get_hybrid_retriever(vectorstore):
    """
    Creates a Hybrid Search Retriever (70% BM25 + 30% FAISS).
    Requires: pip install rank_bm25
    """
    try:
        bm25_retriever = BM25Retriever.from_documents(vectorstore.docstore._dict.values())
        bm25_retriever.k = 3
    except Exception as e:
        print(f"Warning: BM25 failed (text too short?). Using Vector only. {e}")
        return vectorstore.as_retriever(search_kwargs={"k": 4})

    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.7, 0.3]
    )
    return ensemble


def extract_key_entities(text: str, api_provider: str, api_key: str, model: str) -> Dict:
    """
    Extract critical entities from document using the Central Brain factory.
    Returns dict with categories: financial, entities, rules, dates, etc.
    """
    try:
        use_local = (api_provider == "Local/Ollama")
        # === FIX: Pass 'model' to 'cloud_model_name' ===
        llm = get_llm(use_local=use_local, local_model="llama3", api_key=api_key, cloud_model_name=model)

        # Limit to first 100k characters
        text_for_analysis = text[:100000]

        prompt = f"""Analyze this document and extract CRITICAL information in JSON format.
Focus on:
1. Stock codes / Company identifiers
2. Financial amounts with currency (e.g., "RMB 245 billion")
3. Dates and date ranges
4. Rule numbers or regulations (e.g., "13.10")
5. Named entities (people, companies)
6. Count/quantities (e.g., "1,602 cases")
7. Categories or classifications

Return ONLY valid JSON with this structure:
{{
    "stock_codes": ["code1", "code2"],
    "financial_items": [{{"amount": "value", "currency": "type", "context": "description"}}],
    "dates": ["date1", "date2"],
    "rules": ["13.10", "13.09"],
    "named_entities": ["entity1", "entity2"],
    "quantities": [{{"value": "number", "unit": "type", "context": "description"}}],
    "key_sections": ["section1 description", "section2 description"]
}}

DOCUMENT TEXT:
{text_for_analysis}

JSON OUTPUT:"""

        response = llm.invoke(prompt)

        # Parse JSON response
        try:
            response_text = response.content
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                entities = json.loads(json_str)
                return entities
        except Exception as e:
            print(f"Could not parse entities: {str(e)}")
            return {}

    except Exception as e:
        print(f"Error extracting entities: {str(e)}")
        return {}


def semantic_section_comparison(
    source_section: str,
    target_section: str,
    source_entities: Dict,
    target_entities: Dict,
    api_provider: str,
    api_key: str,
    model: str,
    audit_mode: str = "Error Detection"
) -> List[RiskItem]:
    """Compare sections based on selected mode."""
    try:
        use_local = (api_provider == "Local/Ollama")
        # === FIX: Pass 'model' to 'cloud_model_name' ===
        llm = get_llm(use_local=use_local, local_model="llama3", api_key=api_key, cloud_model_name=model)

        # === PROMPT SWITCHING LOGIC ===
        if audit_mode == "Variance Analysis (YoY)":
            # SMART VARIANCE PROMPT (Filters Noise)
            prompt = f"""You are a Senior Financial Analyst. Compare the Reference (Old Year) vs Target (New Year).

**GOAL:** Identify MATERIAL CHANGES only. Ignore "Wording Polish".

**SOURCE (Reference):** {source_section[:4000]}
**TARGET (New):** {target_section[:4000]}

**FILTERING RULES:**
1. 🟢 **REPORT (High Priority):**
   - Any number change (Revenue, Profit, Dates, Quantities).
   - Any Entity change (Director names, Stock codes, Subsidiaries).
   - Meaning Reversal (e.g., "Outlook is positive" -> "Outlook is negative").

2. 🔴 **IGNORE (Noise):**
   - Rewording that means the same thing (e.g., "We grew 5%" vs "Growth was 5%").
   - Formatting changes, page numbers, headers/footers.
   - General marketing fluff rewrites.

**CRITICAL REQUIREMENTS:**
- Return ONLY a valid JSON array. Do NOT include any text before or after the JSON.
- Each object must have: "risk_level", "category", "source_text", "target_text", "error_description", "correction_needed", "confidence"
- If no variances found, return an empty array: []
- Ensure all JSON strings are properly escaped.

**EXAMPLE OUTPUT (if variances found):**
[
    {{
        "risk_level": "WARNING",
        "category": "Financial",
        "source_text": "Profit: $100m",
        "target_text": "Profit: $120m",
        "error_description": "Net profit increased by 20%",
        "correction_needed": "Verify against financial statements",
        "confidence": "HIGH"
    }}
]

RESPOND WITH ONLY THE JSON ARRAY, nothing else."""
        else:
            # LOGICAL PROMPT for Translation/Proofreading (Default)
            prompt = f"""You are a Proofreader checking a Draft against a Source.
Find LOGICAL ERRORS and TRANSLATION MISTAKES.

**SOURCE (Truth):** {source_section[:4000]}
**TARGET (Draft):** {target_section[:4000]}

**INSTRUCTIONS:**
- Ignore valid updates (like 2023->2024).
- Only report mistranslations, logic errors, or missing critical info.

**CRITICAL REQUIREMENTS:**
- Return ONLY a valid JSON array. Do NOT include any text before or after the JSON.
- Each object must have: "risk_level", "category", "source_text", "target_text", "error_description", "correction_needed", "confidence"
- If no errors found, return an empty array: []
- Ensure all JSON strings are properly escaped.

**EXAMPLE OUTPUT (if errors found):**
[
    {{
        "risk_level": "WARNING",
        "category": "Translation",
        "source_text": "Original text here",
        "target_text": "Incorrect translation here",
        "error_description": "Description of the error",
        "correction_needed": "What should be changed",
        "confidence": "HIGH"
    }}
]

RESPOND WITH ONLY THE JSON ARRAY, nothing else."""

        response = llm.invoke(prompt)

        # Parse JSON response
        try:
            content = response.content
            start = content.find('[')
            end = content.rfind(']') + 1
            if start != -1 and end > start:
                data = json.loads(content[start:end])
                return [RiskItem(
                    risk_level=item.get('risk_level', 'WARNING'),
                    category=item.get('category', 'Other'),
                    source_section="Document A",
                    source_text=item.get('source_text', ''),
                    target_text=item.get('target_text', ''),
                    error_description=item.get('error_description', ''),
                    correction_needed=item.get('correction_needed', ''),
                    confidence=item.get('confidence', 'MEDIUM')
                ) for item in data]
            # Return error if JSON brackets not found
            return [RiskItem(
                risk_level='CRITICAL',
                category='System Error',
                source_section='Comparison Engine',
                source_text='',
                target_text='',
                error_description='LLM response did not contain valid JSON array (no [ ] brackets found)',
                correction_needed='Check LLM output. Ensure JSON format is valid.',
                confidence='HIGH'
            )]
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {str(e)}")
            return [RiskItem(
                risk_level='CRITICAL',
                category='System Error',
                source_section='Comparison Engine',
                source_text='',
                target_text='',
                error_description=f'JSON parse error: {str(e)}',
                correction_needed='Verify LLM returned valid JSON array format.',
                confidence='HIGH'
            )]
        except Exception as e:
            print(f"Error parsing comparison results: {str(e)}")
            return [RiskItem(
                risk_level='CRITICAL',
                category='System Error',
                source_section='Comparison Engine',
                source_text='',
                target_text='',
                error_description=f'Unexpected error processing LLM response: {str(e)}',
                correction_needed='Check API connectivity and LLM response format.',
                confidence='HIGH'
            )]

    except Exception as e:
        print(f"Error comparing sections: {str(e)}")
        return [RiskItem(
            risk_level='CRITICAL',
            category='System Error',
            source_section='Comparison Engine',
            source_text='',
            target_text='',
            error_description=f'LLM call failed: {str(e)}',
            correction_needed='Verify API key is valid and LLM service is accessible.',
            confidence='HIGH'
        )]


def run_semantic_audit(
    source_pdf_path: str,
    target_pdf_path: str,
    api_provider: str,
    api_key: str,
    model: str,
    progress_callback=None,
    audit_mode: str = "Error Detection"
) -> Dict:
    """
    Run full semantic audit comparing source vs target documents
    Returns audit results with risk items

    Supports both PDF files and plain text files from Contextual RAG processing
    """
    try:
        # Step 1: Extract texts
        if progress_callback:
            progress_callback(10, "📄 Extracting source document...")

        # Check if it's a text file (from Contextual RAG) or PDF
        if source_pdf_path.endswith('.txt'):
            # Read plain text file
            with open(source_pdf_path, 'r', encoding='utf-8') as f:
                source_text = f.read()

            # === FIX: RECONSTRUCT PAGES FROM TEXT ===
            # The upload module saves text with "[Page X]" markers. We use them to split.
            # Split by [Page X] marker
            raw_pages = re.split(r'\[Page \d+\]', source_text)
            # Filter out empty splits
            source_sections = [{'content': p.strip(), 'page': i} for i, p in enumerate(raw_pages) if p.strip()]

            # Fallback: If no page markers found (e.g. short text), split by length
            if len(source_sections) <= 1:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=100)
                chunks = text_splitter.split_text(source_text)
                source_sections = [{'content': chunk, 'page': i+1} for i, chunk in enumerate(chunks)]
        else:
            # Read PDF file
            source_text, source_sections = extract_pdf_text_with_sections(
                open(source_pdf_path, 'rb')
            )

        if progress_callback:
            progress_callback(20, "📄 Extracting target document...")

        # Check if it's a text file (from Contextual RAG) or PDF
        if target_pdf_path.endswith('.txt'):
            # Read plain text file
            with open(target_pdf_path, 'r', encoding='utf-8') as f:
                target_text = f.read()

            # === FIX: RECONSTRUCT PAGES FOR TARGET ===
            # Split by [Page X] marker
            raw_pages = re.split(r'\[Page \d+\]', target_text)
            # Filter out empty splits
            target_sections = [{'content': p.strip(), 'page': i} for i, p in enumerate(raw_pages) if p.strip()]

            # Fallback: If no page markers found (e.g. short text), split by length
            if len(target_sections) <= 1:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=100)
                chunks = text_splitter.split_text(target_text)
                target_sections = [{'content': chunk, 'page': i+1} for i, chunk in enumerate(chunks)]
        else:
            # Read PDF file
            target_text, target_sections = extract_pdf_text_with_sections(
                open(target_pdf_path, 'rb')
            )

        if not source_text or not target_text:
            return {
                'success': False,
                'error': 'Could not extract text from documents'
            }

        # Step 2: Extract entities
        if progress_callback:
            progress_callback(30, "🔍 Extracting entities from source...")

        source_entities = extract_key_entities(source_text, api_provider, api_key, model)

        if progress_callback:
            progress_callback(40, "🔍 Extracting entities from target...")

        target_entities = extract_key_entities(target_text, api_provider, api_key, model)

        # Step 3: Section-by-section semantic comparison
        if progress_callback:
            progress_callback(50, f"🤖 Comparing {len(source_sections)} sections semantically...")

        all_risk_items = []

        # Compare each source section with target
        for idx, source_section_data in enumerate(source_sections):
            if progress_callback:
                progress = 50 + int((idx / len(source_sections)) * 35)
                # Show WHICH page/section is being checked
                progress_callback(progress, f"🤖 Auditing Section {idx + 1}/{len(source_sections)}...")

            source_content = source_section_data['content']

            # Find corresponding target section (roughly same page)
            target_content = ""
            if idx < len(target_sections):
                target_content = target_sections[idx]['content']
            else:
                # Fallback: use last section if target has fewer pages
                target_content = target_sections[-1]['content'] if target_sections else ""

            # Semantic comparison
            risks = semantic_section_comparison(
                source_content,
                target_content,
                source_entities,
                target_entities,
                api_provider,
                api_key,
                model,
                audit_mode=audit_mode
            )

            all_risk_items.extend(risks)

        # Step 4: Entity-level comparison
        if progress_callback:
            progress_callback(85, "📊 Cross-checking entities...")

        # Create comprehensive entity comparison
        entity_risks = _compare_entities(source_entities, target_entities)
        all_risk_items.extend(entity_risks)

        if progress_callback:
            progress_callback(100, "✅ Audit complete!")

        return {
            'success': True,
            'source_entities': source_entities,
            'target_entities': target_entities,
            'risk_items': all_risk_items,
            'summary': {
                'total_risks': len(all_risk_items),
                'critical_count': len([r for r in all_risk_items if r.risk_level == 'CRITICAL']),
                'warning_count': len([r for r in all_risk_items if r.risk_level == 'WARNING']),
                'source_doc_size': len(source_text),
                'target_doc_size': len(target_text),
                'source_sections': len(source_sections),
                'target_sections': len(target_sections)
            }
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def _compare_entities(source_entities: Dict, target_entities: Dict) -> List[RiskItem]:
    """
    Compare extracted entities between source and target
    Returns risk items for entity mismatches
    """
    risks = []

    # Check stock codes
    source_codes = set(source_entities.get('stock_codes', []))
    target_codes = set(target_entities.get('stock_codes', []))

    if source_codes != target_codes:
        for code in source_codes - target_codes:
            risks.append(RiskItem(
                risk_level='CRITICAL',
                category='Entity',
                source_section='Stock Code',
                source_text=code,
                target_text=str(target_codes) if target_codes else 'NOT FOUND',
                error_description=f'Stock code mismatch: {code} not found in target',
                correction_needed=f'Should be: {code}',
                confidence='HIGH'
            ))

    # Check dates
    source_dates = set(source_entities.get('dates', []))
    target_dates = set(target_entities.get('dates', []))

    if source_dates != target_dates:
        for date in source_dates - target_dates:
            risks.append(RiskItem(
                risk_level='CRITICAL',
                category='Entity',
                source_section='Date',
                source_text=date,
                target_text=str(target_dates - source_dates) if (target_dates - source_dates) else 'MISSING',
                error_description=f'Date mismatch: {date} not found in target',
                correction_needed=f'Should be: {date}',
                confidence='MEDIUM'
            ))

    # Check rules
    source_rules = set(source_entities.get('rules', []))
    target_rules = set(target_entities.get('rules', []))

    if source_rules != target_rules:
        for rule in source_rules - target_rules:
            risks.append(RiskItem(
                risk_level='CRITICAL',
                category='Entity',
                source_section='Regulation',
                source_text=f'Rule {rule}',
                target_text=str(target_rules) if target_rules else 'NOT FOUND',
                error_description=f'Rule number mismatch: {rule} not found in target',
                correction_needed=f'Should be: {rule}',
                confidence='HIGH'
            ))

    return risks


def display_audit_results(results: Dict):
    """Display semantic audit results in Streamlit UI"""

    if not results.get('success'):
        st.error(f"❌ Audit failed: {results.get('error')}")
        return

    # Summary metrics
    summary = results['summary']

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Risk Items",
            summary['total_risks'],
            delta=f"{summary['critical_count']} critical"
        )

    with col2:
        st.metric(
            "Critical Issues",
            summary['critical_count'],
            delta_color="inverse"
        )

    with col3:
        st.metric(
            "Warnings",
            summary['warning_count']
        )

    with col4:
        st.metric(
            "Source Pages",
            summary['source_sections']
        )

    st.markdown("---")

    # Risk items breakdown
    risk_items = results['risk_items']

    if not risk_items:
        st.success("✅ No semantic errors detected!")
        return

    # Group by risk level
    critical_risks = [r for r in risk_items if r.risk_level == 'CRITICAL']
    warning_risks = [r for r in risk_items if r.risk_level == 'WARNING']

    # Critical issues
    if critical_risks:
        st.subheader("🚨 CRITICAL Issues (Must Fix)")

        for idx, risk in enumerate(critical_risks):
            with st.expander(
                f"[{risk.category}] {risk.error_description[:60]}...",
                expanded=(idx == 0)
            ):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Source (Correct):**")
                    st.code(risk.source_text, language="text")

                with col2:
                    st.markdown("**Target (Error):**")
                    st.code(risk.target_text, language="text")

                st.markdown("**Issue:**")
                st.write(risk.error_description)

                st.markdown("**Correction Needed:**")
                st.success(risk.correction_needed)

                st.markdown(f"**Confidence:** {risk.confidence}")

    # Warnings
    if warning_risks:
        st.subheader("⚠️ Warnings (Review)")

        for idx, risk in enumerate(warning_risks):
            with st.expander(
                f"[{risk.category}] {risk.error_description[:60]}...",
                expanded=False
            ):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Source:**")
                    st.code(risk.source_text, language="text")

                with col2:
                    st.markdown("**Target:**")
                    st.code(risk.target_text, language="text")

                st.markdown("**Issue:**")
                st.write(risk.error_description)

                st.markdown(f"**Confidence:** {risk.confidence}")

    # Export option
    st.markdown("---")

    if st.button("📥 Export Audit Report as JSON"):
        audit_report = {
            'timestamp': datetime.now().isoformat(),
            'summary': summary,
            'risk_items': [r.to_dict() for r in risk_items]
        }

        st.json(audit_report)

        # Download button
        import json as json_module
        json_str = json_module.dumps(audit_report, indent=2, ensure_ascii=False)
        st.download_button(
            label="Download Report",
            data=json_str,
            file_name=f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
