"""
Phase 4: Agentic RAG Brain (Step 2 - Full Logic)
Architecture: LangGraph State Machine
Flow: Retrieve -> Grade -> Rewrite -> Generate
"""
import os
from typing import List, TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import our LLM Factory
try:
    from model_config import get_llm
except ImportError:
    # Fallback if running standalone
    print("[WARNING] Could not import model_config. Using default setup.")
    def get_llm(): return None

# Imports for Real RAG
try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. Agent Memory ---
class AgentState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    revision_count: int
    grade: str # 'relevant' or 'irrelevant'

# --- 2. Helper: Get The Brain ---
def get_agent_llm():
    # Force cloud model for the brain (smarter), or local if preferred
    return get_llm(use_local=False, cloud_model_name="gemini-2.5-flash")

# --- 3. Node Workers ---

def retrieve(state: AgentState):
    """Node 1: The Librarian"""
    print(f"[AGENT] Retrieving for query: {state['question']}")
    try:
        # Use all-MiniLM-L6-v2 to match your upload settings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        if not os.path.exists("comparison_doc1"):
            return {"documents": ["Error: No DB found."]}

        vectorstore = FAISS.load_local("comparison_doc1", embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(state["question"])
        doc_texts = [d.page_content for d in docs]

        print(f"[SUCCESS] Retrieved {len(doc_texts)} chunks.")
        return {"documents": doc_texts}
    except Exception as e:
        print(f"[ERROR] Retrieval Failed: {e}")
        return {"documents": [f"Error: {e}"]}

def grade_documents(state: AgentState):
    """Node 2: The Inspector (LLM Based)"""
    print("[AGENT] Grading Documents with AI...")
    question = state["question"]
    docs = state["documents"]

    # Simple check first
    if not docs or "Error" in docs[0]:
        return {"grade": "irrelevant"}

    # LLM Grader
    llm = get_agent_llm()
    prompt = ChatPromptTemplate.from_template(
        """You are a strict auditor.
        User Question: {question}
        Document Content: {context}

        Does the document contain the specific information to answer the question?
        Reply ONLY with 'YES' or 'NO'.
        """
    )
    chain = prompt | llm | StrOutputParser()

    # Check the first doc (most relevant)
    context = docs[0][:2000] # Limit context size
    score = chain.invoke({"question": question, "context": context}).strip().upper()

    print(f"[GRADER] Score: {score}")

    if "YES" in score:
        return {"grade": "relevant"}
    return {"grade": "irrelevant"}

def generate(state: AgentState):
    """Node 3: The Writer (LLM Based)"""
    print("[AGENT] Generating Answer...")
    question = state["question"]
    docs = state["documents"]

    llm = get_agent_llm()
    prompt = ChatPromptTemplate.from_template(
        """You are a financial analyst.
        Context: {context}

        Answer the question: {question}
        If you cannot find the answer in the context, say "I could not find this information in the provided text."
        """
    )
    chain = prompt | llm | StrOutputParser()
    context = "\n\n".join(docs)
    answer = chain.invoke({"question": question, "context": context})

    return {"generation": answer}

def transform_query(state: AgentState):
    """Node 4: The Strategist (Rewrites query)"""
    print("[AGENT] Rewriting Query...")
    question = state["question"]
    llm = get_agent_llm()

    prompt = ChatPromptTemplate.from_template(
        """The search for '{question}' returned no results.
        Suggest a better, alternative keyword search query for a financial report.
        Reply ONLY with the new query text.
        """
    )
    chain = prompt | llm | StrOutputParser()
    new_query = chain.invoke({"question": question})
    print(f"[STRATEGIST] New Query: {new_query}")

    return {"question": new_query, "revision_count": state.get("revision_count", 0) + 1}

# --- 4. Logic & Graph ---

def decide_next_step(state: AgentState) -> Literal["transform_query", "generate"]:
    grade = state.get("grade", "irrelevant")

    if grade == "relevant":
        print("[DECISION] Docs are relevant -> Generate")
        return "generate"

    if state.get("revision_count", 0) < 1: # Limit retries
        print("[DECISION] Docs irrelevant -> Rewrite")
        return "transform_query"

    print("[DECISION] Max retries reached -> Stop")
    return "generate" # Generate "Not found" message

# Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges(
    "grade",
    decide_next_step,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("generate", END)

agent_app = workflow.compile()

if __name__ == "__main__":
    print("--- Testing Agent Brain (Full Logic) ---")
    # Test Question
    inputs = {"question": "What is the total revenue?", "revision_count": 0}
    for output in agent_app.stream(inputs):
        if "generation" in output.get("generate", {}):
            print("\n[FINAL ANSWER]:\n" + output["generate"]["generation"])
