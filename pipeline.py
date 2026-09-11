from typing import Dict, Any
from src.data.preprocessing import construct_ticket_text
from src.classifier.predict import predict_ticket
from src.retrieval.hybrid_retriever import hybrid_search
from src.rag.prompt import construct_rag_prompt
from src.rag.generator import generate_response_from_llm

def run_pipeline(subject: str, body: str) -> Dict[str, Any]:
    """
    Executes the end-to-end Support Ticket Classification and RAG Generation pipeline.
    """
    # 1. Preprocessing
    ticket_text = construct_ticket_text(subject, body)
    
    # 2. Classification
    predictions = predict_ticket(ticket_text)
    
    # 3. Hybrid Retrieval (BM25 + FAISS -> RRF -> Reranker)
    contexts = hybrid_search(ticket_text)
    
    # 4. Prompt Construction
    prompt = construct_rag_prompt(ticket_text, predictions, contexts)
    
    # 5. LLM Response Generation
    generated_response = generate_response_from_llm(prompt)
    
    return {
        "input_ticket": ticket_text,
        "predictions": predictions,
        "retrieved_contexts": contexts,
        "prompt": prompt,
        "generated_response": generated_response
    }