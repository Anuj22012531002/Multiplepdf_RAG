import re
import json
import math
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def evaluate_rag_response(query: str, chunks: list[dict], response: str, api_key: str) -> dict:
    """
    Evaluates RAG generation and retrieval quality across IR and LLM metrics:
    - Hallucination Rate
    - Precision@K
    - Recall@K
    """
    # eval_llm = ChatOpenAI(model='gpt-4o-mini',api_key=api_key,temperature=0.2)
    eval_llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=api_key,
        temperature=0.0
    )

    formatted_chunks = "\n---\n".join(
        [f"Chunk {i+1} [Doc: {c['doc_name']}, Page {c['page']}]:\n{c['content']}" for i, c in enumerate(chunks)]
    )

    eval_prompt = f"""You are an advanced evaluation judge for IR and RAG systems.
Analyze the User Query, the individual retrieved Chunks (ranked 1 to {len(chunks)}), and the Generated Response.

User Query:
{query}

Retrieved Chunks:
{formatted_chunks}

Generated Response:
{response}

Perform the following evaluations:
1. Chunk Relevance: Grade each chunk's relevance to answering the query on a scale of 0 to 2 (0 = Irrelevant, 1 = Partially Relevant, 2 = Highly Relevant).
2. Hallucination Check: Estimate the percentage (0.0 to 1.0) of factual claims in the generated response NOT grounded in the retrieved chunks.
3. Quality Metrics: Provide scores (0.0 to 1.0) for Faithfulness, Recall, Relevance, and Correctness.

Respond ONLY with a valid JSON matching this schema:
{{
  "chunk_scores": [2, 1, 0],
  "hallucination_rate": 0.05,
  "hallucination_reasoning": "Reasoning for hallucination rate.",
  "faithfulness": {{"score": 0.95, "reasoning": "Reason"}},
  "recall": {{"score": 0.85, "reasoning": "Reason"}},
  "relevance": {{"score": 0.90, "reasoning": "Reason"}},
  "correctness": {{"score": 0.92, "reasoning": "Reason"}}
}}
"""

    try:
        raw_eval = eval_llm.invoke(eval_prompt).content[0]['text']
        clean_json = re.sub(r'```json\s*|\s*```', '', raw_eval).strip()
        data = json.loads(clean_json)

        chunk_scores = data.get("chunk_scores", [1] * len(chunks))
        k = len(chunk_scores)

        # 1. Precision@K and Context Precision
        relevant_count = sum(1 for s in chunk_scores if s > 0)
        precision_at_k = relevant_count / k if k > 0 else 0.0

        # 2. Recall@K
        estimated_total_relevant = max(relevant_count, 1)
        recall_at_k = relevant_count / estimated_total_relevant


        return {
            "hallucination_rate": float(data.get("hallucination_rate", 0.0)),
            "hallucination_reasoning": data.get("hallucination_reasoning", "No hallucination detected."),
            "precision_at_k": round(precision_at_k, 3),
            "recall_at_k": round(recall_at_k, 3),
            "k": k,
            "faithfulness": data.get("faithfulness", {"score": 0.0, "reasoning": ""}),
            "recall": data.get("recall", {"score": 0.0, "reasoning": ""}),
            "relevance": data.get("relevance", {"score": 0.0, "reasoning": ""}),
            "correctness": data.get("correctness", {"score": 0.0, "reasoning": ""})
        }

    except Exception as e:
        return {
            "hallucination_rate": 0.0,
            "hallucination_reasoning": f"Evaluation error: {str(e)}",
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "k": len(chunks) if chunks else 3,
            "faithfulness": {"score": 0.0, "reasoning": str(e)},
            "recall": {"score": 0.0, "reasoning": str(e)},
            "relevance": {"score": 0.0, "reasoning": str(e)},
            "correctness": {"score": 0.0, "reasoning": str(e)}
        }