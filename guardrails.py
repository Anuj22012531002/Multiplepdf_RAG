import re
import json
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings

SENSITIVE_PATTERNS = {
    "Credit Card Number": r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
    "Social Security Number (SSN)": r'\b\d{3}-\d{2}-\d{4}\b',
    "Bank Account / IBAN": r'\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b|\b\d{10,18}\b',
    "API / Private Keys": r'\b(?:sk-[a-zA-Z0-9]{32,}|AIzaSy[a-zA-Z0-9_-]{33})\b',
    "GST": r'\b[0-3][0-9][A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b',
    "PAN": r'\b[A-Z]{3}[ABCFGHLJPT][A-Z][0-9]{4}[A-Z]\b' 
}

PROMPT_INJECTION_KEYWORDS = [
    "ignore all previous instructions",
    "disregard prior prompts",
    "you are now in developer mode",
    "jailbreak",
    "system prompt leak",
    "override system instructions"
]


def check_input_guardrails(user_query: str) -> tuple[bool, str]:
    query_lower = user_query.lower()
    for keyword in PROMPT_INJECTION_KEYWORDS:
        if keyword in query_lower:
            return False, f"Input Blocked: Prompt Injection signature detected ('{keyword}')."
    return True, ""


def mask_sensitive_pii(text: str) -> tuple[str, list[str]]:
    detected_types = []
    masked_text = text

    for label, pattern in SENSITIVE_PATTERNS.items():
        matches = re.findall(pattern, masked_text)
        if matches:
            detected_types.append(label)
            masked_text = re.sub(pattern, f"[REDACTED {label.upper()}]", masked_text)

    return masked_text, detected_types


def check_output_guardrails(query: str, context: str, response: str, api_key: str) -> tuple[bool, str, str]:
    cleaned_response, pii_detected = mask_sensitive_pii(response)
    if pii_detected:
        pii_str = ", ".join(pii_detected)
        return False, f"Output Sanitized: Sensitive data detected ({pii_str}). Redacted for security.", cleaned_response

    # eval_llm = ChatOpenAI(model='gpt-4o-mini',api_key=api_key,temperature=0.2)
    eval_llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=api_key,
        temperature=0.0
    )

    guard_prompt = f"""You are a safety and factual integrity guardrail for an AI system.
Analyze the provided User Query, Retrieved Context, and Generated Response.

User Query:
{query}

Retrieved Context:
{context}

Generated Response:
{response}

Evaluate strictly for:
1. Hallucination: Does the response invent facts NOT present in the retrieved context?
2. Output Toxicity: Does the response contain toxic, abusive, hateful, or harmful advice?

Respond ONLY with a valid JSON object matching this schema:
{{
  "is_safe": true/false,
  "violation_type": "None" / "Hallucination" / "Toxicity",
  "reasoning": "1-sentence explanation of the violation or why it is safe."
}}
"""

    try:
        raw_eval = eval_llm.invoke(guard_prompt).content[0]['text']
        clean_json = re.sub(r'```json\s*|\s*```', '', raw_eval).strip()
        result = json.loads(clean_json)

        if not result.get("is_safe", True):
            violation = result.get("violation_type", "Safety Policy Violation")
            reason = result.get("reasoning", "Generated output violated safety guidelines.")
            return False, f"Output Flagged ({violation}): {reason}", cleaned_response

    except Exception:
        pass

    return True, "", cleaned_response