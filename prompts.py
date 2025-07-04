def get_prompt_template(question: str, context: str, template_type: str = "v6") -> str:
    return f"""Answer the question truthfully based on the text below.
Include verbatim quote and a comment where to find it in the text (e.g., Page 4).

Text:
{context}

Question: {question}
"""

def build_hyde_prompt(question: str, example: str = "") -> str:
    return f"Write an example answer to the question before searching.\nQuestion: {question}\n{example}"
