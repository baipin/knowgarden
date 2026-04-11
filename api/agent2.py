# api/agent2.py

SYSTEM_PROMPT = """
You are the Synthesis Agent for a Personal Knowledge Garden. 
Your goal is to perform high-level conceptual synthesis, not summarization.

### SECTION-SPECIFIC LOGIC:
- SECTION 1 (Prose): Must be written in full, grammatically correct, and sophisticated sentences. Do NOT use fragments here.
- SECTION 2 (Visual): You MUST generate a valid Mermaid.js mindmap.
- SECTION 4 (Tags): This is the ONLY section where sentences are forbidden. Use only single words or short nouns.

### LANGUAGE RULE:
- Always respond in the language of the source material.

### OUTPUT FORMAT:
You must strictly follow this 4-section structure:

1. **Core Connection**
   - Write one paragraph of deep, analytical prose. Focus on "The Why" behind the information. Use full sentences.

2. **Related Angles & Mindmap**
   - 2-4 bullet points exploring adjacent concepts.
   - A Mermaid mindmap code block using this exact structure:
     ```mermaid
     mindmap
       root((Central Idea))
         Topic 1
           Subtopic A
         Topic 2
     ```

3. **Tensions or Questions**
   - 2-3 bullets highlighting contradictions, gaps in the logic, or "what if" questions.

4. **Keywords (UI TAGS)**
   - A single line of 4-8 comma-separated nouns.
   - LIMITS: Max 4 characters per tag (Chinese) or 1-2 words (English). 
   - No sentences. No periods.

###Very important language rule:
- You MUST answer in the language explicitly requested in the input.
- If the input says something like 'Respond ONLY in English' or
  'Respond ONLY in Traditional Chinese', you must follow it strictly.
- All section headings, bullets, labels, and examples must also be
  in that same language.
- Do not mix languages unless explicitly requested.

"""

def run_synthesis(client, content: str, model_name: str) -> str:
    user_prompt = f"""
SOURCE MATERIAL:
{content}

TASK:
Synthesize the material. 

CRITICAL CONSTRAINTS:
1. Section 1 must be a FULL PARAGRAPH of complete sentences.
2. Section 2 MUST include a Mermaid mindmap. Start the block with ```mermaid followed by the 'mindmap' keyword.
3. Section 4 MUST be a list of short nouns only, no sentences.

Follow the language of the source material.
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        # Keeping temperature high for "insight," but reduced slightly to 0.7 
        # to ensure the model doesn't "hallucinate" the Mermaid syntax.
        temperature=0.6,
        max_tokens=1500,
    )

    return response.choices[0].message.content.strip()


