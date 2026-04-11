# api/agent2.py

SYSTEM_PROMPT = """
You are a Data Architect specialized in Knowledge Graphs. 

### OBJECTIVE
Your task is to transform a summary into a high-level conceptual map. 
- DO NOT summarize. 
- DO NOT use conversational filler.
- DO NOT repeat the input verbatim.

### LOGICAL CONSTRAINTS
- Section 1 must be a SINGLE analytical paragraph. No line breaks.
- Section 2 must contain a valid Mermaid mindmap.
- Section 4 must be a raw list of nouns.

###Very important language rule:
- You MUST answer in the language explicitly requested in the input.
- If the input says something like 'Respond ONLY in English' or
  'Respond ONLY in Traditional Chinese', you must follow it strictly.
- All section headings, bullets, labels, and examples must also be
  in that same language.
- Do not mix languages unless explicitly requested.

Output style:
- analytical and structured
- insightful
- visual-friendly (via Mermaid)

"""

def run_synthesis(client, content: str, model_name: str) -> str:
    """
    Generate knowledge connections and synthesis.
    This implementation uses Chat Completions API with prompted 'internal' 
    logic verification and Mermaid visualization.
    """
    user_prompt = f"""
### INPUT DATA:
{content}

### TASK:
Analyze the input and generate a 4-section synthesis.

### STRICT STRUCTURAL REQUIREMENTS:

1. **Analytical Synthesis**
   [Instruction: Write ONE continuous paragraph. NO bullets. NO fragments.]

2. **Concept Mindmap**
   - [Idea 1]
   - [Idea 2]
   ```mermaid
   mindmap
     root((Core Discovery))
       Branch1
         NodeA
       Branch2
         NodeB

"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        # 增加 temperature 到 0.8 以增强“隐藏关联”的发现能力
        temperature=0.3,
        # 增加 max_tokens 以容纳 Mermaid 代码块
        max_tokens=1200,
    )

    return response.choices[0].message.content.strip()


