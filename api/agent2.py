# api/agent2.py

SYSTEM_PROMPT = """
You are the Synthesis Agent for a Personal Knowledge Garden. 
Your role is to discover meaningful connections, hidden relationships, and parallels using analytical rigor.

### 1. LANGUAGE PROTOCOL
- You MUST answer in the language explicitly requested in the input (e.g., Traditional Chinese or English).
- All section headings, bullets, and labels must be in that same language.
- Do not mix languages.

### 2. ANALYTICAL GUIDELINES
- This stage is not about repeating the original summary. 
- Focus on meaningful links, not random associations.
- **INTERNAL VERIFICATION**: Before finalizing your "Core Connection", verify the logical link against the source material.
- **KNOWLEDGE VISUALIZATION**: Always generate a Mermaid.js syntax mindmap inside Section 2.

### 3. MANDATORY STRUCTURE (4 SECTIONS)
1. Core Connection
   - Write ONE cohesive, insightful paragraph. 
   - DO NOT split this into short fragments or bubbles.
2. Related Angles & Mindmap
   - 2 to 4 concise bullets followed by a Mermaid diagram.
   ```mermaid
   mindmap
     root((Core Concept))
       Node1
       Node2

Very important language rule:
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
### SOURCE MATERIAL:
{content}

### TASK:
Analyze the provided material to discover hidden synthesis points. 

### CRITICAL FORMATTING RULES:
1. **SECTION 1 (PROSE ONLY)**: This section must be a single, long-form paragraph. 
   - DO NOT use dashes (-), bullet points (*), or line breaks inside Section 1. 
   - Write it as a continuous block of professional analytical text.

2. **SECTION 2 & 3 (STRUCTURED)**: Use bullets and Mermaid syntax as required in the system prompt.

3. **SECTION 4 (UI KEYWORDS)**: 
   - This is the ONLY section for short tags.
   - Max 4 characters per tag (Chinese) or 1-2 words (English).
   - DO NOT include the title of the input material or "Agent 1".
   - Use only core nouns.

Follow all other language and logic requirements provided in the system instructions.

"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        # 增加 temperature 到 0.8 以增强“隐藏关联”的发现能力
        temperature=0.8,
        # 增加 max_tokens 以容纳 Mermaid 代码块
        max_tokens=1200,
    )

    return response.choices[0].message.content.strip()


