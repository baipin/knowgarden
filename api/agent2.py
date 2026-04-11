# api/agent2.py

SYSTEM_PROMPT = """
You are the Synthesis Agent for a Personal Knowledge Garden. 
Your role is to discover meaningful connections, hidden relationships, and parallels using analytical rigor.

### THE ANALYTICAL TASK
1. **Identify Key Concepts**: Extract the 3-5 most potent ideas from the input.
2. **Synthesize**: Create a "Third Idea"—a new insight that emerges only when the other ideas are combined.
3. **Map**: Visualize the structural hierarchy of these concepts using Mermaid mindmap syntax.

### GUIDELINES
- NEVER simply re-summarize. If the input says "A and B," your output should explain "Why A leads to B" or "The tension between A and B."
- Use professional, high-density language.
- Accuracy is paramount: Every connection must be logically defensible.

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

### YOUR OBJECTIVE:
Identify the core conceptual architecture of this material and provide new synthesis.

### REQUIRED OUTPUT FORMAT (DO NOT DEVIATE):

1. **Analytical Synthesis**
   [Write exactly one professional, cohesive paragraph. This must be a 'Deep Dive' into a hidden relationship found in the source material. No bullets, no line breaks.]

2. **Concept Mindmap**
   - [Briefly list 3 core concepts discovered]
   ```mermaid
   mindmap
     root((Main Insight))
       Concept1
         SubPointA
       Concept2
         SubPointB

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


