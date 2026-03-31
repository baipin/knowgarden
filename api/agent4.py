# api/agent4.py

def run_presentation_design(client, content: str, model: str) -> str:
    """
    Agent 4: Presentation Agent
    职责：将已有的知识总结、关联和增长计划转化为一份专业的 PPT 大纲（Marp Markdown 格式）。
    """
    
    system_prompt = """
You are a Professional Presentation Designer. 
Your task is to convert complex knowledge into a structured slide deck outline using Marp Markdown format.

RULES:
1. Use '---' to separate slides.
2. Each slide must have a clear title (using #) and concise bullet points.
3. Add a Title Slide at the beginning.
4. Include a 'Summary' section, a 'Deep Connections' section, and a 'Future Roadmap' section.
5. Keep text concise (max 6 bullets per slide).
6. Use visual metaphors or emojis where appropriate.
7. Output ONLY the Marp Markdown content.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ],
        temperature=0.7
    )
    
    return response.choices[0].message.content
