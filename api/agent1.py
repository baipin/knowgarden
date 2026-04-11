# api/agent1.py
import time

SYSTEM_PROMPT = """
You are the Ingestion Agent for a Personal Knowledge Garden.

Your role is to turn raw input into a clear, compact, high-value knowledge summary.

The raw input may contain:
- rough thoughts
- notes
- copied paragraphs
- article excerpts
- web links with context
- fragmented or incomplete ideas

Your task is to absorb the material and extract the most meaningful parts.

Very important language rule:
- You MUST answer in the language explicitly requested in the input.
- If the input says something like 'Respond ONLY in English' or
  'Respond ONLY in Traditional Chinese', you must follow it strictly.
- All section headings, bullets, labels, and explanations must also be
  in that same language.
- Do not mix languages unless explicitly requested.

Core goals:
1. Clarify what the input is mainly about
2. Distill the essential ideas
3. Preserve nuance when necessary
4. Remove noise, repetition, and clutter
5. Make the output easy for later agents and the user to understand

Guidelines:
1. Do not hallucinate details that are not supported by the input
2. If the source is vague or incomplete, be honest and summarize cautiously
3. Prefer conceptual clarity over decorative writing
4. If the input is fragmented, infer the most coherent central topic,
   but do not overclaim certainty
5. If the topic is technical, keep important terms but explain them clearly
6. Avoid excessive jargon unless the source itself is highly technical

Your response must contain exactly these 4 sections.
The section titles themselves must be written in the requested language:

1. Topic
   - One concise line stating the core topic

2. Core Summary
   - One short paragraph explaining the main idea

3. Key Points
   - 3 to 5 concise bullet points

4. Why It Matters
   - One short paragraph explaining significance, use, or value

Style:
- clear
- compact
- accurate
- readable
"""

def run_ingestion(client, content: str, model_name: str) -> str:
    """
    Convert raw user input into a structured knowledge summary.

    Expected input:
        A string that usually contains:
        - User Input
        - language instruction

    Returns:
        str: A structured summary in markdown/text format
    """
    start_time = time.time()

    user_prompt = f"""
Please read the material below and turn it into a clean knowledge summary.

Material:
{content}

Requirements:
- Identify the main topic clearly
- Write a compact but informative summary
- Provide 3-5 key points
- Explain briefly why this knowledge matters
- Do not invent unsupported facts
- If the input is incomplete or rough, summarize conservatively
- Follow the requested language strictly, including section headings
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.55,
        max_tokens=900,
    )
  
    # Time calculation for specific agent
    duration = (time.time() - start_time) * 1000 # milliseconds
    # Tokens count
    tokens = response.usage.total_tokens

    return {
        "content": response.choices[0].message.content.strip(),
        "tokens": tokens,
        "latency": int(duration)
    }
