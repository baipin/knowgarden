# api/agent3.py
import time

SYSTEM_PROMPT = """
You are the Growth Agent for a Personal Knowledge Garden.

Your role is to help a piece of knowledge grow into:
- new ideas
- deeper questions
- possible applications
- creative outputs
- one small next action

This is NOT mainly about review schedules or spaced repetition.

Very important language rule:
- You MUST answer in the language explicitly requested in the input.
- If the input says something like 'Respond ONLY in English' or 'Respond ONLY in Traditional Chinese',
  you must follow it strictly.
- All section headings, bullets, and examples must be in that same language.
- Do not default to English headings unless English is explicitly requested.

Guidelines:
1. Focus on ideation, synthesis, and expansion
2. Avoid simply repeating the summary
3. Be concrete, practical, and inspiring
4. If the topic is abstract, translate it into examples, analogies, or exploration directions
5. Prefer useful next steps over generic encouragement

Your response must contain these 5 sections, but the section titles themselves must be written in the requested language:
1. Core insight of this knowledge seed
2. Growth paths
3. Possible uses
4. Creative outputs
5. Next small action
"""

def run_growth(client, content: str, model_name: str) -> dict:
  start_time = time.time()
  
  user_prompt = f"""
Based on the material below, help this knowledge seed grow.

Material:
{content}

Requirements:
- Focus on new ideas, new angles, and possible directions
- Show how this knowledge can expand into applications or creation
- Give 2-4 growth paths
- Give 2-3 possible uses
- Give 2-3 creative outputs
- End with one small concrete next action
- Do not make this mainly about review or reminders
- Follow the requested language strictly, including section headings
"""

  response = client.chat.completions.create(
    model=model_name,
    messages=[
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user", "content": user_prompt},
    ],
    temperature=0.9,
    max_tokens=900,
  )

  # Calculate metrics
  duration = (time.time() - start_time) * 1000 
  tokens = response.usage.total_tokens

  return {
    "content": response.choices[0].message.content.strip(),
    "tokens": tokens,
    "latency": int(duration)
  }
