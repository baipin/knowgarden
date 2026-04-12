import os
import sys
import asyncio
import traceback
import time  
import random 
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv


# =========================================================
# 1) 基础路径设置
# =========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)


# =========================================================
# 2) 加载环境变量
# =========================================================
load_dotenv()


# =========================================================
# 3) 导入各 Agent（带容错）
# =========================================================
try:
    import agent1
    import agent2
    import agent3
    import agent4
except Exception as e:
    IMPORT_ERROR = str(e) + "\n" + traceback.format_exc()
else:
    IMPORT_ERROR = None


# =========================================================
# 4) 初始化 FastAPI
# =========================================================
app = FastAPI(title="Knowledge Garden API")


# =========================================================
# 5) 允许前端跨域访问
# =========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# 6) 初始化 LLM Client
# =========================================================
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)


# =========================================================
# 7) 请求体定义
# =========================================================
class KnowledgeRequest(BaseModel):
    content: str
    lang: str = "zh-cn"
    model: str = "gpt-4o-mini"

# 用于生成 PPT 的请求体
class PPTRequest(BaseModel):
    title: str
    summary: str
    connections: str
    growth_plan: str
    lang: str = "zh-cn"
    model: str = "gpt-4o-mini"



# =========================================================
# 8) 多语言映射
# =========================================================
LANG_MAP = {
    "zh-cn": "Simplified Chinese",
    "zh-tw": "Traditional Chinese",
    "en": "English"
}


def build_language_instruction(target_lang: str) -> str:
    selected_lang = LANG_MAP.get(target_lang, "Simplified Chinese")
    return f"""
{'-'*20}
IMPORTANT LANGUAGE RULE:
1. All output must be in {selected_lang}.
2. For PPT/Marp: Slide titles, bullet points, and speaker notes must be in {selected_lang}.
3. For Combined Seeds: Ensure the synthesis and new insights are purely in {selected_lang}.
4. Do not translate technical terms if they are commonly used in English, but provide explanations in {selected_lang}.
{'-'*20}
"""


def build_error_message(target_lang: str, error_type: str = "general") -> str:
    messages = {
        "zh-cn": {
            "import": "后端模块导入失败，请检查 Agent 文件或依赖配置。",
            "general": "后端处理失败，请检查服务日志或控制台错误信息。"
        },
        "zh-tw": {
            "import": "後端模組匯入失敗，請檢查 Agent 檔案或依賴設定。",
            "general": "後端處理失敗，請檢查服務日誌或控制台錯誤訊息。",
        },
        "en": {
            "import": "Backend module import failed. Please check the Agent files or dependencies.",
            "general": "Backend processing failed. Please check the service logs or console error details."
        }
    }
    return messages.get(target_lang, messages["zh-cn"]).get(error_type, messages["zh-cn"]["general"])


def build_title_from_summary(summary: str, fallback_lang: str) -> str:
    cleaned = (summary or "").strip()
    if not cleaned:
        fallback = {
            "zh-cn": "未命名知识种子",
            "zh-tw": "未命名知識種子",
            "en": "Untitled Knowledge Seed"
        }
        return fallback.get(fallback_lang, "Untitled Knowledge Seed")

    first_line = cleaned.splitlines()[0].strip()
    return first_line[:40] if first_line else cleaned[:40]

# =========================================================
# 8.5) Evaluation Logic
# =========================================================
async def get_metrics(raw_input, summary, connections, growth, eval_model, lang):
    """Evaluates the 4 core pillars using the user's interface language."""
    lang_map = {
        "zh-cn": "Simplified Chinese",
        "zh-tw": "Traditional Chinese",
        "en": "English"
    }
    target_lang = lang_map.get(lang, "English")

    try:
        prompt = f"""
  Role: Knowledge Audit Judge
  Task: Evaluate the AI's performance based on the user's RAW INPUT. 

  RAW INPUT: {raw_input}
  SUMMARY: {summary}
  CONNECTIONS: {connections}
  GROWTH_PLAN: {growth}

  Evaluation Criteria (Score 0.0 to 1.0):
  1. Relevance (Thematic Alignment): Does the output stay true to the user's intent, or does it drift into generic AI filler?
  2. Faithfulness (Fact-Checking): Are there any hallucinations? Every claim in the Summary and Connections must be strictly logically derived from the RAW INPUT. Penalize "invented" facts.
  3. Synthesis (Insight & Depth): Look at the 'Core Connection' and 'Friction' sections. Does the AI identify genuine 'aha!' patterns, logical gaps, or non-obvious parallels? High scores require deep analytical reasoning, not just rephrasing.
  4. Actionability (Pragmatic Conversion): Is the 'next small action' a specific, high-leverage step that actually bridges the theory into practice?

  Instructions:
  - Write the 'justification' text in {target_lang} (MAX 30 WORDS).
  - Then provide numerical scores.
  - NO JSON. NO Markdown code blocks. No preamble.

  Format your response EXACTLY like this example:
  justification: [Brief reasoning]
  relevance: 0.85
  faithfulness: 0.90
  synthesis: 0.70
  actionability: 0.80
  """
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=eval_model, 
            messages=[
                {"role": "system", "content": "You are a concise auditor. Respond ONLY with the requested format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=900
        )

        tokens = response.usage.total_tokens
        raw_content = response.choices[0].message.content.strip()

        # Handling Chain-of-Thought in DeepSeek R1/V3
        clean_content = raw_content.split("</think>")[-1].strip()
            
        eval_stats = {
            "relevance": 0.0, "faithfulness": 0.0, 
            "synthesis": 0.0, "actionability": 0.0, 
            "justification": "Parsing failed"
        }

        # Line-by-Line Parsing
        for line in clean_content.split('\n'):
            line = line.replace('*', '').strip() # Remove bolding if AI adds it
            if ':' in line:
                # Split only on the FIRST colon to protect colons in the justification text
                parts = line.split(':', 1)
                key = parts[0].strip().lower() # Lowercase only the key
                val = parts[1].strip()
                
                if key == "justification":
                    eval_stats["justification"] = val
                elif key in eval_stats:
                    try:
                        import re
                        # Search for the first number (float or int) in the value
                        num_match = re.search(r"(\d+\.\d+|\d+)", val)
                        if num_match:
                            eval_stats[key] = float(num_match.group(1))
                    except:
                        pass

        return {"stats": eval_stats, "tokens": tokens}

    except Exception as e:
        print(f"Evaluation Error: {e}")
        return {
            "stats": {"relevance": 0.0, "faithfulness": 0.0, "synthesis": 0.0, "actionability": 0.0, "justification": f"Error: {str(e)[:50]}"},
            "tokens": 0
        }
# =========================================================
# 9) 主路由：/api/grow (同步 Agent2 升级)
# =========================================================
@app.post("/api/grow")
async def grow_knowledge(request: KnowledgeRequest) -> Dict[str, Any]:
    start_time = time.time() # 记录开始时间
    total_tokens = 0
    
    target_lang = request.lang or "zh-cn"
    chosen_model = request.model

    if IMPORT_ERROR:
        return {
            "success": False,
            "error": build_error_message(target_lang, "import"),
            "debug_info": IMPORT_ERROR,
            "data": None
        }

    user_content = request.content.strip()
    if not user_content:
        return {
            "success": False,
            "error": {
                "zh-cn": "输入内容不能为空。",
                "zh-tw": "輸入內容不能為空。",
                "en": "Input content cannot be empty."
            }.get(target_lang, "输入内容不能为空。"),
            "debug_info": None,
            "data": None
        }

    language_instruction = build_language_instruction(target_lang)

    try:
        # Step 1: Ingestion Agent
        ingestion_input = f"User Input:\n{user_content}\n\n{language_instruction}"
        res1 = await asyncio.to_thread(agent1.run_ingestion, client, ingestion_input, chosen_model)
        summary = res1["content"]      # <--- Get the text
        total_tokens += res1["tokens"] # <--- Add the real tokens

        # Step 2: Synthesis Agent
        synthesis_input = f"Knowledge Summary:\n{summary}\n\n{language_instruction}"
        res2 = await asyncio.to_thread(agent2.run_synthesis, client, synthesis_input, chosen_model)
        connections = res2["content"]  
        total_tokens += res2["tokens"]

        # Step 3: Growth Agent
        growth_input = f"Knowledge Summary:\n{summary}\n\nDiscovered Connections:\n{connections}\n\n{language_instruction}"
        res3 = await asyncio.to_thread(agent3.run_growth, client, growth_input, chosen_model)
        growth_plan = res3["content"]  
        total_tokens += res3["tokens"] 
        
        # Step 4: Evaluation 
        eval_model = "deepseek-v3" 
        res_eval = await get_metrics(user_content, summary, connections, growth_plan, eval_model, target_lang)
        evaluation = res_eval["stats"]
        total_tokens += res_eval["tokens"]

        # Final Time calculation
        final_latency = int((time.time() - start_time) * 1000)

        return {
            "success": True,
            "data": {
                "title": build_title_from_summary(summary, target_lang),
                "summary": summary,
                "connections": connections,
                "growth_plan": growth_plan,
                "latency": final_latency,
                "evaluation": evaluation,
                "usage": {
                    "total_tokens": total_tokens,
                    "breakdown": {
                        "ingestion": res1["tokens"],
                        "synthesis": res2["tokens"],
                        "growth": res3["tokens"],
                        "audit": res_eval["tokens"]
                    }
                }
            }
        }

    except Exception as e:
        full_error = traceback.format_exc()
        return {
            "success": False,
            "error": build_error_message(target_lang, "general"),
            "debug_info": full_error,
            "data": None
        }

# =========================================================
# 10) 路由：/api/generate_ppt
# =========================================================
@app.post("/api/generate_ppt")
async def generate_ppt(request: PPTRequest) -> Dict[str, Any]:
    target_lang = request.lang or "zh-cn"
    chosen_model = request.model
    language_instruction = build_language_instruction(target_lang)

    if IMPORT_ERROR:
        return {"success": False, "error": "Agent 4 not found"}

    ppt_context = f"Topic: {request.title}\n---\nSummary:\n{request.summary}\n---\nConnections:\n{request.connections}\n---\nRoadmap:\n{request.growth_plan}\n---\n{language_instruction}"

    try:
        ppt_content = await asyncio.to_thread(agent4.run_presentation_design, client, ppt_context, chosen_model)
        return {
            "success": True,
            "data": {
                "ppt_markdown": ppt_content,
                "model_used": chosen_model
            }
        }
    except Exception as e:
        return {"success": False, "error": "PPT Generation failed", "debug_info": traceback.format_exc()}

# =========================================================
# 11) 运行配置
# =========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
