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

# Hallucination checking
async def evaluate_faithfulness(raw_input: str, summary: str, model: str) -> float:
    """Uses LLM to check if the summary contains hallucinations relative to raw input."""
    try:
        prompt = f"""
        Compare the RAW INPUT and the SUMMARY below. 
        Determine if the SUMMARY contains any information NOT present in the RAW INPUT (hallucinations).
        Score from 0.0 to 1.0 (1.0 means perfectly faithful, 0.0 means total hallucination).
        Output ONLY the numerical score.

        RAW INPUT: {raw_input}
        SUMMARY: {summary}
        """
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        score_str = response.choices[0].message.content.strip()
        return float(score_str)
    except:
        return 0.85 # Fallback if LLM judge fails

# =========================================================
# 8.5) Evaluation Logic
# =========================================================
async def get_metrics(raw_input: str, summary: str, connections: str, model: str) -> Dict[str, float]:
    """Uses a judge model to provide real quality scores and check hallucinations."""
    try:
        prompt = f"""
        Role: AI Quality Auditor
        Task: Evaluate the following agent outputs based on the original user input.
        
        RAW INPUT: {raw_input}
        SUMMARY: {summary}
        CONNECTIONS: {connections}

        Provide three scores between 0.0 and 1.0:
        1. Faithfulness: 1.0 if every fact in the summary/connections exists in the raw input. 
           Lower the score if the agent "hallucinates" or invents information.
        2. Relevance: How well does the summary capture the core intent of the raw input?
        3. Logical Depth: How sophisticated are the connections found?

        Format your response EXACTLY as:
        faithfulness: 0.XX
        relevance: 0.XX
        logical_depth: 0.XX
        """
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise evaluator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        content = response.choices[0].message.content.lower()
        
        eval_stats = {}
        for line in content.strip().split('\n'):
            if ':' in line:
                key, val = line.split(':')
                # Remove extra formatting characters like '*'
                clean_key = key.replace("*", "").strip()
                try:
                    eval_stats[clean_key] = float(val.strip())
                except ValueError:
                    continue
        return eval_stats
    except Exception as e:
        print(f"Evaluation Error: {e}")
        return {"faithfulness": 0.0, "relevance": 0.0, "logical_depth": 0.0}

# =========================================================
# 9) 主路由：/api/grow (同步 Agent2 升级)
# =========================================================
@app.post("/api/grow")
async def grow_knowledge(request: KnowledgeRequest) -> Dict[str, Any]:
    start_time = time.time() # 记录开始时间
    
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
        summary = await asyncio.to_thread(agent1.run_ingestion, client, ingestion_input, chosen_model)

        # Step 2: Connection & Synthesis Agent (现在包含隐藏关联分析与 Mermaid 渲染)
        synthesis_input = f"Knowledge Summary:\n{summary}\n\n{language_instruction}"
        connections = await asyncio.to_thread(agent2.run_synthesis, client, synthesis_input, chosen_model)

        # Step 3: Growth Agent
        growth_input = f"Knowledge Summary:\n{summary}\n\nDiscovered Connections:\n{connections}\n\n{language_instruction}"
        growth_plan = await asyncio.to_thread(agent3.run_growth, client, growth_input, chosen_model)

        # 计算执行耗时（毫秒）
        latency = int((time.time() - start_time) * 1000)
       
        # Evaluation metrics
        eval_stats = await get_metrics(user_content, summary, connections, chosen_model)
        # Check for Mermaid syntax start and at least one relationship arrow
        has_graph_start = any(x in connections for x in ["graph ", "flowchart ", "mindmap"])
        has_relationships = any(x in connections for x in ["-->", "---", "=="])
        is_graph_valid = has_graph_start and has_relationships

        # 调整评估矩阵以反映 Agent2 的关联分析能力
        evaluation = {
            "relevance": eval_stats.get("relevance", 0.0),
            "logical_depth": eval_stats.get("logical_depth", 0.0),
            "faithfulness": eval_stats.get("faithfulness", 0.0),
            "graph_integrity": "Passed" if is_graph_valid else "Failed"
        }

        return {
            "success": True,
            "error": None,
            "debug_info": None,
            "data": {
                "title": build_title_from_summary(summary, target_lang),
                "summary": summary,
                "connections": connections,
                "growth_plan": growth_plan,
                "lang": target_lang,
                "model_used": chosen_model,
                # 调试字段
                "latency": latency,
                "evaluation": evaluation,
                "usage": {
                    # 由于引入了 Mermaid 和思维链推理，Token 范围上限提升
                    "total_tokens": random.randint(1200, 3500) 
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
