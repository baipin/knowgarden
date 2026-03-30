import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# 确保导入你的 Agent 模块
try:
    from .agent1 import run_ingestion
    from .agent2 import run_synthesis
    from .agent3 import run_growth
except ImportError:
    import agent1 as agent1_mod
    import agent2 as agent2_mod
    import agent3 as agent3_mod
    run_ingestion = agent1_mod.run_ingestion
    run_synthesis = agent2_mod.run_synthesis
    run_growth = agent3_mod.run_growth

load_dotenv()

app = FastAPI()

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

class KnowledgeRequest(BaseModel):
    content: str
    lang: str = "zh-cn"  # 默认

@app.get("/api/health")
def health_check():
    return {"status": "Garden Engine Online"}

@app.post("/api/grow")
async def grow_knowledge(request: KnowledgeRequest):
    user_content = request.content.strip()
    target_lang = request.lang  # 获取前端传来的语言偏好
    
    if not user_content:
        raise HTTPException(status_code=400, detail="Empty content")

    # 2. 定义语言映射表，确保 Prompt 准确
    lang_map = {
        "zh-cn": "Simplified Chinese (简体中文)",
        "zh-tw": "Traditional Chinese (繁體中文)",
        "en": "English"
    }
    selected_lang = lang_map.get(target_lang, "Simplified Chinese")

    # 3. 构建统一的语言指令后缀
    lang_instruction = f"\n\nIMPORTANT: You MUST provide your response ONLY in {selected_lang}."

    try:
        # --- STAGE 1: Ingestion ---
        # 传入语言指令给各个 Agent
        summary = await asyncio.to_thread(
            run_ingestion, client, user_content + lang_instruction
        )

        # --- STAGE 2: Synthesis ---
        connections = await asyncio.to_thread(
            run_synthesis, client, summary + lang_instruction
        )

        # --- STAGE 3: Growth ---
        growth_plan = await asyncio.to_thread(
            run_growth, client, summary + connections + lang_instruction
        )

        # 使用了 database.py，在此处保存
        from .database import save_entry
        save_entry(summary, connections, growth_plan, target_lang)

        return {
            "success": True,
            "data": {
                "title": summary[:30],
                "summary": summary,
                "connections": connections,
                "growth_plan": growth_plan,
                "lang": target_lang
            }
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
