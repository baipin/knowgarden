import os
import sys
import asyncio
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# --- 1. 路径修复 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

# --- 2. 导入 Agent (带容错) ---
try:
    import agent1
    import agent2
    import agent3
except Exception as e:
    # 如果导入阶段就失败，记录下来
    IMPORT_ERROR = str(e) + "\n" + traceback.format_exc()
else:
    IMPORT_ERROR = None

load_dotenv()

app = FastAPI()

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
    lang: str = "zh-cn"

@app.post("/api/grow")
async def grow_knowledge(request: KnowledgeRequest):
    # 准备返回的初始结构
    result = {
        "success": False,
        "data": None,
        "debug_info": None
    }
    
    if IMPORT_ERROR:
        return {"success": False, "error": "Import Error", "debug_info": IMPORT_ERROR}

    user_content = request.content.strip()
    target_lang = request.lang 
    
    lang_map = {
        "zh-cn": "Simplified Chinese",
        "zh-tw": "Traditional Chinese",
        "en": "English"
    }
    selected_lang = lang_map.get(target_lang, "Simplified Chinese")
    lang_instruction = f"\n\nIMPORTANT: Response ONLY in {selected_lang}."

    try:
        # --- 依次执行 Agent ---
        # 1. Ingestion
        summary = await asyncio.to_thread(
            agent1.run_ingestion, client, user_content + lang_instruction
        )

        # 2. Synthesis
        connections = await asyncio.to_thread(
            agent2.run_synthesis, client, summary + lang_instruction
        )

        # 3. Growth
        growth_plan = await asyncio.to_thread(
            agent3.run_growth, client, summary + connections + lang_instruction
        )

        # 执行成功
        return {
            "success": True,
            "data": {
                "title": summary[:20].split('\n')[0],
                "summary": summary,
                "connections": connections,
                "growth_plan": growth_plan,
                "lang": target_lang
            }
        }

    except Exception as e:
        # --- 关键：如果报错，将错误堆栈抛出到前端 ---
        full_error = traceback.format_exc()
        print(f"Backend Error: {full_error}") # 服务器日志也能看到

        # 定义不同语言的错误消息
        error_messages = {
            "zh-cn": "导入失败，请检查后端服务",
            "zh-tw": "匯入失敗，請檢查後端服務",
            "en": "Import Error, please check the backend service"
        }

        error_message = error_messages.get(target_lang, "Import Error")

        return {
            "success": False,
            "error": error_message,
            "debug_info": full_error, # 这里包含了报错的具体行数
            "data": {
                "summary": "处理失败",
                "connections": "N/A",
                "growth_plan": f"错误详情: {str(e)}"
            }
        }
