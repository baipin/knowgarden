import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# 导入你的 Agent 模块
# 注意：在 Vercel 部署时，建议使用绝对导入或确保路径在 PYTHONPATH 中
from agents.ingestion import run_ingestion
from agents.synthesis import run_synthesis
from agents.growth import run_growth

load_dotenv()

app = FastAPI()

# 跨域配置：允许根目录的 index.html 访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 DeepSeek 客户端
# 统一在这里初始化并传递给 Agent，避免在每个文件里重复创建客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

class KnowledgeRequest(BaseModel):
    content: str

@app.post("/api/grow")
async def grow_knowledge(request: KnowledgeRequest):
    if not request.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")

    try:
        # --- STAGE 1: Ingestion ---
        # 使用 asyncio.to_thread 防止同步 SDK 阻塞异步 FastAPI
        summary = await asyncio.to_thread(
            run_ingestion, client, request.content
        )

        # --- STAGE 2: Synthesis ---
        # 传入上一步的结果
        connections = await asyncio.to_thread(
            run_synthesis, client, summary
        )

        # --- STAGE 3: Growth ---
        # 结合前两步的信息生成最终建议
        growth_plan = await asyncio.to_thread(
            run_growth, client, summary, connections
        )

        # 返回统一的 JSON 结果
        return {
            "success": True,
            "data": {
                "summary": summary,
                "connections": connections,
                "growth_plan": growth_plan,
                "status": "fully_grown"
            }
        }

    except Exception as e:
        print(f"Workflow Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Agent workflow failed")

@app.get("/api/health")
def health():
    return {"status": "ok", "agents": "3/3 online"}