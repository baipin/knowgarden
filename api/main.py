import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# 导入你的 Agent 模块 (注意：在 api 目录下，使用相对导入)
try:
    from .agent1 import run_ingestion
    from .agent2 import run_synthesis
    from .agent3 import run_growth
except ImportError:
    # 兼容本地直接运行 main.py 的情况
    import agent1 as agent1_mod
    import agent2 as agent2_mod
    import agent3 as agent3_mod
    run_ingestion = agent1_mod.run_ingestion
    run_synthesis = agent2_mod.run_synthesis
    run_growth = agent3_mod.run_growth

load_dotenv()

app = FastAPI()

# 允许跨域请求 (前端 index.html 在根目录时必须配置)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 DeepSeek 客户端 (统一在此初始化并传递给各 Agent)
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 定义前端发送的数据结构
class KnowledgeRequest(BaseModel):
    content: str

# --- 路由定义 ---

@app.get("/api/health")
def health_check():
    """用于检查后端是否部署成功的探针"""
    return {"status": "Garden Engine is Online", "provider": "DeepSeek"}

@app.post("/api/grow")
async def grow_knowledge(request: KnowledgeRequest):
    """
    核心接口：执行 Agent 链式逻辑
    1. Ingestion (总结) -> 2. Synthesis (连接) -> 3. Growth (行动)
    """
    user_content = request.content.strip()
    
    if not user_content:
        raise HTTPException(status_code=400, detail="内容不能为空")

    try:
        # --- STAGE 1: Ingestion (Agent 1) ---
        # 提取核心摘要
        summary = await asyncio.to_thread(
            run_ingestion, client, user_content
        )

        # --- STAGE 2: Synthesis (Agent 2) ---
        # 基于摘要发现关联 (传入 Stage 1 的结果)
        connections = await asyncio.to_thread(
            run_synthesis, client, summary
        )

        # --- STAGE 3: Growth (Agent 3) ---
        # 催生学习计划 (传入 Stage 1 & 2 的结果)
        growth_plan = await asyncio.to_thread(
            run_growth, client, summary, connections
        )

        # 返回完整数据包给前端
        return {
            "success": True,
            "data": {
                "title": summary[:20] + "...", # 简单截取作为标题
                "summary": summary,
                "connections": connections,
                "growth_plan": growth_plan,
                "raw_input": user_content[:50]
            }
        }

    except Exception as e:
        # 打印详细错误到 Vercel 日志
        print(f"Agent Chain Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"园丁机器人遇到了点麻烦: {str(e)}")

# 如果需要在本地直接测试运行：python api/main.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)