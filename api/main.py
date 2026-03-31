import os
import sys
import asyncio
import traceback
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv


# =========================================================
# 1) 基础路径设置
# ---------------------------------------------------------
# 目的：
# - 确保当前 api 目录可以被 Python 正确导入
# - 这样同目录下的 agent1.py / agent2.py / agent3.py 可以直接 import
# =========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)


# =========================================================
# 2) 加载环境变量
# ---------------------------------------------------------
# 需要在 .env 中配置：
# DEEPSEEK_API_KEY=你的key
# =========================================================
load_dotenv()


# =========================================================
# 3) 导入各 Agent（带容错）
# ---------------------------------------------------------
# 约定：
# - agent1.py 需要提供函数：run_ingestion(client, content: str) -> str
# - agent2.py 需要提供函数：run_synthesis(client, content: str) -> str
# - agent3.py 需要提供函数：run_growth(client, content: str) -> str
#
# 注意：
# - 三个函数都应当是“同步函数”（def），因为这里统一用 asyncio.to_thread 包装
# - 如果协作者写成 async def，则这里的调用方式也要跟着改
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
# ---------------------------------------------------------
# 开发期为了方便，先放开全部来源
# 如果以后上线，建议收紧 allow_origins
# =========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# 6) 初始化 LLM Client
# ---------------------------------------------------------
# 这里使用 OpenAI 兼容写法，但 base_url 指向 DeepSeek
# 所有 Agent 都复用这个 client
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
# ---------------------------------------------------------
# 用于把前端的语言代码转换成更明确的 LLM 指令
# =========================================================
LANG_MAP = {
    "zh-cn": "Simplified Chinese",
    "zh-tw": "Traditional Chinese",
    "en": "English"
}


def build_language_instruction(target_lang: str) -> str:
    """
    根据前端传入的语言代码，构造统一的语言约束提示。

    这段提示会传给所有 Agent，确保：
    - 输出正文使用用户选择的语言
    - 标题、项目符号、例子也使用同一种语言
    - 不要中英混杂（除非用户明确要求）
    """
    selected_lang = LANG_MAP.get(target_lang, "Simplified Chinese")
    return f"""
IMPORTANT LANGUAGE RULE:
Respond ONLY in {selected_lang}.
All section headings, bullets, labels, examples, and explanations must also be in {selected_lang}.
Do not mix languages unless the user explicitly asks for it.
"""


def build_error_message(target_lang: str, error_type: str = "general") -> str:
    """
    返回给前端展示的简短错误信息。
    debug_info 会保留详细堆栈，便于开发排查。
    """
    messages = {
        "zh-cn": {
            "import": "后端模块导入失败，请检查 Agent 文件或依赖配置。",
            "general": "后端处理失败，请检查服务日志或控制台错误信息。"
        },
        "zh-tw": {
            "import": "後端模組匯入失敗，請檢查 Agent 檔案或依賴設定。",
            "general": "後端處理失敗，請檢查服務日誌或控制台錯誤訊息。"
        },
        "en": {
            "import": "Backend module import failed. Please check the Agent files or dependencies.",
            "general": "Backend processing failed. Please check the service logs or console error details."
        }
    }
    return messages.get(target_lang, messages["zh-cn"]).get(error_type, messages["zh-cn"]["general"])


def build_title_from_summary(summary: str, fallback_lang: str) -> str:
    """
    从 summary 里提取一个简短标题，给前端卡片展示使用。
    这里尽量简单稳妥，不依赖复杂规则。
    """
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
# 9) Agent 调用约定（给协作者看）
# ---------------------------------------------------------
# Agent 1: Ingestion Agent
#   函数签名：
#       run_ingestion(client, content: str) -> str
#   输入：
#       原始用户内容 + 语言约束
#   输出：
#       对输入知识的总结 / 提炼结果（纯文本）
#
# Agent 2: Connection & Synthesis Agent
#   函数签名：
#       run_synthesis(client, content: str) -> str
#   输入：
#       结构化字符串，至少包含 summary + 语言约束
#   输出：
#       关联、延展、隐藏联系、对比、补充视角等（纯文本）
#
# Agent 3: Growth Agent
#   函数签名：
#       run_growth(client, content: str) -> str
#   输入：
#       结构化字符串，至少包含 summary + connections + 语言约束
#   输出：
#       知识如何“生长”：新问题、新方向、应用场景、创作想法、下一步行动（纯文本）
#
# 统一建议：
# - 每个 Agent 只做自己的职责，不要重复前一个 Agent 的全部工作
# - 输出尽量是“可直接展示”的 markdown / 文本，不需要 JSON
# - 必须严格遵守 content 中给到的语言约束
# =========================================================


@app.post("/api/grow")
async def grow_knowledge(request: KnowledgeRequest) -> Dict[str, Any]:
    """
    主流程：
    1. 接收前端输入
    2. 调用 Agent1 做总结
    3. 调用 Agent2 找关联
    4. 调用 Agent3 催生新想法
    5. 返回给前端统一展示

    返回结构：
    {
      "success": True/False,
      "data": {
        "title": str,
        "summary": str,
        "connections": str,
        "growth_plan": str,
        "lang": str
      },
      "error": str | None,
      "debug_info": str | None
    }
    """
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
        # =================================================
        # Step 1: Ingestion Agent
        # -------------------------------------------------
        # 输入：用户原始内容 + 语言约束
        # 输出：摘要 / 提炼结果
        # =================================================
        ingestion_input = f"""
User Input:
{user_content}

{language_instruction}
"""

        summary = await asyncio.to_thread(
            agent1.run_ingestion,
            client,
            ingestion_input,
            chosen_model
        )

        # =================================================
        # Step 2: Connection & Synthesis Agent
        # -------------------------------------------------
        # 输入：来自 Agent1 的总结 + 语言约束
        # 输出：知识之间的隐藏关联、对照、扩展方向
        # =================================================
        synthesis_input = f"""
Knowledge Summary:
{summary}

{language_instruction}
"""

        connections = await asyncio.to_thread(
            agent2.run_synthesis,
            client,
            synthesis_input,
            chosen_model
        )

        # =================================================
        # Step 3: Growth Agent
        # -------------------------------------------------
        # 输入：summary + connections + 语言约束
        # 输出：知识如何继续长出新问题、新应用、新创作
        # =================================================
        growth_input = f"""
Knowledge Summary:
{summary}

Discovered Connections:
{connections}

{language_instruction}
"""

        growth_plan = await asyncio.to_thread(
            agent3.run_growth,
            client,
            growth_input,
            chosen_model
        )

        return {
            "success": True,
            "error": None,
            "debug_info": None,
            "data": {
                "title": build_title_from_summary(summary, target_lang),
                "summary": summary,
                "connections": connections,
                "growth_plan": growth_plan,
                "lang": target_lang
            }
        }

    except Exception as e:
        full_error = traceback.format_exc()
        print(f"Backend Error:\n{full_error}")

        return {
            "success": False,
            "error": build_error_message(target_lang, "general"),
            "debug_info": full_error,
            "data": {
                "title": None,
                "summary": None,
                "connections": None,
                "growth_plan": None,
                "lang": target_lang
            }
        }

# =========================================================
# 10) 路由：唤醒 Agent 4 生成 PPT (/api/generate_ppt)
# =========================================================
@app.post("/api/generate_ppt")
async def generate_ppt(request: PPTRequest) -> Dict[str, Any]:
    """
    接收 Garden 卡片数据，调用 Agent 4 转化为 Marp PPT 格式
    """
    target_lang = request.lang or "zh-cn"
    chosen_model = request.model
    language_instruction = build_language_instruction(target_lang)

    if IMPORT_ERROR:
        return {"success": False, "error": "Agent 4 not found"}

    # 构建给 Agent 4 的专业指令
    ppt_context = f"""
Topic: {request.title}
---
Summary:
{request.summary}
---
Connections:
{request.connections}
---
Roadmap:
{request.growth_plan}
---
{language_instruction}
"""

    try:
        # 调用新 Agent
        ppt_content = await asyncio.to_thread(
            agent4.run_presentation_design,
            client,
            ppt_context,
            chosen_model
        )

        return {
            "success": True,
            "data": {
                "ppt_markdown": ppt_content,
                "model_used": chosen_model
            }
        }
    except Exception as e:
        return {
            "success": False, 
            "error": "PPT Generation failed", 
            "debug_info": traceback.format_exc()
        }

# =========================================================
# 11) 运行配置
# =========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
