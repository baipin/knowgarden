from openai import OpenAI
import os

def run_ingestion(client, content):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个知识总结专家。"},
            {"role": "user", "content": f"请总结内容：{content}"}
        ]
    )
    return response.choices[0].message.content