# server.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from contextlib import asynccontextmanager
import os
import logging
from datetime import datetime

# 配置日志器（模块级）
# 配置更详细的日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/prompt_logs.log'),  # 同时输出到文件
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"HF_ENDPOINT set to: {os.environ.get('HF_ENDPOINT')}")
# # 全局日志配置（仅在首次导入时生效）
# if not logger.handlers:
#     handler = logging.StreamHandler()
#     formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#     logger.setLevel(logging.INFO)  # 可根据需要调整为 logging.DEBUG

# app = FastAPI()
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    pipe("你好")  # 模型预热
    yield
    # shutdown（可选）

app = FastAPI(lifespan=lifespan)

def get_torch_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"   # Apple Silicon
    else:
        return "cpu"

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map=get_torch_device(),
    trust_remote_code=True
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=80,
    temperature=0.0,
    do_sample=False
)

class ChatReq(BaseModel):
    model: str
    messages: list

@app.post("/v1/chat/completions")
def chat(req: ChatReq):
    # 记录完整的请求信息
    logger.info("=" * 80)
    logger.info(f"收到请求时间: {datetime.now().isoformat()}")
    logger.info(f"请求模型: {req.model}")

    # 记录完整的messages结构
    logger.info("完整的messages结构:")
    for i, msg in enumerate(req.messages):
        logger.info(f"  [{i}] role={msg['role']}")
        logger.info(f"      content={msg['content'][:200]}...")  # 只显示前200字符

    # 提取最后一个用户消息作为prompt
    user_message = None
    for msg in reversed(req.messages):
        if msg["role"] == "user":
            user_message = msg["content"]
            break

    if user_message:
        logger.info(f"最终用户prompt: {user_message}")

    logger.info(
        "FastAPI Rewrite request received, model=%s, prompt=%s",
        req.model,
        req.messages[-1]["content"]
    )
    prompt = req.messages[-1]["content"]
    logger.info(f"实际发送给模型的prompt长度: {len(prompt)} 字符")
    out = pipe(prompt)[0]["generated_text"]
    return {
        "choices": [
            {"message": {"role": "assistant", "content": out}}
        ]
    }

@app.get("/health")
def health():
    return {"status": "okk"}

