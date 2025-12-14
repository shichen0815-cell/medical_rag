# remote_rewrite_llm.py
import requests
import logging
from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)

class RemoteRewriteLLM(Runnable):
    """
    远程 Query Rewrite LLM
    - OpenAI 兼容接口
    - vLLM / SGLang / FastAPI 均可
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        timeout: float = 2.0,
        api_key: str | None = None,
    ):
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.api_key = api_key

    def invoke(self, input: str, **kwargs) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": input}
            ],
            "temperature": 0.0,
            "max_tokens": 64
        }

        try:
            resp = requests.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()

            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

        except Exception as e:
            logger.warning(f"[Rewrite LLM] 远程调用失败: {e}")
            raise
