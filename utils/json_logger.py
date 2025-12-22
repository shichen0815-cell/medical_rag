import json
from typing import Any

def to_json_str(data: Any, *, ensure_ascii=False, indent=2) -> str:
    """
    将 Python 对象安全转换为 JSON 字符串
    - None  -> null
    - True  -> true
    - False -> false
    """
    return json.dumps(
        data,
        ensure_ascii=ensure_ascii,
        indent=indent,
        default=str  # 避免 datetime / enum 等对象炸日志
    )
