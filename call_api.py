"""调用告警匹配 Web 服务的小客户端。

确保服务已启动：uvicorn api:app --host 127.0.0.1 --port 8000
然后运行：python call_api.py
改下面的 alert 文本即可测试不同告警。
"""

import json
import urllib.request

API_URL = "http://127.0.0.1:8000/match"

alert = "2026-06-24 网关 10.0.0.1 大量 502 后端无响应"  # 改成你要测试的告警


def call(alert_text: str) -> dict:
    data = json.dumps({"alert": alert_text}).encode("utf-8")
    req = urllib.request.Request(
        API_URL, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


if __name__ == "__main__":
    print(f"告警: {alert}\n")
    result = call(alert)
    print(json.dumps(result, ensure_ascii=False, indent=2))
