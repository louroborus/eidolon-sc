import httpx

schema = {
    "type": "object",
    "properties": {"check": {
        "type": "object",
        "properties": {"noul": {"type": "number"}},
        "required": ["noul"],
        "additionalProperties": False,
    }},
    "required": ["check"],
    "additionalProperties": False,
}
prompt = (
    '对话状态：\n{\n "测试": "启动检查"\n}\n\n'
    "需要你回答的问题：\n\n"
    "[check] 这是一条测试消息吗？\n  给一个 0 到 1 之间的数：越接近 1 越偏向「是」，越接近 0 越偏向「否」。\n\n"
    "只输出一个 JSON 对象，键是：check。"
)
r = httpx.post("http://127.0.0.1:8095/v1/chat/completions", json={
    "model": "local-judge",
    "messages": [
        {"role": "system", "content": "你是一个角色扮演对话里的判断模型。用户会给你对话状态和若干个打分问题，你只输出一个 JSON 对象，不要解释，不要输出任何 JSON 之外的东西。"},
        {"role": "user", "content": prompt},
    ],
    "response_format": {"type": "json_schema",
                        "json_schema": {"name": "judge", "schema": schema, "strict": True}},
    "temperature": 0,
    "max_tokens": 700,
}, timeout=300)
print("status", r.status_code)
print("raw:", repr(r.text[:1200]))
