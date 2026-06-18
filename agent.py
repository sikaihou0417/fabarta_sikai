import json
import requests

# ====== 配置 ======
API_KEY = "sk-8b4d73c0e60d41d8ab4680a751f8652b"
MODEL_ENDPOINT = "https://api.deepseek.com/chat/completions"   # 注意去掉前面多余的 tab 和空格

# ====== 定义工具 ======
def get_exchange_rate(from_currency, to_currency):
    """模拟查汇率（先写死数据，任务三再换真的）"""
    rates = {"USD_CNY": 7.25, "CNY_USD": 0.138, "EUR_CNY": 7.80}
    key = f"{from_currency}_{to_currency}"
    return rates.get(key, f"未找到{from_currency}到{to_currency}的汇率")

def calculate(expression):
    try:
        allowed = set("0123456789+-*/(). ")
        if all(c in allowed for c in expression):
            result = eval(expression)
            return str(result)
        else:
            return "表达式包含不支持的字符"
    except Exception as e:
        return f"计算错误: {e}"

# 工具描述（给模型看的）
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_exchange_rate",
        "description": "查询两种货币之间的汇率",
        "parameters": {
            "type": "object",
            "properties": {
                "from_currency": {"type": "string", "description": "源货币代码，如USD"},
                "to_currency": {"type": "string", "description": "目标货币代码，如CNY"}
            },
            "required": ["from_currency", "to_currency"]
        }
    }
},  # 逗号不能少
{
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "计算数学表达式的结果，支持加减乘除和括号",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，如 '100*7.25/3'"
                }
            },
            "required": ["expression"]
        }
    }
}]

# ====== 调用模型的函数 ======
def call_model(messages, tools=None):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-v4-pro",
        "messages": messages,
        "max_tokens": 1000
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    
    resp = requests.post(MODEL_ENDPOINT, headers=headers, json=payload, timeout=30)
    # print("状态码:", resp.status_code)        # 已注释
    # print("返回内容:", resp.text)            # 已注释
    return resp.json()

# ====== 工具执行函数 ======
def execute_tool(tool_call):
    func_name = tool_call["function"]["name"]
    args = json.loads(tool_call["function"]["arguments"])
    
    if func_name == "get_exchange_rate":
        return get_exchange_rate(args["from_currency"], args["to_currency"])
    elif func_name == "calculate":          # 加上计算器分支
        return calculate(args["expression"])
    return f"未知工具: {func_name}"

# ====== Agent主循环 ======
def agent_loop():
    messages = [{"role": "system", "content": "你是一个智能助手，可以使用工具帮用户查汇率。"}]
    while True:
        user_input = input("你：")
        if user_input.lower() in ["exit", "quit"]:
            break
        messages.append({"role": "user", "content": user_input})
        
        response = call_model(messages, tools=TOOLS)
        if "error" in response:
            print("API错误:", response["error"]["message"])
            continue
        if "choices" not in response or not response["choices"]:
            print("响应格式异常:", response)
            continue
        
        choice = response["choices"][0]
        message = choice["message"]
        
        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            for tool_call in tool_calls:
                result = execute_tool(tool_call)
                messages.append(message)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": str(result)
                })
            response2 = call_model(messages)
            final_choice = response2["choices"][0]
            final_message = final_choice["message"]
            print("AI:", final_message.get("content", ""))
            messages.append(final_message)
        else:
            print("AI:", message.get("content", ""))
            messages.append(message)

if __name__ == "__main__":
    agent_loop()