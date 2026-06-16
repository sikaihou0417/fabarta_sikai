import subprocess
import time
import sys

# 配置：改成你本机已下载的 Qwen 模型名称
MODEL_NAME = "qwen3:8b"   # 或者 "qwen:7b", "qwen2:7b" 等

def ask_question(question):
    # 构建命令：ollama run 模型名，然后通过管道输入问题
    # 注意：ollama run 默认会进入交互模式，我们使用 echo 将问题传给 stdin
    cmd = ['ollama', 'run', MODEL_NAME]
    
    start_time = time.time()
    
    try:
        # 使用 Popen 以便实时获取输出（流式）
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        # 将问题写入进程的 stdin，然后关闭 stdin 表示输入结束
        stdout, stderr = proc.communicate(input=question, timeout=120)
        end_time = time.time()
        elapsed = end_time - start_time
        
        if proc.returncode == 0:
            print("\n🤖 回答：")
            print(stdout.strip())
            print(f"\n⏱️ 总耗时：{elapsed:.2f} 秒")
        else:
            print(f"❌ 命令执行失败，退出码 {proc.returncode}")
            if stderr:
                print("错误信息：", stderr)
    except subprocess.TimeoutExpired:
        proc.kill()
        print("❌ 模型响应超时（120秒）")
    except FileNotFoundError:
        print("❌ 找不到 ollama 命令，请确保 Ollama 已安装并在 PATH 中")
    except Exception as e:
        print(f"❌ 发生错误：{e}")

def main():
    print(f"使用本地模型：{MODEL_NAME}")
    print("输入问题后按回车，输入 exit 退出。\n")
    while True:
        user_input = input("🧑‍💻 你：")
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input.strip():
            continue
        ask_question(user_input)

if __name__ == "__main__":
    main()