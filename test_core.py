# --- 运行实验 ---
from core import CompressedSemanticKernel

# 模型路径作为参数传入
MODEL_PATH = r"../diffusion2/Qwen/Qwen3-0.6B"
kernel = CompressedSemanticKernel(MODEL_PATH)

# 1. 存入 10 条语义规则 (模拟 4-bit 压缩转存)
data_bank = [
    "Security: All encryption keys must be stored in HSM.",
    "Network: Port 80 is for HTTP, 443 for HTTPS.",
    "Admin: Access to root requires biometric scan.",
    "Policy: Data retention period is 5 years.",
    "System: Kernel version 6.1 active.",
    "Hardware: GPU temperature threshold 85C.",
    "API: Rate limit set to 1000 req/min.",
    "Logic: If error 404, redirect to index.",
    "Auth: OAuth2 tokens expire in 3600s.",
    "Path: Config files located in /etc/sys/"
]

for i, txt in enumerate(data_bank):
    kernel.store_rule(txt, i)
    print(f"已压缩转存规则 {i}")

# 2. 精准召回测试
hit, score = kernel.recall("Where should encryption keys be stored?")
if hit:
    print(f"【命中规则】: {hit['content']}")
    print(f"【语义得分】: {score:.4f} (4-bit 压缩后)")

# 3. 诱发崩溃测试 (输入无关信息)
kernel.recall("What is the weather today?")