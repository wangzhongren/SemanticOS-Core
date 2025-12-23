import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class CompressedSemanticKernel:
    def __init__(self, model_path, torch_dtype=torch.float16, device_map="cpu"):
        """
        初始化压缩语义内核
        
        Args:
            model_path (str): 预训练模型路径
            torch_dtype: 模型数据类型，默认 torch.float16
            device_map: 设备映射，默认 "cpu"
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch_dtype, 
            device_map=device_map
        )
        
        # 模拟系统内存中的压缩事实载荷区 [cite: 28]
        self.compressed_memory = []
        self.threshold = 0.70  # 内核寻址中断阈值 [cite: 49]

    def compress_4bit(self, tensor):
        """将 FP16 向量压缩为 4-bit 存储"""
        q_min, q_max = -8, 7
        min_val, max_val = tensor.min(), tensor.max()
        scale = (max_val - min_val) / (q_max - q_min)
        # 量化过程
        q_tensor = torch.round((tensor - min_val) / (scale + 1e-8) + q_min).clamp(q_min, q_max)
        return q_tensor.to(torch.int8), scale, min_val

    def decompress_4bit(self, q_tensor, scale, min_val):
        """召回时解压回 FP16 进行语义计算"""
        q_min = -8
        return (q_tensor.float() - q_min) * scale + min_val

    def store_rule(self, text, rule_id):
        """提取语义并压缩转存 [cite: 26, 30]"""
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # 获取最后一个 token 的隐藏状态作为 Key (语义地址) [cite: 17]
            key_vector = outputs.hidden_states[-1][:, -1, :]
            
            # 执行 4-bit 压缩存储
            q_key, scale, min_v = self.compress_4bit(key_vector)
            self.compressed_memory.append({
                "id": rule_id,
                "q_key": q_key,
                "scale": scale,
                "min_v": min_v,
                "content": text
            })

    def recall(self, query_text):
        """执行语义寻址召回 [cite: 55]"""
        inputs = self.tokenizer(query_text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            query_vector = outputs.hidden_states[-1][:, -1, :]

        best_score = -1
        best_match = None

        print(f"\n[内核寻址] Query: '{query_text}'")
        for item in self.compressed_memory:
            # 动态解压并计算余弦相似度 (模拟点积寻址) [cite: 14]
            recovered_key = self.decompress_4bit(item["q_key"], item["scale"], item["min_v"])
            score = F.cosine_similarity(query_vector, recovered_key, dim=-1).item()
            
            if score > best_score:
                best_score = score
                best_match = item

        # 逻辑判定与异常协议 [cite: 48, 71]
        if best_score < self.threshold:
            self.print_crash_report(query_text, best_score)
            return None
        
        return best_match, best_score

    def print_crash_report(self, q, score):
        """模拟论文中的《语义内核崩溃报告》 [cite: 51]"""
        print("\n" + "!"*40)
        print("《语义内核崩溃报告》")
        print(f"触发事件: 语义地址未命中 (Address Miss)")
        print(f"当前 Query: {q}")
        print(f"最高匹配得分: {score:.4f} (低于阈值 {self.threshold})")
        print(f"系统判定: 类型-Q 无法分类原型 [cite: 67]")
        print("!"*40 + "\n")