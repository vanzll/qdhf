import os

# 禁用 hf_transfer 下载加速
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "false"

from diffusers import StableDiffusionPipeline
import torch

model_id = "stabilityai/stable-diffusion-2-1-base"
torch_dtype = torch.float16  # 设置你需要的torch类型

# 加载模型
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    safety_checker=None,
    requires_safety_checker=False
)

# 打印模型路径确认
print(pipe._model.config._name_or_path)
