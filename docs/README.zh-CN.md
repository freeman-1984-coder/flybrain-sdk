# flybrain-sdk：无需 CUDA 的果蝇连接组仿真 SDK

Python MVP 已实现 CPU 运行、感觉输入、动作读出、状态保存与恢复。
安装后运行 `python examples/quickstart.py` 即可离线体验。

```sh
python -m pip install -e ".[dev]"
python examples/quickstart.py
pytest
```

内置 toy 是人工设计的 12 神经元 LIF 回路，用来验证 SDK 的完整流程。
它不是生物果蝇脑。真实数据不会打包进安装包：目录内预置 MaleCNS v1.0、
FlyWire v783 的官方来源、大小、许可证和下载入口，使用者按需下载并缓存。

```python
from flybrain import FlyBrain, list_models, fetch_model

brain = FlyBrain.load("toy")
brain.stimulate("food", duration_ms=100)
brain.step(100)
print(brain.action().to_dict())
print([(m["id"], m["status"]) for m in list_models()])
# 明确调用才联网下载；这里只下载约 1.1 MB 的神经元 ID 文件。
paths = fetch_model("flywire-v783", assets=["neuron_ids"])
```

真实连接组是神经连接数据，还需要参数、感觉/动作映射和转换器，才能成为
可直接加载的仿真模型。目前没有宣称全脑实时运行、学习能力或真实果蝇行为。
WASM/CUDA 只预留接口。欢迎通过 Issue 和 Pull Request 一起完善；无需 GPU。

[英文首页](../README.md) · [贡献指南](../CONTRIBUTING.md) · [真实数据接入计划](real-data.md)
