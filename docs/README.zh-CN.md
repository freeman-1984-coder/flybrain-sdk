# flybrain-sdk：无需 CUDA 的果蝇连接组仿真 SDK

0.4 新增：统一 Session 循环、可替换输入/输出/环境、避障和声音模板、完整过程回放，以及可继续生成的 WAV。试用[示例与音频](https://freeman-1984-coder.github.io/flybrain-sdk/demos.html)，查看[接入指南](demo-kits.md)。示例页是明确标注的录制回放；实验室仍是浏览器现场仿真。


无需安装也能先玩：[在线回路实验室](https://freeman-1984-coder.github.io/flybrain-sdk/lab.html)。可给细胞刺激、静默输出、查看活动；真实回路点击后才下载。页面还能导出可编辑的单文件 HTML demo，或导出实验记录，在 Python 中重放并核验结果。浏览器当前使用 JavaScript CPU，不需要 CUDA 或 WASM。详见[实验室与 demo 指南](browser-lab.md)。


## 0.2 alpha：真实小回路与开放接口

新增 313 个真实 MaleCNS 神经元、20,607 条连接的实验模型，约 3.8 MB，
使用时显式下载，之后可以离线重用。连接来源真实；LIF 参数、刺激和动作映射是建模假设。

```python
from flybrain import FlyBrain

brain = FlyBrain.load("male-cns-escape-v1", download=True)
gf = brain.neurons.select(cell_type="DNp01")
brain.bind_readout({"flash": gf})
brain.stimulate("looming_left", duration_ms=100)
brain.advance(duration_ms=100)
print(brain.action().to_dict())
```

支持按 ID/注释选择细胞、直接注入电流、只观察指定细胞，以及可恢复的静默干预。
新检查点保存自定义输出和刺激，继续兼容旧检查点。CUDA/WASM 尚未实现。
详见 [API](api.md)、[模型卡](../models/male-cns-escape-v1/README.md) 和
[整体设计](rfcs/0001-open-runtime-and-demo-kits.zh-CN.md)。


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

## Godot 游戏接入（0.4.0a4）

安装 Python 包后，在仓库目录运行 `python examples/godot/bridge.py`，
再用 Godot 4.7.2 打开 `examples/godot/project.godot`，按 F5 和 Run / pause。
默认人工小回路离线运行；服务增加 `--model male-cns-escape-v1 --download`
可按需下载真实 313 神经元子回路。游戏负责世界，SDK 负责神经计算，
感觉编码与动作读出均可替换。支持成对保存/恢复世界与脑状态，断线停止。

[完整使用说明](../examples/godot/README.md) · [网站示例](https://freeman-1984-coder.github.io/flybrain-sdk/demos.html#godot)
