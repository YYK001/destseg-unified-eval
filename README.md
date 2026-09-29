# destseg-unified-eval

Official DeSTSeg pretrained inference and unified evaluation on MVTec AD.

使用作者官方模型及 MVTec AD 15 类预训练权重，保存主分支 FP32 异常图和官方 Top-100 图像分数，并分别运行官方指标与项目统一复评。当前只支持 **MVTec AD**，不训练、不扩展 VisA/BTAD，不使用 DINOv3/MAD/LOCAL/GUIDED 检测或校准。

**仓库只保存代码、测试、依赖说明和文档。** 权重、数据集、预测、评价结果、实验报告、运行日志和本地环境均不入库。没有在本地下载权重；权重之后在 Kaggle/服务器端准备。

## Kaggle 克隆与同步

在开启 Internet 的 Kaggle notebook 中运行：

```python
!git clone --recurse-submodules https://github.com/YYK001/destseg-unified-eval.git /kaggle/working/destseg-unified-eval
%cd /kaggle/working/destseg-unified-eval
```

后续同步：

```python
%cd /kaggle/working/destseg-unified-eval
!git pull --ff-only
!git submodule update --init --recursive
```

官方源码以 submodule 固定到 `f168ba576e0a917ae5e037b677e7d37b741ea3d8`，保留原始 Apple 许可证。克隆与同步只获取源码，不下载权重、不启动推理。若普通 clone 未初始化 submodule，执行上面的 submodule update 即可。

依赖要求与之前 PatchCore 不同：官方依赖为 torch2.0.0、torchvision0.15.1、timm0.6.12、torchmetrics0.10.3、anomalib0.4.0，运行方案使用**独立 Python3.10 环境**。不要将这套旧版依赖直接安装进现有 Kaggle Python3.12/PatchCore 环境。克隆完成不代表推理环境已验证。

## 运行入口

参阅 [完整运行文档](external_baselines/destseg_mvtec_pretrained/README.md)，包含路径参数化和双 T4 按类别分进程的完整命令：

1. 准备独立环境与官方权重，`check-weights --load` 核对全部 15 类并严格加载。
2. 单类 `infer --limit 2 --compare-official`，对照官方输入和输出。
3. 两个独立进程 `infer --shard 0/2 --device cuda:0` 与 `--shard 1/2 --device cuda:1`，保存全测试集预测。
4. `official` 回放保存的两分支输出，通过原始 `eval.evaluate` 计算官方指标；另提供未包装 `eval.py` 交叉检查命令。
5. `unified` 从保存预测计算五指标、快速 CUDA AUPRO 和 1%/5% FPR 诊断。
6. 两套结果分别 `summarize`，按类别等权 macro，不能混为一套官方结果。

路径示例：代码根目录为 `/kaggle/working/destseg-unified-eval`；数据和权重指向实际 `/kaggle/input/...`；`--output-dir` 指向代码仓库外的 `/kaggle/working/destseg_outputs_<唯一名称>`。

无需模型环境即可查看入口：

```python
!python -m external_baselines.destseg_mvtec_pretrained --help
```

## 代码复用与验证范围

官方网络/预处理/指标源码未修改；公共数据与评价依赖按原模块路径保留，复用已有 PatchCore/GLASS 数据清单、全图映射、评价函数与类别汇总。这些文件中的其他方法定义不是本仓库运行入口，不需要 DINO 模型权重，也不执行其他方法训练。仓库不包含 PatchCore/GLASS 上游检测模型。

本地标准库测试 **4 项通过**；5 项 tensor/图像/NPZ 数值测试因本地没有科学计算环境而跳过。真实权重严格加载、真实推理、官方评价回放对照及 CUDA 复评尚待运行环境验证，不能称为正式复现成功。详见 [验证说明](external_baselines/destseg_mvtec_pretrained/LOCAL_VALIDATION.md)。

```bash
python -m unittest discover -s external_baselines/destseg_mvtec_pretrained/tests -v
```

有完整运行环境后必须确认 5 项 runtime 测试实际执行，没有 skipped。
