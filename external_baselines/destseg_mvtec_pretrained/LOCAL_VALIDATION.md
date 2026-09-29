# 本地接入验证记录

日期：2026-09-29。原始实验工作区完成接入后，代码单独整理到本仓库；运行日志与实验结果不入库。

本轮仅代码接入与标准库验证。按用户最新要求不在本地下载权重或安装推理环境；没有连接服务器、上传、启动训练/全量评测、提交或推送 Git。官方源码已在前序步骤下载，既有代码、输出及报告未修改。

## 实际执行

1. 固定官方 commit `f168ba576e0a917ae5e037b677e7d37b741ea3d8`，检查 tracked 文件未修改及许可证/指定源码存在。
2. Python AST 解析新增全部 Python 文件，检查官方网络返回顺序与 sigmoid、Top-100 源码契约。
3. 用临时目录里的占位文件验证权重清单发现、缺失拒绝、重复候选拒绝、显式映射和同一文件重复映射拒绝。占位文件仅用于文件发现测试，从未加载为模型。
4. 验证两个类别 shard 分别为 8/7 类，无交集且并集恰为全部 15 类；拒绝未知/重复类别。
5. 实际调用 `check-weights`：发现 0/15，写出全部类别清单及未验证状态，以预期退出码 2 停止；没有导入或初始化检测模型。
6. 五个命令 `check-weights / infer / official / unified / summarize` 的 `--help` 均返回 0，可在无 torch 的本地使用。

命令：

```powershell
python -m unittest discover -s external_baselines/destseg_mvtec_pretrained/tests -v
python -m external_baselines.destseg_mvtec_pretrained check-weights --output-dir external_baselines/destseg_mvtec_pretrained/validation/local_20260929/weight_check
```

第二条命令的目录已经由本次验证生成；若复查，应使用新的输出目录，现有目录会被拒绝覆盖。

结果：**4 项标准库测试通过，5 项数值测试跳过，无失败**。当前 Python 3.12.10，未安装 torch、torchvision、numpy、Pillow、timm、scipy、scikit-learn。`OK (skipped=5)` 不代表数值验证通过。

## 已编写但未执行的 5 项数值测试

`tests/test_runtime.py` 在独立运行环境中使用真实 tensor、合成 RGB/二值 mask 和 NPZ，验证：

- 适配 RGB 预处理与原始 `MVTecDataset.__getitem__` 完全一致。
- 主分支是 segmentation，256 插值及降序排序 Top-100 分数一致，无额外 sigmoid。
- 303×517 非方形原图映射为 75×129 全图，无额外裁剪或补边。
- 保存/读取 FP32 图、图像分数、统一评价输入及 GT 快照保持一致。
- 直接调用既有类别 macro：等权而不是按样本数加权，N/A 排除规则与有效类别数正确。

这些测试不构造随机 DeSTSeg 来假称真实复现，不替代真实预训练权重 smoke。

## 仍待服务器验证

- 独立 Python3.10/旧版官方依赖环境的安装解析、导入和双 T4 CUDA 可用性。
- 真实 15 类权重完整性、每类 `strict=True` 加载，实际文件名与映射核对。
- 少量真实图像的 `--compare-official` 输入/主分支输出/Top-100 对照。
- 官方缓存回放评价与未包装 `eval.py` 的真实指标等价性。
- 既有快速 CUDA AUPRO、固定 FPR 及完整 15 类统一评价的真实运行。

本轮没有任何正式 DeSTSeg 结果数值，不以 PatchCore/GLASS 结果或测试值填充。

## 原工作区的证据文件（不随代码提交）

- `validation/local_20260929/unittest.log`：实际 unittest 输出。
- `validation/local_20260929/validation.json`：Python/依赖可用性、退出码、未运行环节。
- `validation/local_20260929/weight_check.log`：实际缺失权重检查输出。
- `validation/local_20260929/weight_check/weights.csv`：15 类对应路径和 missing 状态。
- `validation/local_20260929/weight_check/status.json`：固定源码、协议及全部缺失类别。
