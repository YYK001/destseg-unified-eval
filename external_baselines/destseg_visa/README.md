# DeSTSeg 在 VisA 上的正常样本训练与统一复评

这是官方 DeSTSeg 方法在 VisA 的训练适配，不是作者发布的 VisA 权重或官方 VisA benchmark。使用官方源码 commit `f168ba576e0a917ae5e037b677e7d37b741ea3d8`，保留上游许可证，不修改上游网络。不得使用 MVTec/BTAD checkpoint 初始化 VisA。

## 固定定义与复用

- 使用项目原有 `visa_one_class_records` 解析原版 `split_csv/1cls.csv`。12 类共 8659 张正常训练图、2162 张测试图（962 正常、1200 异常）；逐类计数与已有 PatchCore VisA 档案一致。只接受此划分，不把全量正常图重新随机切分。
- 训练读取 CSV 元数据以验证划分，但只检查和打开当前类别的正常训练图，不打开测试图或 mask，不用测试指标选权重。没有额外 calibration split。
- 原始 `DeSTSeg(dest=True, ed=True)`：ImageNet ResNet18 teacher，随机 student/segmentation。复用 BTAD runner 的官方 SGD、损失、1000 步 student + 4000 步 segmentation、每类 seed42、batch32、workers2、FP32。学生 cosine loss 仍按官方 batch 求和。
- 直接调用上游 `MVTecDataset.__getitem__`，只替换 VisA JPG 文件清单；复用 DTD 5640 图及官方 Perlin 合成。所有类别关闭旋转，这是冻结的 VisA 适配选择，不声称是作者的 VisA 默认配置。
- 固定最终 5000 步；保存 `DeSTSeg_VISA_5000_<category>.pckl`，严格加载。smoke 每阶段各20步，不能代替正式权重。当前不支持中断续训：快照不含优化器、RNG、数据迭代状态。
- 主分支 `output_segmentation` 已经 sigmoid；整图 RGB 双线性256、官方 ImageNet 归一化、eval/no_grad；输出双线性到256，align_corners=False；图像分数使用该图最高100像素均值，不加平滑或归一化。
- 保存 FP32 主/辅助图、Top-100分数及原图尺寸；统一映射为原图 H//4、W//4 全图，不裁剪、不补边、不重算图像分数。
- 统一 GT 复用原始 mask 最近邻缩放后 `>0`；官方指标路径先把原始 mask 所有 `>0` 前景变成0/255，再按官方 ToTensor、双线性 antialias Resize、>=0.5。支持 VisA 多种前景值。空异常 mask 保留，官方 mask 派生标签与 CSV 标签差异写入 `label_audit.csv`。
- 两套评价复用 MVTec cached-output evaluator：`official` 调用原始 `eval.evaluate`，主/辅助分支分别保留 IAP/IAP90 等指标，IAP不是 Image AP；`unified` 复用 PatchCore calculate/summarize、快速 CUDA AUPRO200阈值、average precision、1%/5% FPR 与区域/小区域覆盖。类别等权 macro，不混合12类像素。

## 环境与数据检查

复用已在 Kaggle 验证的独立环境：Python3.10、torch2.0.0+cu118、torchvision0.15.1+cu118、timm0.6.12、torchmetrics0.10.3、anomalib0.4.0。依赖安装详见相邻 `destseg_mvtec_pretrained/README.md`；不要修改 Notebook 默认环境。每阶段会保存实际依赖版本和官方源码状态。

以下是 Bash 命令，Notebook 可放进 `%%bash` 单元。把数据根目录改为实际包含 `split_csv/1cls.csv` 的目录。源码更新须等这些改动同步到 GitHub 后再 pull。

```bash
cd /kaggle/working/destseg-unified-eval
git pull --ff-only
git submodule update --init --recursive
PY=/kaggle/working/destseg-venv/bin/python
VISA_ROOT=/kaggle/input/YOUR_VISA_DATASET/VisA_20220922
DTD_ROOT=/kaggle/working/destseg_assets/dtd/images
$PY -m external_baselines.destseg_visa.check_data --dataset-root "$VISA_ROOT"
$PY -m external_baselines.destseg_btad.prepare_dtd --directory /kaggle/working/destseg_assets
$PY -m unittest discover -s external_baselines/destseg_visa/tests -v
$PY -m unittest discover -s external_baselines/destseg_btad/tests -v
$PY -m unittest discover -s external_baselines/destseg_mvtec_pretrained/tests -v
```

DTD 已完整时跳过下载；本地不下载数据/权重。模块导入和测试均不会启动正式训练。

## 单类 smoke

```bash
cd /kaggle/working/destseg-unified-eval
PY=/kaggle/working/destseg-venv/bin/python
MOD=external_baselines.destseg_visa
VISA_ROOT=/kaggle/input/YOUR_VISA_DATASET/VisA_20220922
DTD_ROOT=/kaggle/working/destseg_assets/dtd/images
SMOKE=/kaggle/working/destseg_visa_smoke_$(date -u +%Y%m%dT%H%M%S)
$PY -u -m "$MOD" train --dataset-root "$VISA_ROOT" --dtd-root "$DTD_ROOT" \
  --output-dir "$SMOKE" --categories candle --device cuda:0 --batch-size 32 --workers 2 --smoke-steps 20
$PY -u -m "$MOD" infer --dataset-root "$VISA_ROOT" --training-dir "$SMOKE" \
  --output-dir "${SMOKE}_predict" --categories candle --device cuda:0 --limit 2
```

检查两个训练阶段loss有限、显存、耗时、complete.json、strict加载、两张测试图分数保存。smoke禁止进入正式复评；本地无科学计算环境，不把语法和合成测试当成真实复现通过。

## 双 T4 完整流程：训练 → 推理 → 两套评价 → 打包

默认每卡6类，类别独立进程、每类加载后释放模型；无DDP。推理batch1。类别列表按交错方式分到两卡。日志实时打印，并保存到 OUT_logs。

```bash
cd /kaggle/working/destseg-unified-eval
PY=/kaggle/working/destseg-venv/bin/python
VISA_ROOT=/kaggle/input/YOUR_VISA_DATASET/VisA_20220922
DTD_ROOT=/kaggle/working/destseg_assets/dtd/images
STAMP=$(date -u +%Y%m%dT%H%M%S)
TRAIN=/kaggle/working/destseg_visa_train_$STAMP
OUT=/kaggle/working/destseg_visa_eval_$STAMP
$PY -u -m external_baselines.destseg_visa.pipeline --phase all \
  --dataset-root "$VISA_ROOT" --dtd-root "$DTD_ROOT" \
  --training-dir "$TRAIN" --output-dir "$OUT" \
  --devices cuda:0 cuda:1 --batch-size 32 --workers 2 --seed 42
```

完成后自动产生 `${OUT}_report.zip`：逐类和macro两套指标、FPR结果、运行配置、清单/逐图分数、完成状态、loss、耗时/显存、日志；不包含权重、预测数组、数据集或虚拟环境。保留原始模型与预测，下载这个小包即可写报告，不必通过API扫描整个环境。

12类总训练量明显大于BTAD。先根据 smoke 和可用会话时间决定是否分批，不保证一次 Kaggle 会话能跑完。可以使用 `--phase train --categories candle capsules cashew chewinggum` 分批训练；下一批使用同一 TRAIN 和不同类别。已存在的类别阶段不覆盖，中断类别需另用新目录重新训练。跨会话需要自行保留已经完成的训练目录，不能仅依赖内存变量。

全部训练完成后，可用 `--phase evaluate`、同一 TRAIN、新的 OUT 执行推理和两套评价，不会重训。选定部分类别的汇总明确标为 selected_categories，不是完整12类macro。

## 仅重评已有预测（不重新前向）

在同一个 Bash 单元先定义 PY、VISA_ROOT、OUT；首次手工评价用 v1，已存在则换 v2 等新名称，禁止覆盖。以下顺序示例用一张T4；完整pipeline已提供双卡分配。

```bash
$PY -u -m external_baselines.destseg_visa unified --dataset-root "$VISA_ROOT" \
  --output-dir "$OUT" --device cuda:0 --name v2
$PY -m external_baselines.destseg_visa summarize --output-dir "$OUT" --kind unified --name v2
$PY -u -m external_baselines.destseg_visa official --output-dir "$OUT" --device cuda:0 --name v2
$PY -m external_baselines.destseg_visa summarize --output-dir "$OUT" --kind official --name v2
```

## 本地验证与待验证项

标准库测试覆盖完整划分计数、缺失样本拒绝、训练不依赖测试文件、类别选择/双卡分组、协议隔离及语法。运行时mask测试需在独立环境执行；复用 MVTec 的预处理/Top100/非方形坐标/保存回读/macro测试和BTAD的两阶段优化器测试。

尚未执行 VisA 真实 smoke、正式训练或全量评测，尚无 VisA DeSTSeg 指标。所有训练资源和效果需由实际 Kaggle 运行确认。
