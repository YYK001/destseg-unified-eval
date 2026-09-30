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

默认每卡6类，类别独立进程、每类加载后释放模型；无DDP。每类训练后立即推理（batch1）和两套评价，然后才启动下一类；类别列表交错分到两卡。日志实时打印，并保存到 OUT_logs。

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

### 新的从头运行入口：面向12小时会话

```bash
cd /kaggle/working/destseg-unified-eval
/kaggle/working/destseg-venv/bin/python -u -m external_baselines.destseg_visa.fresh_run \
  --dataset-root /kaggle/input/datasets/tensura3607/amazon-visa-anomaly \
  --dtd-root /kaggle/working/destseg_assets/dtd/images --max-hours 10.75
```

保留Notebook的环境安装和DTD/VisA准备单元，移除旧的正式训练调用，使用上述唯一正式入口。自动先运行本地/运行时测试，再做candle、cashew各20+20步短训练检查两类退出路径，通过才用新目录从头训练。不会加载之前任何DeSTSeg类别权重；teacher仍使用官方ImageNet初始化。

Kaggle文档规定Save & Run从头到尾须在12小时内完成，含安装准备。本入口默认10.75小时（从入口启动计算），留出环境准备和平台保存余量；如果环境准备已花超过1小时，请进一步降低预算。pipeline在预算末尾另留5分钟归档。12类全部排入队列，但不是保证12类都能在该窗口完成。每卡在启动下一类前用初始2.75小时、已观测同卡整类耗时乘1.15的较大值判断是否有时间，时间不足就记录pending并结束。运行中的进程也有硬工作截止时间，故极端情况下某类会被中断，该类不标完成。

只有完成训练、推理、两套评价的类别进入最终汇总。`pipeline_status.json`明确记录requested/categories、completed、pending和complete/partial_budget/failed。部分结果不冒充12类macro。发生失败时入口会打印RUN FAILED并写launch_failure.json，随后正常结束Notebook以利平台发布文件；**进程退出码0不是实验成功证明**，以状态文件和类别完成记录为准。环境崩溃或平台强杀仍无法保证输出发布。

2026-09-30资源统计修复：VisA各阶段与DeSTSeg训练改用`AllocatorResources`，不启动nvidia-smi采样线程、不创建采样子进程、不额外调用CUDA synchronize。保留各进程各设备PyTorch allocated/reserved峰值；整卡显存观测明确标为disabled，不把其缺失填为0。训练阶段计时现在包含该阶段checkpoint保存，属于host wall time，不与旧版独立保存计时直接比较。最终模型和complete.json写入位于统计context退出之前；统计写入报错仅告警，不阻止已完成模型保存。仍可能存在训练、磁盘或驱动故障，启动器监控负责停止和记录，未在本地无GPU环境宣称真实长跑通过。

### 训练结束停滞后的运行保护更新

2026-09-30 的首批后台日志显示 candle 已打印5000步但没有类别完成消息，GPU1另两类完成后整个流程仍未退出。由于未取到后台进程栈或权重目录，实际阻塞位置仍未确认，不把本更新声称为已经验证的根因修复。

- DataLoader 显式使用 `spawn`，避免在CUDA与采样线程启动后fork；仍维持workers2、每轮重建worker，不启用persistent_workers，不改训练步数、优化器或损失。启动方式记录到run.json；实际耗时和随机增强序列不承诺与旧运行逐值一致，需重新做smoke。
- 每张卡逐类别启动独立进程，显示PID、保存checkpoint、写完成状态、资源统计结束、worker清理和进程退出日志。阶段收尾超过120秒会打印一次Python栈。
- 启动器每60秒输出等待状态；训练/推理连续900秒无子进程输出则判失败并停止进程组，取消另一卡当前任务。评价使用独立3600秒无输出上限，可通过 `--idle-timeout` / `--evaluation-idle-timeout` 显式修改；心跳不算训练进展。
- 成功生成report.zip；捕获到失败仍生成partial_records.zip及pipeline_failure.json，不伪称评价完成。凡已有原子保存的最终权重，另生成recovery.zip（包含权重与记录），恢复前必须验证complete.json、协议和strict加载；不自动认定残缺阶段可恢复。
- 这些保护只能处理启动器仍在运行时的异常。Kaggle强制终止、磁盘写满或用户取消可能阻止归档，不能保证平台发布输出。后台Notebook包装cell应在非零退出时打印失败并结束该cell，不再启动其他实验，也不要主动关闭整个Session。

标准库测试覆盖完整划分计数、缺失样本拒绝、训练不依赖测试文件、类别选择/双卡分组、协议隔离及语法。运行时mask测试需在独立环境执行；复用 MVTec 的预处理/Top100/非方形坐标/保存回读/macro测试和BTAD的两阶段优化器测试。

尚未执行 VisA 真实 smoke、正式训练或全量评测，尚无 VisA DeSTSeg 指标。所有训练资源和效果需由实际 Kaggle 运行确认。
