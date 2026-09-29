# DeSTSeg 官方 MVTec AD 预训练推理与复评

本目录只接入 MVTec AD 的 15 类官方预训练模型，不训练，不接入本项目的 DINOv3/MAD/LOCAL/GUIDED 检测或校准流程。项目中带这些名称的模块仅作为现有数据、指标及汇总函数的提供者。

**交付状态：本地代码接入完成；官方源码已获取且未修改；实际权重 0/15。真实严格加载、真实推理、CUDA 数值验证及两套正式评价尚未执行。** 本地没有 torch/numpy/Pillow 等科学计算依赖；标准库检查与数值测试的状态分别记录在 `LOCAL_VALIDATION.md`。这里的代码交付不能称为正式复现成功。

按 2026-09-29 的最新要求，本地只交付代码和文档，**不下载权重、不安装推理环境**。权重由用户之后在服务器准备；下文下载位置、权重检查和运行命令均为服务器使用说明，本地缺失权重不会阻塞本轮代码交付。

## 1. 官方来源与权重

- 源码：<https://github.com/apple-aiml-research/ml-destseg>
- 固定 commit：`f168ba576e0a917ae5e037b677e7d37b741ea3d8`。
- 本地源码：`../destseg_official`，保留 `.git`、原始 `LICENSE` 和 `ACKNOWLEDGEMENTS`。
- 上游许可证以原始 Apple `LICENSE` 为准，**不是 Apache-2.0**。
- 已阅读 README、constant.py、model/destseg.py、data/mvtec_dataset.py、eval.py、model/metrics.py、requirements.txt。
- 官方 README 的权重分享：<https://www.icloud.com.cn/iclouddrive/0a3OPg_3wcMs38yDpWNRnRW9Q#saved%5Fmodel>。
- 2026-09-28：浏览工具无法打开分享，Python HTTP 请求能取得 HTML 分享页，但未取得权重文件。不能据此认定链接失效或官方权重不全。
- 预期目录：`external_baselines/destseg_official/saved_model/`，也可用 `--weights-dir` 指定其他目录。不会自动训练或使用替代来源的权重。

当前缺失全部类别：bottle、cable、capsule、carpet、grid、hazelnut、leather、metal_nut、pill、screw、tile、toothbrush、transistor、wood、zipper。

`check-weights` 递归发现精确文件名 `DeSTSeg_MVTec_5000_<category>.pckl`，生成 15 类 `weights.csv`。重复候选拒绝执行；文件非空只标记 `present_not_loaded`。实际下载文件名不同可传 `--weight-map mapping.json`，内容为包含全部 15 类的 `{ "bottle": "实际相对路径.pckl", ... }`，相对路径基于 weights-dir，亦支持绝对路径。禁止同一文件映射多个类别。

`check-weights --load` 与推理均直接构造官方 `DeSTSeg(dest=True, ed=True)`，加载原始 tensor state_dict，`strict=True`；不剥离前缀、不忽略 key、不把包装 checkpoint 自动猜测成 state_dict、不以随机权重继续。文件名中的 5000 仅作匹配，不推断训练或模型选择历史。

官方 TeacherNet 构造时调用 `timm.create_model('resnet18', pretrained=True, ...)`，可能额外下载 ImageNet ResNet18 初始化权重，然后由完整 DeSTSeg state_dict 覆盖。保持原始构造不变；无网运行应先在独立环境中预热 `TORCH_HOME` 缓存。这份初始化文件不能替代 15 类 DeSTSeg 权重。

## 2. 冻结的推理定义

1. RGB 整图由 PIL `Image.BILINEAR` 缩放为 256×256；直接调用官方 Dataset 的 `final_preprocessing`，即 ToTensor 与官方 ImageNet mean/std。没有中心裁剪或补边。
2. 每次一个类别、batch size 固定为 1、指定 CUDA 设备；官方模型 `.eval()`，`torch.no_grad()`，FP32，无 AMP/额外 sigmoid。保存主 `output_segmentation` 与独立辅助 `output_de_st`。
3. 两分支按官方 eval.py 双线性上采样到 256×256，`align_corners=False`。图像分数严格使用 `torch.sort(..., descending=True)[:, :100].mean(dim=1)`。
4. 主方法始终 segmentation。没有逐类择优、平滑、归一化、截断、阈值或校准。
5. 逐图 FP32 NPZ 同时保存 `input_map`（主分支 256）、`auxiliary_map`（辅助 256）、`evaluation_map`（主分支全图 H//4、W//4）、两分支原始 Top-100 分数、身份信息及官方 mask 快照。
6. 模型只收到 RGB tensor。标签/mask 在前向后才用于审计与评价缓存，不改变模型、权重或输出选择。
7. `sample_scores.csv` / `samples.json` 保留相对路径、类别、原图尺寸、输入/评价尺寸、统一标签、官方标签和分数。分数来自 256 尺寸，统一复评不改算。

## 3. 两套评价分开保存

### 官方评价：`official`

入口调用固定 commit **原始 `eval.evaluate`**，因此使用原始 `model.metrics.AUPRO` / `IAPS` 和原始 torchmetrics AUROC/AP。主分支 `DeSTSeg` 与辅助 `DeST` 各自记录 6 项：`IAP, IAP90, AUPRO, AP, AUC, detect_AUC`。其中 AP/AUC 是像素指标，detect_AUC 是图像 AUROC，IAP/IAP90 是实例指标，**均不能把 IAP 写成 Image AP**。官方没有输出 Image AP，本适配不补造官方 Image AP。

为分离推理与评价，仅临时替换 `eval.evaluate` 的 Dataset 为顺序缓存读取器，模型参数替换为 FP32 输出回放对象；指标代码、分数公式、batch 顺序、插值调用不改。回放对象不执行检测模型。缓存的官方 mask 在推理阶段直接由原始 `MVTecDataset.__getitem__` 生成：ToTensor → 双线性 Resize，antialias=True → `<0.5` 为 0、其余为 1。官方图像标签仍由 mask 的最大值派生。

缓存输出已是 256，官方 evaluate 再调用相同尺寸插值，不引入裁剪或缩放尺寸变化。入口会校验缓存 dtype、形状、身份、mask 二值性、mask 派生标签及 Top-100 分数。官方评价不需要原数据集或模型权重，但需要官方源码和旧版评价依赖。其日志注明 `cached output replay`；保存原始打印日志和未四舍五入的 scalar CSV/JSON。

上游原始 `eval.py` 仍完整保留，可独立运行原始前向+评价，命令见后文。缓存回放与原始完整评价的真实数值等价性尚待权重到位后验证，不宣称已对齐。任何未运行或失败的官方评价都不会用统一复评值补齐。

### 统一复评：`unified`

直接复用 PatchCore 入口的 `saved_inputs`、`calculate`、`summarize`，保留：

- 图像 AUROC、Image AP 使用保存的官方 Top-100 分数及项目数据标签；覆盖既有 evaluate_fast 默认的 map-max 图像分数。
- Pixel AUROC、Pixel AP 与 AUPRO@0.3 在类别全部测试像素上计算。AP 是 average precision。
- 主分支 256 浮点异常图通过已有 `map_to_evaluation` 全图双线性映射至原 H//4、W//4，align_corners=False；无 GLASS 裁剪区域补齐。
- 原 mask 走已有 `evaluation_mask` / `_load_mask`：L 模式、PIL nearest 缩放、>0 为缺陷。
- AUPRO 保持 `fast_cuda_aupro_fpr0.3_thresholds200`，200 个全局范围线性阈值与现有四连通区域；不切换 ADEval，不允许隐式 CPU fallback。
- 固定 1%/5% FPR 使用现有 `fixed_fpr_diagnostics`：严格大于边界、整组排除 ties；输出实际 FPR、缺陷像素召回、区域平均覆盖、小区域覆盖及区域数量。小区域阈值沿用区域面积 ≤ 单图评价像素数的 0.001。
- 先逐类计算，再类别等权 macro；N/A 的小区域指标只在有效类别间平均并记录有效类别数，不按样本量或像素量加权。
- 固定 FPR 是测试集事后诊断，不是部署阈值或模型调参。

统一 GT 在第一次评价时由 `--dataset-root` 生成缓存；后续评价可省略数据集路径。`label_audit.csv` 明确记录统一标签与官方 mask 派生标签的差异，任何一方定义都不自动修改。

## 4. 复用函数与修改边界

| 来源 | 直接复用 |
|---|---|
| 官方模型 | `model.destseg.DeSTSeg`，完整 teacher/student/segmentation 网络与 forward |
| 官方数据 | `MVTecDataset`，`final_preprocessing`，`__getitem__` 的 mask 处理，constant.py 常量 |
| 官方评价 | `eval.evaluate`，`model.metrics.AUPRO/IAPS`，torchmetrics0.10.3 |
| GLASS MVTec 适配 | `glass_mvtec_pretrained.data.records`；只复用数据清单、计数验证，不使用其裁剪函数 |
| PatchCore 数据与存储 | `data.metadata/map_to_evaluation/evaluation_mask`，`storage` 的 JSON/CSV/NPZ 与阶段记录 |
| PatchCore 评价 | `evaluation.saved_inputs/calculate/summarize`，其下游既有 Image AP、快速 CUDA 像素指标、固定 FPR、mean_available |
| 资源记录 | `patchcore_official_eval.resources.Resources/environment`，补记 anomalib/torchmetrics/imgaug/kornia 等版本 |

没有修改官方源码、既有 PatchCore/GLASS 代码、既有实验输出和报告。没有增加网络实现、连通域实现或通用框架。源码检查要求固定 commit 且 tracked 文件无修改；下载的权重为 untracked 文件，不影响检查。输出目录存在即拒绝覆盖；中断产生 `failure.json` 或缺少 `complete.json`，重跑用新目录/新评价名，不静默续跑或清理旧结果。

## 5. 独立服务器环境（本轮未执行）

以下为 Linux/Bash 命令，**仅供之后在双 T4 服务器手工运行**。本轮没有连接服务器、安装环境、上传、训练或全量评测。不要直接安装到已有 PatchCore/GLASS/Kaggle Python3.12 环境。

`requirements.server.txt` 保留官方所有直接依赖版本，补充旧 anomalib 导入及项目评价所需依赖约束。优先独立 Python3.10 + torch2.0.0/torchvision0.15.1 + CUDA11.8 wheel，适配 T4。这个安装方案尚未在服务器解析/导入验证，实际运行记录以每阶段 environment.json 为准。上游无兼容补丁；如果环境检查失败应先修复独立环境，不能静默改指标实现。

```bash
set -euo pipefail
export WORKSPACE=/path/to/experiment
export MVTEC_ROOT=/path/to/mvtec_anomaly_detection
export WEIGHTS=/path/to/saved_model
export OFFICIAL="$WORKSPACE/external_baselines/destseg_official"
export RUN_BASE=/path/to/outputs
export STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
export OUT="$RUN_BASE/destseg_mvtec_${STAMP}"
export SMOKE="$RUN_BASE/destseg_smoke_${STAMP}"
export CHECK="$RUN_BASE/destseg_weights_${STAMP}"
export TORCH_HOME=/path/to/torch_cache
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
export MOD=external_baselines.destseg_mvtec_pretrained

conda create -n destseg-official python=3.10 pip=24.0 -y
conda activate destseg-official
cd "$WORKSPACE"
python -m pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r external_baselines/destseg_mvtec_pretrained/requirements.server.txt
python -m pip check
python - <<'PY'
import torch, torchvision, timm, torchmetrics, anomalib
from external_baselines.destseg_mvtec_pretrained.official import modules
from external_baselines.destseg_mvtec_pretrained.protocol import OFFICIAL_ROOT
modules(OFFICIAL_ROOT, evaluation=True)
assert torch.cuda.is_available() and torch.cuda.device_count() == 2
print('torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('torchvision:', torchvision.__version__, 'timm:', timm.__version__)
print('torchmetrics:', torchmetrics.__version__, 'anomalib:', anomalib.__version__)
for i in (0, 1):
    print(i, torch.cuda.get_device_name(i), torch.cuda.mem_get_info(i))
PY
python -m unittest discover -s external_baselines/destseg_mvtec_pretrained/tests -v
```

独立环境下 5 个 runtime tests 必须实际通过，不能仅看 unittest 的 OK 而忽略 skipped。上游 `requirements.txt` 中同时存在经 anomalib 引入的 opencv-python 与官方 opencv-python-headless，本补充文件将两者锁到相同版本，不用新 OpenCV 拉入 NumPy2。

工作区必须保留现有项目数据/评价依赖目录；不是只拷贝这个适配目录即可运行。特别是已有 `DINOv3` 下的评价工具、`patchcore_official_eval`、`glass_mvtec_pretrained` 和其现有依赖需要可导入，但不会加载 DINOv3 检测模型。以工作区根目录作为 cwd。

## 6. 权重检查 → 单类 smoke → 15 类推理

先从官方分享下载并解压真实权重至 `$WEIGHTS`，然后：

```bash
python -m "$MOD" check-weights --weights-dir "$WEIGHTS" \
  --official-root "$OFFICIAL" --output-dir "$CHECK" --load
```

默认检查全部 15 类；缺失以退出码 2 停止，加载不匹配以异常非零退出，不启动推理。`--load` 逐类 CPU 严格加载后释放，不运行模型前向。检查会写 15 类对应表，即使缺失也可查状态。

```bash
python -m "$MOD" infer --dataset-root "$MVTEC_ROOT" --weights-dir "$WEIGHTS" \
  --official-root "$OFFICIAL" --output-dir "$SMOKE" --categories bottle \
  --device cuda:0 --limit 2 --compare-official
```

smoke 尽可能取正常/异常各一张，仍按原始 Dataset 顺序保存。额外与官方 `__getitem__()['img']` 路径比较输入、主分支输出和 Top-100 分数，要求逐值相等。`--limit` 始终标记 smoke，即使限制大于样本总数也不能当正式结果汇总。smoke 不是正式评价；检查 `official_path_comparison.json` 和 `complete.json` 后再手动执行正式命令。

双卡按类别拆分为两个独立进程，模型及评价都不会跨卡。以下 shell 的 wait 只用于检查两个子进程退出码，没有 DDP 或跨进程 barrier：

```bash
mkdir -p "${OUT}_logs"
python -m "$MOD" infer --dataset-root "$MVTEC_ROOT" --weights-dir "$WEIGHTS" \
  --official-root "$OFFICIAL" --output-dir "$OUT" --device cuda:0 --shard 0/2 \
  > "${OUT}_logs/infer_gpu0.log" 2>&1 &
p0=$!
python -m "$MOD" infer --dataset-root "$MVTEC_ROOT" --weights-dir "$WEIGHTS" \
  --official-root "$OFFICIAL" --output-dir "$OUT" --device cuda:1 --shard 1/2 \
  > "${OUT}_logs/infer_gpu1.log" 2>&1 &
p1=$!
s0=0; wait "$p0" || s0=$?
s1=0; wait "$p1" || s1=$?
test "$s0" -eq 0 && test "$s1" -eq 0
```

`tail -f "${OUT}_logs/infer_gpu0.log" "${OUT}_logs/infer_gpu1.log"` 可在另一终端看进度。若希望直接打印单进程日志，去掉重定向；逐 25 张和每类末尾有进度输出。

默认 shard 0/2：bottle、capsule、grid、leather、pill、tile、transistor、zipper（8 类）。shard 1/2：cable、carpet、hazelnut、metal_nut、screw、toothbrush、wood（7 类）。也可用 `--categories` 显式选择并在汇总时指定相同集合。这里未做负载均衡或性能 benchmark。

## 7. 保存预测后，分别评价与汇总

两种评价建议顺序执行，避免同一 GPU 同时运行多个重评价进程。每套评价内部可按相同 shard 使用双卡。

### 官方原始指标，缓存回放

```bash
python -m "$MOD" official --official-root "$OFFICIAL" --output-dir "$OUT" \
  --name v1 --device cuda:0 --shard 0/2 > "${OUT}_logs/official_gpu0.log" 2>&1 &
p0=$!
python -m "$MOD" official --official-root "$OFFICIAL" --output-dir "$OUT" \
  --name v1 --device cuda:1 --shard 1/2 > "${OUT}_logs/official_gpu1.log" 2>&1 &
p1=$!
s0=0; wait "$p0" || s0=$?
s1=0; wait "$p1" || s1=$?
test "$s0" -eq 0 && test "$s1" -eq 0
python -m "$MOD" summarize --output-dir "$OUT" --kind official --name v1
```

如果需要直接运行**完全未包装的原始入口**作交叉核验，下列单类命令重新执行模型前向；上游入口会删除同名日志目录，因此使用独立新目录，不指向适配结果目录：

```bash
NATIVE_LOG="${OUT}_native_bottle_${STAMP}"
test ! -e "$NATIVE_LOG"
mkdir -p "$NATIVE_LOG"
(
  cd "$OFFICIAL"
  python eval.py --gpu_id 0 --num_workers 0 --bs 1 --T 100 --category bottle \
    --mvtec_path "${MVTEC_ROOT%/}/" --checkpoint_path "$WEIGHTS" \
    --base_model_name DeSTSeg_MVTec_5000_ --log_path "$NATIVE_LOG"
) > "$NATIVE_LOG/official_stdout.log" 2>&1
```

原始入口只接受官方前后缀命名；使用自定义 weight-map 时以上命令需按实际文件另行安排对应目录，不能据此误读其他权重。缓存回放保存未舍入 scalar，可与原始 TensorBoard scalar 对照；stdout 只有 4 位小数。

### 统一复评

```bash
python -m "$MOD" unified --dataset-root "$MVTEC_ROOT" --output-dir "$OUT" \
  --name v1 --device cuda:0 --shard 0/2 > "${OUT}_logs/unified_gpu0.log" 2>&1 &
p0=$!
python -m "$MOD" unified --dataset-root "$MVTEC_ROOT" --output-dir "$OUT" \
  --name v1 --device cuda:1 --shard 1/2 > "${OUT}_logs/unified_gpu1.log" 2>&1 &
p1=$!
s0=0; wait "$p0" || s0=$?
s1=0; wait "$p1" || s1=$?
test "$s0" -eq 0 && test "$s1" -eq 0
python -m "$MOD" summarize --output-dir "$OUT" --kind unified --name v1
```

后续重评用新 `--name v2`。统一 GT 已缓存后可不传 `--dataset-root`；官方重评本来就完全读取缓存。每次只保留一个类别的评价输入，完成后释放，再处理下一类。汇总要求所有指定类别完成，默认全部 15 类，并核验 1725 张（正常 467，异常 1258）。子集宏平均明确标注不是全数据集结果。

## 8. 输出结构和记录含义

```text
$OUT/
  <category>/
    predict/
      run.json, arguments.json, environment.json, state.json, complete.json
      weight_inventory.csv, checkpoint_load.json, data_counts.csv
      samples.json, sample_scores.csv, label_audit.csv
      maps/000000.npz ...
      resources.csv, nvidia_smi_samples.jsonl
      official_path_comparison.json
    ground_truth/maps/000000.npz ...    # 首次统一复评创建
    official/v1/
      official.log, official_results.csv, result.json
      arguments.json, environment.json, resources.csv, complete.json ...
    unified/v1/
      category_metrics.csv, fixed_fpr.csv, mask_audit.csv, result.json
      arguments.json, environment.json, resources.csv, complete.json ...
  summaries/v1/
    official/official_category_metrics.csv, official_macro_metrics.csv, complete.json
    unified/category_metrics.csv, macro_metrics.csv, fixed_fpr.csv, fixed_fpr_macro.csv, complete.json
```

`run.json` 中两套 evaluation 的 `not_run` 是**推理完成时的初始快照**；后续状态以各评价目录 `complete.json` / `failure.json` 为准，不回写共享状态。缺少 complete 不代表成功。官方未运行就没有其结果，不用统一结果填充。正常运行记录 CUDA 逻辑设备名称、型号、峰值 allocated/reserved 与可获取的 nvidia-smi 物理 UUID 显存采样。采样值不是精确峰值；不是专门性能 benchmark，也不把两张卡的不同时间峰值相加。阶段耗时包括其对应加载/保存或评价工作，不声称是纯网络 FPS。

模型权重和浮点结果可能较大，本轮不新增大型文件哈希闭包，也不自动删除旧数据。需要重跑时选择新的 OUT/SMOKE/CHECK 或评价名，保留既有实验。
