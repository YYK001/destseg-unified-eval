# DeSTSeg：BTAD 三类正常样本训练与复评

这是**官方 DeSTSeg 方法在 BTAD 上的训练适配**，不是作者发布的 BTAD 预训练模型或官方 BTAD benchmark。本入口不使用 MVTec 预训练 checkpoint。与已有 MVTec 预训练复评共存；本轮不接入 VisA。

## 冻结方案

| 项目 | 设置 |
|---|---|
| 官方源码 | `f168ba576e0a917ae5e037b677e7d37b741ea3d8`，上游 tracked 文件不修改 |
| 模型 | 原始 `DeSTSeg(dest=True, ed=True)`；ImageNet ResNet18 teacher，随机 student/segmentation |
| 数据 | BTAD 01/02/03 各自完整 train/ok，400/399/1000张；无校准划分 |
| 合成异常 | 官方 MVTecDataset 训练 `__getitem__` 与原始 DTD＋Perlin，aug_prob=1.0 |
| 增强选择 | 所有 BTAD 类别均不旋转；这是预先固定的适配选择，不声称是官方 BTAD 默认值 |
| 输入 | RGB PIL bilinear整图256，官方 ImageNet normalization |
| 训练 | 每类5000步：前1000步更新学生，后4000步更新分割网络 |
| 优化器 | 官方SGD，momentum0.9、weight_decay1e-4；学生lr0.4，分割res/head为0.1/0.01 |
| 损失 | 原始cosine_similarity_loss（保留batch求和）、focal_loss(gamma4)＋l1_loss |
| 数值 | FP32，不加AMP、梯度累积或额外冻结逻辑 |
| batch | 默认32；显式修改会记入协议，不自动降batch，不声称减batch与默认等价；禁止batch1 |
| seed/worker | seed42，默认每进程2worker，初始化Python/NumPy/Torch/imgaug及官方全局augmenter随机源 |
| 选择 | 固定最终第5000步；训练期间不读取测试图/GT、不评价、不按测试表现选checkpoint |

与原 train.py 相比，训练循环仅改为显式两阶段、正常训练清单适配、日志和资源记录；保持同一DataLoader迭代器跨越1000步边界。官方每1000步测试评价被移到训练后，避免混淆最终模型选择与测试集使用。两个阶段分别切换 student/segmentation 的 train/eval，并只step各自优化器；不额外把第二阶段student参数requires_grad设为False。Teacher行为和损失归约不改。

官方数据集原来只glob PNG；这里只替换 `mvtec_paths` 为已有BTAD文件列表，包含BMP，直接执行原始训练 `__getitem__`，不重写网络、损失或Perlin增强。

## 双 T4 与存储

每卡独立训练一类：GPU0先01再03，GPU1训练02，无DDP。每类最终模型约134MiB，另外保留学生阶段1000步快照；没有预留全数据GPU缓存。默认batch32是否适合当前T4、CPU是否拖慢增强，需要短训练实测。训练smoke必须覆盖两个阶段，不能只测推理或学生前几步。

`--smoke-steps 20` 实际运行20步学生＋20步分割，产物标记smoke，仅用于功能/资源检查，不能续当5000步正式模型。正式训练从新初始化开始。当前不提供恢复训练：1000步快照只含模型参数，不含SGD动量/RNG/数据位置；中断必须使用新目录重新训练，不能伪称精确续训。后续若需要恢复功能，应单独实现与验证。

训练输出含run/environment/arguments、训练和DTD清单、逐步losses.csv、分阶段resources.csv、显存原始采样、checkpoint.json及complete/failure。样本不保存到仓库；不覆盖已有阶段目录。

## Kaggle 准备与 smoke

使用已验证的 `/kaggle/working/destseg-venv/bin/python`（Python3.10/torch2.0.0+cu118），不向Notebook默认Python安装旧依赖。先同步Git仓库，然后把BTAD数据集挂载到Kaggle。

DTD 官方来源为 <https://www.robots.ox.ac.uk/~vgg/data/dtd/>，官方下载脚本使用 `dtd-r1.0.1.tar.gz`，47种纹理共5640张。可直接挂载现有DTD，`--dtd-root`必须指向包含47个纹理子目录的`images`，也可在Kaggle下载。下列准备命令不会启动训练、不会下载DeSTSeg权重：

```bash
cd /kaggle/working/destseg-unified-eval
git pull --ff-only
git submodule update --init --recursive
PY=/kaggle/working/destseg-venv/bin/python
$PY -m external_baselines.destseg_btad.prepare_dtd --directory /kaggle/working/destseg_assets
$PY -m unittest discover -s external_baselines/destseg_btad/tests -v
$PY -m unittest discover -s external_baselines/destseg_mvtec_pretrained/tests -v
```

不在本地电脑下载数据/权重或训练。Kaggle已有DTD时直接使用挂载目录，无需上述下载。

路径参数化（以下命令是Bash；Notebook也可通过subprocess列表传参）：

```bash
PY=/kaggle/working/destseg-venv/bin/python
MOD=external_baselines.destseg_btad
BTAD_ROOT=/kaggle/input/datasets/thtuan/btad-beantech-anomaly-detection/BTech_Dataset_transformed
DTD_ROOT=/kaggle/working/destseg_assets/dtd/images
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SMOKE=/kaggle/working/destseg_btad_smoke_$STAMP
TRAIN=/kaggle/working/destseg_btad_train_$STAMP
OUT=/kaggle/working/destseg_btad_eval_$STAMP

$PY -u -m "$MOD" train --dataset-root "$BTAD_ROOT" --dtd-root "$DTD_ROOT" \
  --output-dir "$SMOKE" --categories 01 --device cuda:0 --batch-size 32 --workers 2 --smoke-steps 20
```

检查`01/train/resources.csv`中student_train和segmentation_train的显存与耗时、loss是否有限以及complete.json。smoke步耗时含首次加载/预热，不是严格性能benchmark。若OOM停止后再讨论显式batch调整，不自动改变协议。

可进一步使用smoke模型做两张测试图的保存链路检查（不会产生正式指标）：

```bash
$PY -u -m "$MOD" infer --dataset-root "$BTAD_ROOT" --training-dir "$SMOKE" \
  --output-dir "${SMOKE}_predict" --categories 01 --device cuda:0 --limit 2
```

## 正式训练（smoke通过后手工运行）

```bash
mkdir -p "${TRAIN}_logs"
$PY -u -m "$MOD" train --dataset-root "$BTAD_ROOT" --dtd-root "$DTD_ROOT" \
  --output-dir "$TRAIN" --categories 01 03 --device cuda:0 --batch-size 32 --workers 2 \
  > "${TRAIN}_logs/gpu0.log" 2>&1 &
p0=$!
$PY -u -m "$MOD" train --dataset-root "$BTAD_ROOT" --dtd-root "$DTD_ROOT" \
  --output-dir "$TRAIN" --categories 02 --device cuda:1 --batch-size 32 --workers 2 \
  > "${TRAIN}_logs/gpu1.log" 2>&1 &
p1=$!
s0=0; wait "$p0" || s0=$?
s1=0; wait "$p1" || s1=$?
test "$s0" -eq 0 && test "$s1" -eq 0
```

每50步和阶段首尾打印loss/平均步耗时，另一个终端可`tail -f`上述日志。只使用各类train/DeSTSeg_BTAD_5000_<类>.pckl；不使用student_step1000或smoke模型评价正式结果。推理严格加载并验证训练协议与完成状态。

## 推理和两套评价

保持主分支sigmoid后256图和官方Top-100分数定义。统一全图H//4、W//4、快速CUDA AUPRO、AP、固定FPR及宏平均复用原有函数，不引入新指标或连通域实现。

BTAD mask可能使用0/1或0/255编码，先按原语义`>0`转成0/255，再应用官方mask预处理（ToTensor、双线性antialias、>=0.5）。这是明确的BTAD输入适配，不是修改官方指标。已知BTAD包含异常标签但空mask的样本，保留记录并分别评价：统一图像标签沿用test/ko，官方指标实现用处理后mask派生标签，差异写入label_audit.csv，不删除或重标统一标签。

推理使用独立输出目录，训练产物保留在TRAIN中。每阶段两个独立进程可按相同01/03和02分配；以下简单顺序命令也可直接运行，不会占用两张卡的合并显存：

```bash
$PY -u -m "$MOD" infer --dataset-root "$BTAD_ROOT" --training-dir "$TRAIN" \
  --output-dir "$OUT" --categories all --device cuda:0
$PY -u -m "$MOD" unified --dataset-root "$BTAD_ROOT" --output-dir "$OUT" --device cuda:0 --name v1
$PY -m "$MOD" summarize --output-dir "$OUT" --kind unified --name v1
$PY -u -m "$MOD" official --output-dir "$OUT" --device cuda:0 --name v1
$PY -m "$MOD" summarize --output-dir "$OUT" --kind official --name v1
```

默认汇总验证全部741张（正常451、异常290）、三类完整，禁止混合不同batch/seed协议或smoke预测。两套结果分目录；`official`意为**官方指标实现应用于BTAD适配数据**，不是官方已发布BTAD结果。IAP/IAP90仍是实例指标，主/辅助分支分别记录。

## 本地验证状态

本地无torch/numpy环境。标准库检查覆盖协议、1000/4000与smoke阶段长度、batch1拒绝、类别集合、源码语法及直接复用官方训练方法。两项运行时测试已提供：0/1与0/255 mask语义等价、两阶段优化器及BatchNorm更新隔离；Tiny合成网络只检查训练循环，不证明真实DeSTSeg训练成功。

真实训练显存、速度、收敛、BTAD预测和指标全部待Kaggle验证。MVTec已完成的9项运行环境测试和正式结果不作为BTAD训练通过的证明。GitHub只放代码文档，DTD、训练权重、预测、日志和报告都不入库。
