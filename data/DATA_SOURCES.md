# 数据来源与许可

本项目用了两类数据：**合成数据**（业务场景）和 **公开标注语料**（真实语言分布）。
两类数据的角色不同，不能混用，也不该混报指标。

---

## 一、公开标注语料（真实数据）

### 1. ChnSentiCorp — 谭松波中文酒店评论语料

| 项 | 内容 |
|---|---|
| 规模 | 7,766 条（本项清洗后 7,758 条） |
| 任务 | 情感二分类（正面 / 负面） |
| 内容 | 酒店住客评论，平均 129 字，含大量"优点+缺点"混合表达 |
| 原始出处 | 谭松波（中科院计算所）构建，2008 |
| 获取渠道 | [HuggingFace: dirtycomputer/ChnSentiCorp_htl_all](https://hf-mirror.com/datasets/dirtycomputer/ChnSentiCorp_htl_all) |
| 引用 | 谭松波, 王素格, 等. 中文情感挖掘语料 ChnSentiCorp. 中国计算机学会, 2008 |
| 许可 | 学术研究用途。**商用前需自行确认授权** |

**已观察到的噪声**：存在少数标注与文本矛盾的情况。例如第一条"正面"样本：

> 距离川沙公路较近,但是公交指示不对,如果是"蔡陆线"的话,会非常麻烦.建议用别的路线.房间较为简单.

通篇是负面描述却标为正面。这类噪声约占个位数百分比，属于公开语料的常态，
评估时需要在结论里说明，而不是当模型错误。

### 2. weibo_senti_100k — 微博情感语料

| 项 | 内容 |
|---|---|
| 规模 | 119,988 条（本项抽样使用 37,172 条） |
| 任务 | 情感二分类，正负各半 |
| 内容 | 微博口语表达，平均 66 字，含 `[哈哈]` 类表情占位符 |
| 获取渠道 | [HuggingFace: dirtycomputer/weibo_senti_100k](https://hf-mirror.com/datasets/dirtycomputer/weibo_senti_100k) |
| 原始出处 | [SophonPlus/ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus) |
| 许可 | 该仓库明示 MIT；数据本身为微博公开内容，**商用需自行评估合规性** |

**预处理**：去掉 `[表情]` 占位符（如 `[哈哈][爱你]`），这类 token 对情感没有稳定含义，
保留反而会让模型学到表情符号的分布而非语言本身。

---

## 二、合成数据（业务场景）

| 项 | 内容 |
|---|---|
| 规模 | 900 会话 / 2,232 条客户消息 |
| 任务 | 意图四分类 + 情感三分类（含**中性**） |
| 生成方式 | `src/build_dataset.py`，固定种子 42 |
| 标注来源 | 完全由模板生成，标签即生成参数 |

### 为什么需要合成数据

公开语料**没有**覆盖业务真正需要的两件事：

1. **意图标注**：ChnSentiCorp / weibo 只有情感极性，没有"咨询/投诉/退款/物流"的意图标签
2. **中性类**：两个公开语料都是二分类，没有"中性"这个类别

### 合成数据的已知缺陷（实测）

用纯合成数据训出的模型，在真实 ChnSentiCorp 上只有 **79.3%** 准确率，
且**真实好评被判负面的错误率达 25%**。原因是合成正面样本基本是
"谢谢/好评/很满意"这类**显式感谢语**，而真实好评是
"环境安静、早餐丰富、位置方便"这类**客观描述**。

这正是引入公开语料的原因。

---

## 三、两类数据的使用边界

| | 合成数据 | 公开语料 |
|---|---|---|
| 参与**意图**训练 | ✅ | ❌（无意图标注） |
| 参与**情感**训练 | ✅ | ✅ |
| 参与**测试集** | ✅（test split） | ❌（用户不重叠，见下） |
| 报告指标口径 | 业务场景内 | 真实分布外 |

**关于泄漏**：公开语料与本项目测试集来源完全不同（酒店评论/微博 vs 客服对话），
不存在句子级重叠。训练时公开语料只进 train，不进 test。

**关于指标口径**：报告里会分别给出
- 客服场景测试集（合成分布）
- 分布外手写测试集（`data/test_hard.jsonl`）
- 真实公开语料（ChnSentiCorp，二分类）

三个口径不能混着报，尤其不能只报最高的那个。

---

## 四、重新获取数据

```bash
cd customer-voice/data/external
BASE=https://hf-mirror.com/datasets

curl -L -o chnsenticorp_htl.csv \
  "$BASE/dirtycomputer/ChnSentiCorp_htl_all/resolve/main/ChnSentiCorp_htl_all.csv"

curl -L -o weibo_senti_100k.csv \
  "$BASE/dirtycomputer/weibo_senti_100k/resolve/main/weibo_senti_100k.csv"

# 清洗并转成统一格式
cd ../..
PYTHONPATH=src python3 src/import_external.py --weibo-limit 40000
```

> 注：`datasets` 5.x 已不支持加载 ChnSentiCorp 的脚本格式，
> 所以这里直接下载 CSV，不依赖 `datasets` 库。