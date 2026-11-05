"""数据与任务 schema：标签体系、字段约定。

这个文件是所有模块的"单一事实来源"：
- 分类任务 / 情感任务分别有哪些标签
- 数据库表结构的字段名
后续 build_dataset / model / web 都从这里 import，避免“同一个标签三处写错”。
"""

from dataclasses import dataclass, field
from typing import Final

# ---------------------------------------------------------------- 意图分类
# 3~5 类的业务场景。选这几类是因为它们覆盖了电商客服 80% 的真实工单。
INTENT_LABELS: Final[dict[str, str]] = {
    "咨询": "商品/政策/物流等信息询问，尚未实际发生交易纠纷",
    "投诉": "对服务或商品表达不满，要求处理/道歉/补偿",
    "退款": "明确的退款、退货、换货诉求",
    "物流": "包裹时效、丢件、错发、地址修改等物流问题",
}
INTENT_NAMES: Final[list[str]] = list(INTENT_LABELS.keys())

# 情感用 3 档而非 5 档：客服场景下“中性/无感”和“强烈情绪”的区分更稳定
SENTIMENT_LABELS: Final[dict[str, str]] = {
    "负面": "生气、失望、催促、抱怨、质疑",
    "中性": "客观陈述、咨询、无明确情绪色彩",
    "正面": "感谢、认可、满意、表扬",
}
SENTIMENT_NAMES: Final[list[str]] = list(SENTIMENT_LABELS.keys())


@dataclass(frozen=True)
class Message:
    """单条对话消息（用户或客服）。"""

    session_id: str
    turn: int          # 会话内轮次，从 0 开始
    role: str          # "customer" | "agent"
    text: str
    intent: str = ""      # 仅 customer 标注；agent 为空
    sentiment: str = ""   # 同上


@dataclass
class Session:
    """一次完整会话。"""

    session_id: str
    channel: str          # "text" | "image"(OCR) | "review"
    messages: list[Message] = field(default_factory=list)

    @property
    def customer_texts(self) -> list[str]:
        return [m.text for m in self.messages if m.role == "customer"]


# ---------------------------------------------------------------- 难度梯度
# 硬样本是这套数据的核心价值：简单句式谁都能分类，真正难的是
# “礼貌但是投诉”“提问但是催促”“中性措辞里藏着负面”。
HARD_CASES: Final[dict[str, list[str]]] = {
    "礼貌型投诉": ["您这边服务真是让我大开眼界", "麻烦问一下这个质量你们自己用吗"],
    "中性措辞带负面": ["好的我收到了，那大概什么时候能解决"],
    "咨询与催单混淆": ["问下这个订单还发不发，已经等了三天了"],
    "反问式": ["这就是你们说的次日达？", "这也能叫正品？"],
}