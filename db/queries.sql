-- =============================================================
-- 看板与报告用到的核心 SQL
-- 每条都标注了：用到什么 SQL 知识点、为什么这样写
-- =============================================================

-- ------------------------------------------------------------
-- Q1  会话级汇总：每个会话有几条客户消息、几条负面
-- 知识点：JOIN + 条件聚合（COUNT + CASE WHEN）+ GROUP BY
-- 为什么：看板首屏的会话列表就是这条
-- ------------------------------------------------------------
SELECT
    s.session_id,
    s.channel,
    COUNT(m.msg_id)                                        AS total_msgs,
    SUM(CASE WHEN m.role = 'customer' THEN 1 ELSE 0 END)    AS customer_msgs,
    SUM(CASE WHEN m.sentiment = '负面'   THEN 1 ELSE 0 END) AS negative_msgs
FROM sessions s
LEFT JOIN messages m ON m.session_id = s.session_id
GROUP BY s.session_id, s.channel
ORDER BY negative_msgs DESC, s.session_id
LIMIT 20;


-- ------------------------------------------------------------
-- Q2  意图 × 情感 交叉表
-- 知识点：GROUP BY 两个维度；这是给前端热力图直接喂数的格式
-- ------------------------------------------------------------
SELECT
    m.intent,
    m.sentiment,
    COUNT(*) AS n
FROM messages m
WHERE m.role = 'customer'
GROUP BY m.intent, m.sentiment
ORDER BY m.intent, m.sentiment;


-- ------------------------------------------------------------
-- Q3  高风险会话：负面消息占比 >= 50% 且客户消息 >= 2
-- 知识点：GROUP BY + HAVING 过滤聚合值 + 浮点除法
-- 为什么：这对应业务动作——“优先派人工客服”
-- ------------------------------------------------------------
SELECT
    s.session_id,
    SUM(CASE WHEN m.sentiment = '负面' THEN 1 ELSE 0 END) * 1.0
        / COUNT(m.msg_id) AS negative_ratio,
    COUNT(m.msg_id) AS customer_msgs
FROM sessions s
JOIN messages m ON m.session_id = s.session_id AND m.role = 'customer'
GROUP BY s.session_id
HAVING negative_ratio >= 0.5 AND customer_msgs >= 2
ORDER BY negative_ratio DESC, customer_msgs DESC;


-- ------------------------------------------------------------
-- Q4  模型对比：同一批消息，两个模型版本的意图预测在哪些上好/差
-- 知识点：同一张表自连接（self JOIN）+ 别名，用来并排比较两个模型
-- 为什么：这是“微调 vs baseline”对比的数据库实现
--       版本名按 evaluations 里的 macro_f1 动态取，不写死
-- ------------------------------------------------------------
WITH base AS (
    SELECT model_version FROM evaluations
    WHERE task='intent' AND metric='macro_f1'
    ORDER BY value ASC LIMIT 1        -- 最弱的当 baseline
), tuned AS (
    SELECT model_version FROM evaluations
    WHERE task='intent' AND metric='macro_f1'
    ORDER BY value DESC LIMIT 1       -- 最强的当 tuned
)
SELECT
    p_base.msg_id,
    substr(m.text, 1, 40) AS text,
    m.intent              AS gold,
    p_base.label          AS pred_baseline,
    p_tuned.label         AS pred_finetuned,
    ROUND(p_tuned.confidence, 3) AS conf_finetuned
FROM predictions p_base
JOIN predictions p_tuned
    ON p_tuned.msg_id = p_base.msg_id
   AND p_tuned.task   = p_base.task
   AND p_tuned.model_version = (SELECT model_version FROM tuned)
JOIN messages m ON m.msg_id = p_base.msg_id
WHERE p_base.model_version = (SELECT model_version FROM base)
  AND p_base.task = 'intent'
  AND m.role = 'customer'
  AND p_base.label <> m.intent      -- baseline 错
  AND p_tuned.label = m.intent      -- 微调对
ORDER BY conf_finetuned DESC
LIMIT 30;


-- ------------------------------------------------------------
-- Q5  按意图统计平均置信度，找出模型最"犹豫"的类别
-- 知识点：AVG + 窗口函数 RANK() OVER
-- 为什么：置信度低的类别就是下一步该补数据的类别
-- ------------------------------------------------------------
SELECT
    intent,
    ROUND(AVG(confidence), 4) AS avg_conf,
    COUNT(*)                  AS n,
    RANK() OVER (ORDER BY AVG(confidence) ASC) AS least_confident_rank
FROM predictions p
JOIN messages m ON m.msg_id = p.msg_id
WHERE p.task = 'intent'
GROUP BY intent;


-- ------------------------------------------------------------
-- Q6  每个会话最后一次客户发言（用窗口函数取"最新一条"）
-- 知识点：ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...)
-- 为什么：客服接手时最需要看的是"客户最后一句说了什么"
-- ------------------------------------------------------------
SELECT session_id, turn, text, intent, sentiment
FROM (
    SELECT
        m.*,
        ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY turn DESC) AS rn
    FROM messages m
    WHERE role = 'customer'
) t
WHERE rn = 1
ORDER BY session_id;


-- ------------------------------------------------------------
-- Q7  使用索引 vs 不使用的执行计划对比（面试可讲）
-- 知识点：EXPLAIN QUERY PLAN
-- 把下面 INDEXED 设为 verify，再删除 idx_messages_session 做对比
-- ------------------------------------------------------------
EXPLAIN QUERY PLAN
SELECT msg_id, text FROM messages WHERE session_id = 'S0001';


-- ------------------------------------------------------------
-- Q8  事务示例：一次写入"预测 + 评测指标"，要么都成功要么都回滚
-- 知识点：BEGIN / COMMIT / ROLLBACK
-- ------------------------------------------------------------
-- BEGIN;
-- INSERT INTO predictions (...) VALUES (...);
-- INSERT INTO evaluations (...) VALUES (...);
-- COMMIT;