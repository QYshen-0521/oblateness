#!/usr/bin/env bash
# preCompact: 在上下文压缩前提示用户让 Agent 读取根目录 worklog.md（无法代 Agent 自动 @ 文件）。
set -euo pipefail
exec python3 -c '
import json
import sys

try:
    data = json.load(sys.stdin)
except json.JSONDecodeError:
    data = {}

pct = data.get("context_usage_percent")
n_compact = data.get("messages_to_compact")
parts = ["【即将压缩上下文】"]
if pct is not None:
    parts.append(f"当前约 {pct}% 占用。")
if n_compact is not None:
    parts.append(f"将摘要约 {n_compact} 条消息。")
parts.append(
    "请在压缩后的下一条消息中 @worklog.md，"
    "或要求 Agent 先用 Read 读取仓库根目录 worklog.md，"
    "再依据 Task Board / Session Logs 继续（Planner/Executor 专用 Chat 尤须如此）。"
)
msg = "".join(parts)
print(json.dumps({"user_message": msg}, ensure_ascii=False))
'
