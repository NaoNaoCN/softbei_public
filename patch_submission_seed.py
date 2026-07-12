"""
patch_submission_seed.py
提交快照后处理脚本（幂等）。把「知识库 auto-index 只写 document_chunk」的结果补齐成
「本地上传文档」的等价效果，使教材《动手学深度学习》在资源库中显示为 doc 资源、
可预览 PDF、可构建图谱。

做两件事，直接改写 submission_seed.sql：
  1. document_chunk 块内，逐字段把 chunk_id / doc_id / parent_chunk_id 的前缀
     `d2l-zh` 改成 `doc_d2lzh`（满足前端 `kp_id LIKE 'doc_%'` 的展示条件，
     并与 KG 构建按 doc_id 拉块对齐）。绝不触碰 text 列（个别行正文含
     GitHub 链接 d2l-ai/d2l-zh，全局替换会损坏正文）。
  2. resource_meta 块插入 1 行 demo 账号的 doc 记录，content 含 `d2l-zh.pdf`
     以便前端预览接口按文件名到 uploaded_docs/ 读取。

用法：
    python patch_submission_seed.py                 # 就地打补丁
    python patch_submission_seed.py --check          # 只检查是否已打过补丁，不改写

配套步骤（本脚本不做，见「评委导入与测试说明.md」第五节）：
    - 复制 knowledge_base/深度学习/d2l-zh.pdf 到 uploaded_docs/d2l-zh.pdf
"""

from __future__ import annotations

import argparse
import sys

SEED_FILE = "submission_seed.sql"

OLD_PREFIX = "d2l-zh"
NEW_PREFIX = "doc_d2lzh"

# resource_meta 列顺序：id user_id kp_id resource_type title content content_json created_at
DEMO_USER_ID = "333556670864809984"
DOC_RESOURCE_ID = "333557000000000001"   # 新雪花段，远离现有 id，避免主键冲突
DOC_KP_ID = NEW_PREFIX
DOC_TITLE = "动手学深度学习"
PDF_FILENAME = "d2l-zh.pdf"
CHUNK_TOTAL = 4561                        # document_chunk 总行数（父+子）
DOC_CREATED_AT = "2026-07-09 10:50:00"    # 晚于 chunk 索引时间

CHUNK_COPY_HEADER = "COPY public.document_chunk (id, chunk_id, doc_id, collection_name, text, source, page, section, user_id, created_at, embedding, metadata, parent_chunk_id, is_parent, text_search) FROM stdin;"
RM_COPY_HEADER = "COPY public.resource_meta (id, user_id, kp_id, resource_type, title, content, content_json, created_at) FROM stdin;"
TERM = "\\."

# document_chunk 列索引（0-based，按 CHUNK_COPY_HEADER 顺序）
COL_CHUNK_ID = 1
COL_DOC_ID = 2
COL_PARENT_CHUNK_ID = 12


def _retighten_prefix(field: str) -> str:
    """把以 OLD_PREFIX 开头的标识符字段前缀替换为 NEW_PREFIX。空/\\N 原样返回。"""
    if field == "\\N" or field == "":
        return field
    if field == OLD_PREFIX:
        return NEW_PREFIX
    if field.startswith(OLD_PREFIX + "_"):
        return NEW_PREFIX + field[len(OLD_PREFIX):]
    return field


def _find_block(lines: list[str], header: str) -> tuple[int, int]:
    """返回 (header_idx, term_idx)。term_idx 为该 COPY 块的 \\. 所在行下标。"""
    try:
        h = lines.index(header)
    except ValueError:
        raise SystemExit(f"[patch] 未找到 COPY 头：{header[:60]}...")
    t = h + 1
    while t < len(lines) and lines[t] != TERM:
        t += 1
    if t >= len(lines):
        raise SystemExit("[patch] 未找到该 COPY 块的终止符 \\.")
    return h, t


def patch_chunks(lines: list[str]) -> int:
    """逐字段改写 document_chunk 块的三列前缀，返回改动的行数。"""
    h, t = _find_block(lines, CHUNK_COPY_HEADER)
    changed = 0
    for i in range(h + 1, t):
        row = lines[i]
        cols = row.split("\t")
        if len(cols) != 15:
            raise SystemExit(f"[patch] document_chunk 第 {i+1} 行列数异常：{len(cols)}（应为 15）")
        before = (cols[COL_CHUNK_ID], cols[COL_DOC_ID], cols[COL_PARENT_CHUNK_ID])
        cols[COL_CHUNK_ID] = _retighten_prefix(cols[COL_CHUNK_ID])
        cols[COL_DOC_ID] = _retighten_prefix(cols[COL_DOC_ID])
        cols[COL_PARENT_CHUNK_ID] = _retighten_prefix(cols[COL_PARENT_CHUNK_ID])
        after = (cols[COL_CHUNK_ID], cols[COL_DOC_ID], cols[COL_PARENT_CHUNK_ID])
        if before != after:
            lines[i] = "\t".join(cols)
            changed += 1
    return changed


def patch_resource_meta(lines: list[str]) -> bool:
    """在 resource_meta 块插入 demo 的 doc 记录。已存在则跳过。返回是否插入。"""
    h, t = _find_block(lines, RM_COPY_HEADER)
    # 幂等：块内已有该 doc 记录则不重复插入
    for i in range(h + 1, t):
        if lines[i].startswith(DOC_RESOURCE_ID + "\t") or (
            "\t" + DOC_KP_ID + "\tdoc\t" in lines[i]
        ):
            return False
    content = f"已导入文档：{PDF_FILENAME}，共 {CHUNK_TOTAL} 个文本块"
    row = "\t".join([
        DOC_RESOURCE_ID, DEMO_USER_ID, DOC_KP_ID, "doc",
        DOC_TITLE, content, "\\N", DOC_CREATED_AT,
    ])
    lines.insert(t, row)   # 插到 \. 之前
    return True


def already_patched(text: str) -> bool:
    return (NEW_PREFIX + "_") in text or ("\t" + DOC_KP_ID + "\tdoc\t") in text


def main() -> None:
    parser = argparse.ArgumentParser(description="提交快照后处理补丁（幂等）")
    parser.add_argument("--check", action="store_true", help="只检查是否已打补丁")
    parser.add_argument("--file", default=SEED_FILE, help="快照文件路径")
    args = parser.parse_args()

    with open(args.file, encoding="utf-8", errors="strict", newline="") as f:
        raw = f.read()
    # 保留行尾风格：统一按 \n 切分（pg_dump 输出为 \n）
    lines = raw.split("\n")
    trailing_newline = raw.endswith("\n")
    if trailing_newline:
        lines = lines[:-1]

    if args.check:
        print("已打补丁" if already_patched(raw) else "未打补丁")
        return

    if already_patched(raw):
        print("[patch] 检测到已打过补丁，跳过（幂等）。")
        return

    n_chunks = patch_chunks(lines)
    inserted = patch_resource_meta(lines)

    out = "\n".join(lines) + ("\n" if trailing_newline else "")
    with open(args.file, "w", encoding="utf-8", newline="") as f:
        f.write(out)

    print(f"[patch] document_chunk 改写 {n_chunks} 行前缀 {OLD_PREFIX} -> {NEW_PREFIX}")
    print(f"[patch] resource_meta {'已插入 1 行 doc 记录' if inserted else '记录已存在，未插入'}")
    print(f"[patch] doc kp_id={DOC_KP_ID}  title={DOC_TITLE}  pdf={PDF_FILENAME}")
    print("[patch] 完成。请确保已复制 PDF 到 uploaded_docs/d2l-zh.pdf")


if __name__ == "__main__":
    main()
