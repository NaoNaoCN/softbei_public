"""文档文件清理服务：定期清理 uploaded_docs/ 目录中的过期文件。

清理规则：
1. 已索引文件：超过 retention_days 天后删除
2. 孤儿文件（导入失败/中断）：超过 orphan_retention_days 天后删除
3. 处理中保护：mtime < 5 分钟前的文件跳过
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger

from backend.config import config


UPLOAD_DIR = Path(__file__).parent.parent.parent / config.storage.upload_dir

# upload 文件名格式：{uuid_hex_N}_{original_name}
UPLOAD_NAME_PATTERN = re.compile(rf"^[a-f0-9]{{{config.storage.doc_id_hex_length}}}_.+")


def _resolve_safe_path(filename: str) -> Optional[Path]:
    """取纯文件名（防路径穿越），返回 uploaded_docs/ 下的完整路径。"""
    safe_name = Path(filename).name
    if safe_name != filename or ".." in filename:
        return None
    return UPLOAD_DIR / safe_name


def _extract_original_name(filename: str) -> str:
    """从 upload 文件名中提取原始文件名。

    "aa77bc5f96cf_d2l-zh.pdf" → "d2l-zh.pdf"
    "a1b2c3d4e5f6_my doc.docx" → "my doc.docx"
    """
    m = re.match(r"^[a-f0-9]{12}_(.+)", filename)
    return m.group(1) if m else filename


async def cleanup_uploaded_docs(
    retention_days: int | None = None,
    orphan_retention_days: int | None = None,
    dry_run: bool = False,
) -> dict:
    """
    清理 uploaded_docs/ 目录中的过期文件。

    规则：
    1. 文件有对应 ResourceMeta 记录 + mtime > retention_days → 删除
    2. 文件无对应 ResourceMeta 记录 + mtime > orphan_retention_days → 删除
    3. 文件 mtime < MIN_FILE_AGE_SECONDS → 跳过（处理中保护）

    :param retention_days:        已索引文件的保留天数
    :param orphan_retention_days: 孤儿文件的保留天数
    :param dry_run:               True 时仅预览不删除
    :return:                      清理结果摘要
    """
    from backend.db.crud import select
    from backend.db.database import _session_factory
    from backend.db.models import ResourceMeta

    _retention_days = retention_days if retention_days is not None else config.storage.cleanup.retention_days
    _orphan_retention_days = orphan_retention_days if orphan_retention_days is not None else config.storage.cleanup.orphan_retention_days

    now = time.time()
    indexed_cutoff = now - _retention_days * 86400
    orphan_cutoff = now - _orphan_retention_days * 86400
    active_cutoff = now - config.storage.cleanup.min_file_age_seconds

    indexed_original_names: set[str] = set()
    try:
        if _session_factory is not None:
            async with _session_factory() as db:
                resources = await select(db, ResourceMeta, filters={"resource_type": "doc"})
                for r in resources:
                    if r.content:
                        # content 格式："已导入文档：filename.pdf，共 N 个文本块"
                        m = re.search(r"已导入(?:文档|PDF|DOCX|Markdown|TXT)：(.+?)[，,]", r.content)
                        if m:
                            indexed_original_names.add(m.group(1))
    except Exception as e:
        logger.warning(f"[Cleanup] 查询 ResourceMeta 失败，将按孤儿策略处理所有文件: {e}")

    if not UPLOAD_DIR.exists():
        logger.info(f"[Cleanup] 目录不存在: {UPLOAD_DIR}")
        return {"scanned": 0, "deleted": 0, "skipped_active": 0,
                "skipped_retained": 0, "errors": [], "details": []}

    files = sorted(
        [f for f in UPLOAD_DIR.iterdir()
         if f.is_file() and f.suffix.lower() in set(config.storage.supported_extensions)],
        key=lambda f: f.stat().st_mtime,
    )

    deleted: list[dict] = []
    skipped_active: list[dict] = []
    skipped_retained: list[dict] = []
    errors: list[dict] = []

    for f in files:
        try:
            file_mtime = f.stat().st_mtime
            file_age_seconds = now - file_mtime
            file_age_days = file_age_seconds / 86400

            # 规则 3：处理中保护
            if file_mtime > active_cutoff:
                skipped_active.append({
                    "file": f.name,
                    "age_seconds": round(file_age_seconds),
                    "reason": "正在处理（mtime < 5 分钟）",
                })
                continue

            # 检查是否为 upload 命名的文件（uuid_original_name）
            if not UPLOAD_NAME_PATTERN.match(f.name):
                skipped_retained.append({
                    "file": f.name,
                    "age_days": round(file_age_days, 1),
                    "reason": "非上传文件命名格式，不归清理管辖",
                })
                continue

            original_name = _extract_original_name(f.name)
            is_indexed = original_name in indexed_original_names

            if is_indexed:
                # 规则 1：已索引文件
                if file_mtime < indexed_cutoff:
                    reason = f"已索引文件过期 ({round(file_age_days, 1)}d > {_retention_days}d)"
                    if not dry_run:
                        f.unlink()
                        logger.info(f"[Cleanup] 删除过期索引文件: {f.name} (age={file_age_days:.1f}d)")
                    deleted.append({"file": f.name, "age_days": round(file_age_days, 1),
                                    "size_bytes": f.stat().st_size, "type": "expired_indexed",
                                    "reason": reason})
                else:
                    skipped_retained.append({
                        "file": f.name,
                        "age_days": round(file_age_days, 1),
                        "reason": f"已索引，保留期内 ({round(file_age_days, 1)}d < {_retention_days}d)",
                    })
            else:
                # 规则 2：孤儿文件
                if file_mtime < orphan_cutoff:
                    reason = f"孤儿文件过期 ({round(file_age_days, 1)}d > {_orphan_retention_days}d)"
                    if not dry_run:
                        f.unlink()
                        logger.info(f"[Cleanup] 删除孤儿文件: {f.name} (age={file_age_days:.1f}d)")
                    deleted.append({"file": f.name, "age_days": round(file_age_days, 1),
                                    "size_bytes": f.stat().st_size, "type": "orphan",
                                    "reason": reason})
                else:
                    skipped_retained.append({
                        "file": f.name,
                        "age_days": round(file_age_days, 1),
                        "reason": f"孤儿文件，保留期内 ({round(file_age_days, 1)}d < {_orphan_retention_days}d)",
                    })

        except Exception as e:
            logger.error(f"[Cleanup] 处理文件 {f.name} 失败: {e}")
            errors.append({"file": f.name, "error": str(e)})

    total_deleted_bytes = sum(d.get("size_bytes", 0) for d in deleted)
    result = {
        "scanned": len(files),
        "deleted": len(deleted),
        "deleted_bytes": total_deleted_bytes,
        "skipped_active": len(skipped_active),
        "skipped_retained": len(skipped_retained),
        "errors": errors,
        "details": {
            "deleted": deleted,
            "skipped_active": skipped_active,
            "skipped_retained": skipped_retained,
        },
    }

    if dry_run:
        logger.info(
            f"[Cleanup] DRY RUN 完成: scanned={len(files)}, would_delete={len(deleted)}, "
            f"would_free_bytes={total_deleted_bytes}, skipped_active={len(skipped_active)}, "
            f"skipped_retained={len(skipped_retained)}, errors={len(errors)}"
        )
    else:
        logger.info(
            f"[Cleanup] 清理完成: scanned={len(files)}, deleted={len(deleted)}, "
            f"freed_bytes={total_deleted_bytes}, skipped_active={len(skipped_active)}, "
            f"skipped_retained={len(skipped_retained)}, errors={len(errors)}"
        )

    return result


_cleanup_config_cache: Optional[dict] = None


def _get_cleanup_config() -> dict:
    """读取清理配置（带缓存，避免每次循环都解析 yaml）。"""
    global _cleanup_config_cache
    if _cleanup_config_cache is not None:
        return _cleanup_config_cache

    try:
        sc = config.storage.cleanup
        _cleanup_config_cache = {
            "enabled": sc.enabled,
            "retention_days": sc.retention_days,
            "orphan_retention_days": sc.orphan_retention_days,
            "interval_hours": sc.interval_hours,
        }
        return _cleanup_config_cache
    except Exception:
        return {
            "enabled": True,
            "retention_days": 30,
            "orphan_retention_days": 7,
            "interval_hours": 24,
        }


async def start_cleanup_task() -> None:
    """启动文档文件清理后台任务，每 24 小时执行一次。"""
    cfg = _get_cleanup_config()
    if not cfg["enabled"]:
        logger.info("[Cleanup] 文档文件自动清理已禁用")
        return

    interval_seconds = cfg["interval_hours"] * 3600
    logger.info(
        f"[Cleanup] 启动文档文件清理后台任务 "
        f"(retention={cfg['retention_days']}d, orphan={cfg['orphan_retention_days']}d, "
        f"每 {cfg['interval_hours']}h 执行一次)"
    )

    while True:
        try:
            await asyncio.sleep(interval_seconds)
            # 每次执行时刷新配置（支持运行时修改配置后重启生效）
            _cleanup_config_cache = None
            cfg = _get_cleanup_config()
            if not cfg["enabled"]:
                continue
            await cleanup_uploaded_docs(
                retention_days=cfg["retention_days"],
                orphan_retention_days=cfg["orphan_retention_days"],
                dry_run=False,
            )
        except asyncio.CancelledError:
            logger.info("[Cleanup] 文档文件清理任务已取消")
            break
        except Exception as e:
            logger.error(f"[Cleanup] 清理任务出错: {e}")
