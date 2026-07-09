"""
backend/rag/loader.py
文档加载器：将本地文件（PDF / DOCX / Markdown / TXT）解析为纯文本块，
并附加结构化元数据，供后续索引使用。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

import pymupdf4llm           # PDF → Markdown
import mammoth               # DOCX → Markdown

from langchain_community.document_loaders import (
    PyPDFLoader,           # PDF 加载（保留用于 load_directory 等场景）
)
from langchain_core.documents import Document

from backend.config import config


# ----------------------------------------------------------
# 数据结构
# ----------------------------------------------------------

@dataclass
class TextChunk:
    """单个文本块，携带来源元数据。"""
    chunk_id: str               # 唯一标识，格式：{doc_id}_{chunk_index}
    text: str                   # 纯文本内容
    doc_id: str                 # 所属文档 ID
    source_path: str            # 原始文件路径
    page: Optional[int] = None  # 页码（PDF）
    section: Optional[str] = None  # 章节标题
    parent_chunk_id: Optional[str] = None  # 父块 chunk_id（子块回指父块）
    is_parent: bool = False     # 自身是否为父块
    metadata: dict = field(default_factory=dict)  # 额外元数据

    def to_langchain_doc(self) -> Document:
        """转换为 LangChain Document。"""
        return Document(
            page_content=self.text,
            metadata={
                "chunk_id": self.chunk_id,
                "doc_id": self.doc_id,
                "source": self.source_path,
                "page": self.page,
                "section": self.section,
                **self.metadata,
            }
        )

    @classmethod
    def from_langchain_doc(cls, doc: Document, doc_id: str, chunk_index: int) -> "TextChunk":
        """从 LangChain Document 转换。"""
        meta = doc.metadata
        return cls(
            chunk_id=f"{doc_id}_{chunk_index}",
            text=doc.page_content,
            doc_id=doc_id,
            source_path=meta.get("source", ""),
            page=meta.get("page"),
            section=meta.get("section"),
            metadata={k: v for k, v in meta.items()
                      if k not in ("chunk_id", "doc_id", "source", "page", "section")},
        )


# ----------------------------------------------------------
# 核心接口
# ----------------------------------------------------------

def extract_toc(file_path: str | Path) -> list[dict] | None:
    """
    从 PDF 提取目录（outline/bookmarks）结构。

    返回 [{"title": "注意力机制", "page": 42, "level": 2}, ...] 或 None。
    仅支持 PDF 格式；outline 为空或解析失败返回 None。
    """
    from pypdf import PdfReader

    path = Path(file_path)
    if path.suffix.lower() != ".pdf":
        return None

    try:
        reader = PdfReader(str(path))
        outline = reader.outline
        if not outline:
            return None
    except Exception:
        return None

    toc_items: list[dict] = []

    def _resolve_page(dest) -> int:
        """将 outline destination 解析为 0-based 页码。"""
        try:
            if hasattr(dest, "page"):
                page_obj = dest.page
                # page_obj 可能是 IndirectObject，需要 get_object
                if hasattr(page_obj, "get_object"):
                    page_obj = page_obj.get_object()
                for i, p in enumerate(reader.pages):
                    if p.get_object() == page_obj:
                        return i
            # fallback: 尝试 page_number 属性
            if hasattr(dest, "page_number"):
                return dest.page_number
        except Exception:
            pass
        return 0

    def _walk(items, level: int = 1):
        for item in items:
            if isinstance(item, list):
                _walk(item, level + 1)
            else:
                try:
                    title = item.title if hasattr(item, "title") else str(item.get("/Title", ""))
                    title = title.strip()
                    if not title:
                        continue
                    page = _resolve_page(item)
                    toc_items.append({"title": title, "page": page, "level": level})
                except Exception:
                    continue

    _walk(outline)
    return toc_items if toc_items else None


def convert_to_markdown(file_path: str | Path) -> str:
    """
    将任意支持格式的文件转换为 Markdown 文本。

    支持格式：
    - .pdf  → pymupdf4llm（保留标题、表格、列表等结构）
    - .docx / .doc → mammoth
    - .md / .txt → 直接读取

    :param file_path: 文件路径
    :return:          Markdown 文本
    :raises RuntimeError:  无法转换时抛出
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return pymupdf4llm.to_markdown(str(path))

    elif suffix in (".docx", ".doc"):
        try:
            with open(path, "rb") as f:
                result = mammoth.convert_to_markdown(f)
            if result.messages:
                for msg in result.messages:
                    logger.warning(f"[mammoth] {path.name}: {msg}")
            return result.value
        except Exception as e:
            raise RuntimeError(f"Failed to convert DOCX to Markdown: {path}: {e}") from e

    elif suffix in (".md", ".txt"):
        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:
            raise RuntimeError(f"Failed to read file: {path}: {e}") from e

    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def load_file(file_path: str | Path, doc_id: Optional[str] = None) -> list[TextChunk]:
    """
    加载单个文件：先转换为 Markdown，再按章节切分为文本块。

    对于 PDF，优先使用 pymupdf4llm 转换为结构化 Markdown；
    若失败则自动回退到 PyPDFLoader 逐页解析。

    支持格式：.pdf / .docx / .doc / .md / .txt
    :param file_path: 文件路径
    :param doc_id:    文档 ID，默认使用文件名（无扩展名）
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    _doc_id = doc_id or path.stem

    # PDF: 优先 pymupdf4llm → 回退 PyPDFLoader
    if path.suffix.lower() == ".pdf":
        try:
            md_text = pymupdf4llm.to_markdown(str(path))
        except Exception as e:
            logger.warning(
                f"[loader] pymupdf4llm 转换失败，回退到 PyPDFLoader: {path.name}: {e}"
            )
            try:
                chunks = _load_pdf(path, _doc_id)
                chunks = _backfill_page_numbers(chunks, str(path))
                return chunks
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load PDF with both pymupdf4llm and PyPDFLoader: {path}: "
                    f"primary={e}, fallback={e2}"
                ) from e2
        chunks = _parse_markdown_to_chunks(md_text, _doc_id, str(path))
        chunks = _backfill_page_numbers(chunks, str(path))
        return chunks

    # 其他格式：统一转为 Markdown → 按标题切分 → 字符级切分
    md_text = convert_to_markdown(path)
    return _parse_markdown_to_chunks(md_text, _doc_id, str(path))


def load_directory(
    dir_path: str | Path,
    glob_pattern: str = "**/*",
    recursive: bool = True,
) -> list[TextChunk]:
    """
    递归扫描目录，加载所有支持格式的文档。

    :param dir_path:      目录路径
    :param glob_pattern:  文件匹配模式
    :param recursive:     是否递归扫描子目录
    :return:              所有文档块列表
    """
    supported = {".pdf", ".docx", ".doc", ".md", ".txt"}
    chunks: list[TextChunk] = []

    base_path = Path(dir_path)
    pattern = glob_pattern if recursive else glob_pattern.replace("**/", "")

    for p in base_path.glob(pattern):
        if p.is_file() and p.suffix.lower() in supported:
            try:
                file_chunks = load_file(p)
                chunks.extend(file_chunks)
            except Exception as e:
                logger.warning(f"[loader] Skipping {p}: {e}")

    return chunks


def _find_table_regions(text: str) -> list[tuple[int, int]]:
    """
    找到 Markdown 管道表格区域，返回 (start, end) 位置列表。

    识别格式：
      | Header1 | Header2 |
      |---------|---------|
      | Data1   | Data2   |
    """
    pattern = re.compile(
        r'^\|.+\|[ \t]*\n'        # 表头行
        r'^\|[-:| ]+\|[ \t]*\n'   # 分隔行（|:---|:---:| 等）
        r'(?:^\|.+\|[ \t]*\n)*',  # 数据行（0 或多行）
        re.MULTILINE,
    )
    return [(m.start(), m.end()) for m in pattern.finditer(text)]


def split_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """
    将长文本按 chunk_size（字符数）切分，相邻块保留 overlap 字符的上下文。
    尽量在句子边界处切分，避免在词/句中间截断。

    结构化内容保护：
    - 代码块（```...```）：切分前替换为占位符，切分后还原，确保语法完整
    - Markdown 表格：识别表格边界，在行尾切分；超长表格分割时重复表头

    :param text:        原始文本
    :param chunk_size:  每块字符数，默认使用配置值
    :param overlap:     重叠字符数，默认使用配置值
    :return:            文本块列表
    """
    chunk_size = chunk_size or config.rag.chunk_size
    overlap = overlap or config.rag.chunk_overlap

    if not text or len(text) <= chunk_size:
        return [text] if text else []

    # overlap 不能超过 chunk_size，否则会导致无限循环
    overlap = min(overlap, chunk_size - 1)

    # ---- 保护代码块：占位符策略（同 _split_markdown_by_headers） ----
    code_block_pattern = re.compile(r'```[\s\S]*?```')
    code_blocks: dict[str, str] = {}
    code_counter = [0]

    def _save_code(match: re.Match) -> str:
        key = f"__CODE_BLOCK_{code_counter[0]}__"
        code_blocks[key] = match.group(0)
        code_counter[0] += 1
        return key

    protected_text = code_block_pattern.sub(_save_code, text)

    def _restore_code(t: str) -> str:
        for key, original in code_blocks.items():
            t = t.replace(key, original)
        return t

    # ---- 找到表格区域 ----
    table_regions = _find_table_regions(protected_text)

    def _get_enclosing_table(pos: int) -> tuple[int, int] | None:
        """若 pos 落在某个表格区域内，返回 (table_start, table_end)。"""
        for ts, te in table_regions:
            if ts <= pos < te:
                return (ts, te)
        return None

    def _get_table_header(region_start: int) -> str:
        """提取表格区域的表头+分隔行。"""
        nl1 = protected_text.find('\n', region_start)
        if nl1 == -1:
            return ''
        nl2 = protected_text.find('\n', nl1 + 1)
        if nl2 == -1:
            return ''
        return protected_text[region_start:nl2 + 1]

    # 句子边界正则：。！？.!?\n
    sentence_endings = re.compile(r'[。！？.!?\n]')

    chunks: list[str] = []
    start = 0
    while start < len(protected_text):
        end = min(start + chunk_size, len(protected_text))

        if end < len(protected_text):
            table_info = _get_enclosing_table(end)
            if table_info is not None:
                ts, te = table_info
                if te - start <= chunk_size:
                    # 整个表格能放进当前 chunk，扩展到表格结束
                    end = te
                else:
                    # 表格太大，在行边界处切分
                    nl = protected_text.find('\n', end)
                    if nl != -1 and nl < te:
                        end = nl + 1
                    else:
                        # 找不到换行符时往前找最近的行尾
                        nl = protected_text.rfind('\n', start, end)
                        if nl != -1 and nl > ts:
                            end = nl + 1
            else:
                # 不在表格内，找句子边界
                lookback = min(int(chunk_size * config.rag.split_sentence_lookback_ratio), end - start)
                match = None
                for m in sentence_endings.finditer(protected_text, end - lookback, end):
                    match = m
                if match is not None and match.end() > start:
                    end = match.end()

        chunk_text = protected_text[start:end].strip()

        # 若当前 chunk 从表格内部开始（续前表），重复表头
        table_info_start = _get_enclosing_table(start)
        if table_info_start is not None and start > table_info_start[0]:
            header = _get_table_header(table_info_start[0])
            if header and not chunk_text.startswith(header.rstrip()):
                chunk_text = header + chunk_text

        # 还原代码块
        chunk_text = _restore_code(chunk_text)
        chunks.append(chunk_text)

        # 确保 forward progress：start 必须严格递增
        next_start = end - overlap

        # 若重叠区域落在表格内，对齐到行首，避免产生残缺行
        if _get_enclosing_table(next_start) is not None:
            nl = protected_text.rfind('\n', 0, next_start)
            if nl != -1:
                next_start = nl + 1

        if next_start <= start:
            next_start = start + chunk_size - overlap
        start = max(start + 1, next_start)

    return chunks


def docs_to_chunks(docs: list[Document], doc_id: str) -> list[TextChunk]:
    """
    将 LangChain Document 列表转换为 TextChunk 列表，
    并根据配置进行二次切分。

    :param docs:    LangChain Document 列表
    :param doc_id:  文档 ID
    :return:        TextChunk 列表
    """
    chunks: list[TextChunk] = []
    chunk_index = 0

    for doc in docs:
        text = doc.page_content.strip()
        if not text:
            continue

        # 先按配置大小切分
        sub_chunks = split_text(text)

        for sub_chunk in sub_chunks:
            chunk = TextChunk.from_langchain_doc(doc, doc_id, chunk_index)
            chunk.text = sub_chunk
            chunk.chunk_id = f"{doc_id}_{chunk_index}"
            chunks.append(chunk)
            chunk_index += 1

    return chunks


# ----------------------------------------------------------
# 私有实现：使用 LangChain Document Loader
# ----------------------------------------------------------

def _load_pdf(path: Path, doc_id: str) -> list[TextChunk]:
    """
    使用 PyPDFLoader 解析 PDF，按页切分。

    :param path:   PDF 文件路径
    :param doc_id: 文档 ID
    :return:       TextChunk 列表
    """
    try:
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        return docs_to_chunks(docs, doc_id)
    except Exception as e:
        raise RuntimeError(f"Failed to load PDF {path}: {e}") from e


# ----------------------------------------------------------
# 页码回填
# ----------------------------------------------------------

def _backfill_page_numbers(
    chunks: list[TextChunk],
    source_path: str,
) -> list[TextChunk]:
    """
    利用 PDF 目录（outline）将页码回填到 TextChunk.page。

    匹配策略：
    1. 若 chunk 已有 page（来自 PyPDFLoader 等），跳过
    2. 用 chunk.section 与 TOC entry.title 做包含匹配
    3. 未匹配的 chunk 继承最近已知页码

    :param chunks:      分块列表
    :param source_path: 原始 PDF 文件路径
    :return:            回填页码后的 chunks（原地修改）
    """
    toc = extract_toc(source_path)
    if not toc:
        return chunks

    # 构建 section → page 映射（相同标题取第一个出现的页码）
    toc_map: dict[str, int] = {}
    for entry in toc:
        title = entry["title"]
        if title not in toc_map:
            toc_map[title] = entry["page"]

    last_page: int | None = None
    for chunk in chunks:
        if chunk.page is not None:
            last_page = chunk.page
            continue

        # 尝试精确匹配 section 标题
        section = (chunk.section or "").strip()
        if section and section in toc_map:
            chunk.page = toc_map[section]
            last_page = chunk.page
            continue

        # 尝试包含匹配：chunk.section 包含 TOC title 或反之
        matched_page = None
        if section:
            for toc_title, page in toc_map.items():
                if toc_title in section or section in toc_title:
                    matched_page = page
                    break

        if matched_page is not None:
            chunk.page = matched_page
            last_page = matched_page
        elif last_page is not None:
            # 继承最近已知页码（连续章节假设）
            chunk.page = last_page

    return chunks


def _split_into_parents(
    text: str,
    parent_max_chars: int = 2000,
) -> list[str]:
    """
    将一段文本切分为父块。优先在段落边界（\\n\\n）处切分，
    保护代码块（```...```）和表格的完整性。

    :param text:             原始文本
    :param parent_max_chars: 父块最大字符数
    :return:                 父块文本列表
    """
    if len(text) <= parent_max_chars:
        return [text] if text.strip() else []

    # 保护代码块：占位符（与 split_text 一致）
    code_pattern = re.compile(r'```[\s\S]*?```')
    code_blocks: dict[str, str] = {}
    code_counter = [0]

    def _save_code(match: re.Match) -> str:
        key = f"__PC_CODE_{code_counter[0]}__"
        code_blocks[key] = match.group(0)
        code_counter[0] += 1
        return key

    protected = code_pattern.sub(_save_code, text)

    def _restore(t: str) -> str:
        for key, original in code_blocks.items():
            t = t.replace(key, original)
        return t

    # 找到表格区域
    table_regions = _find_table_regions(protected) if _find_table_regions else []

    def _in_table(pos: int) -> bool:
        for ts, te in table_regions:
            if ts <= pos < te:
                return True
        return False

    chunks: list[str] = []
    start = 0

    while start < len(protected):
        end = min(start + parent_max_chars, len(protected))

        if end < len(protected):
            # 在 parent_max_chars 附近找安全的切分点:
            # 优先级: \\n\\n (段落边界) > \\n (行尾) > 句子边界 > 硬截断
            lookback = min(config.rag.parent_chunking.parent_split_lookback, end - start)
            search_start = max(start, end - lookback)

            best = -1
            best_priority = -1

            for pos in range(end, search_start - 1, -1):
                if _in_table(pos):
                    continue  # 不在表格内切分

                # 检查代码块占位符边界（__PC_CODE_N__）
                if pos > 0 and protected[pos-1:pos+13].startswith('__PC_CODE_'):
                    continue

                if protected[pos:pos+2] == '\n\n':
                    # 还要确认这个 \n\n 不在代码占位符范围内
                    best = pos + 2
                    best_priority = 3
                    break
                elif best_priority < 2 and protected[pos] == '\n':
                    best = pos + 1
                    best_priority = 2
                elif best_priority < 1 and protected[pos] in '。！？.!?':
                    best = pos + 1
                    best_priority = 1

            if best > start and best_priority >= 0:
                end = best

        chunk_text = protected[start:end].strip()
        chunk_text = _restore(chunk_text)
        if chunk_text:
            chunks.append(chunk_text)
        start = end

    return chunks


# ----------------------------------------------------------
# 私有辅助函数
# ----------------------------------------------------------

def _parse_markdown_to_chunks(
    md_text: str,
    doc_id: str,
    source_path: str,
) -> list[TextChunk]:
    """
    将 Markdown 文本按标题（# / ## / ###）切分章节，每节再做字符级切分。

    当 config.rag.parent_chunking.enabled 为 True 时，使用父子切割模式：
    - 章节文本先按 parent_max_chars 切分为父块（不嵌入）
    - 每个父块再按 child_chunk_size 切分为子块（嵌入 + 检索）
    - 子块命中后，retriever 自动回填父块文本

    :param md_text:     Markdown 文本
    :param doc_id:      文档 ID
    :param source_path: 原始文件路径
    :return:            TextChunk 列表（含父块和子块）
    """
    parent_cfg = getattr(config.rag, 'parent_chunking', None)
    if parent_cfg and getattr(parent_cfg, 'enabled', False):
        return _parse_markdown_to_chunks_parent_child(
            md_text, doc_id, source_path,
            parent_max_chars=getattr(parent_cfg, 'parent_max_chars', 2000),
            child_chunk_size=getattr(parent_cfg, 'child_chunk_size', None),
        )

    # ---- 原逻辑：固定切分（向后兼容） ----
    sections = _split_markdown_by_headers(md_text)

    # 从文件路径提取课程名（取直接父目录名）
    course = Path(source_path).parent.name if Path(source_path).parent.name else None

    chunks: list[TextChunk] = []
    chunk_index = 0

    for section_title, section_text in sections:
        sub_chunks = split_text(section_text)
        for sub_chunk in sub_chunks:
            meta = {}
            # 自动检测语言
            lang = _detect_language(sub_chunk)
            if lang:
                meta["language"] = lang
            # 自动填充课程名
            if course:
                meta["course"] = course

            chunk = TextChunk(
                chunk_id=f"{doc_id}_{chunk_index}",
                text=sub_chunk,
                doc_id=doc_id,
                source_path=source_path,
                section=section_title,
                metadata=meta,
            )
            chunks.append(chunk)
            chunk_index += 1

    return chunks


def _parse_markdown_to_chunks_parent_child(
    md_text: str,
    doc_id: str,
    source_path: str,
    parent_max_chars: int = 2000,
    child_chunk_size: int | None = None,
) -> list[TextChunk]:
    """
    父子切割模式：章节 → 父块 → 子块。

    父块存储完整章节/大段文本（不嵌入），子块参与向量检索。
    子块通过 parent_chunk_id 回指父块，检索后自动回填父块上下文。

    :param md_text:          Markdown 文本
    :param doc_id:           文档 ID
    :param source_path:      原始文件路径
    :param parent_max_chars: 父块最大字符数
    :param child_chunk_size: 子块大小，None 使用 config.rag.chunk_size
    :return:                 TextChunk 列表（父块在前，子块在后）
    """
    from backend.config import config as _cfg

    sections = _split_markdown_by_headers(md_text)
    course = Path(source_path).parent.name if Path(source_path).parent.name else None
    child_size = child_chunk_size or _cfg.rag.chunk_size

    chunks: list[TextChunk] = []
    chunk_index = 0

    for section_title, section_text in sections:
        # Step 1: 将章节切分为父块（保护代码块/表格完整性）
        parent_texts = _split_into_parents(section_text, parent_max_chars)

        for parent_text in parent_texts:
            parent_id = f"{doc_id}_p{chunk_index}"
            chunk_index += 1

            # 父块元数据
            parent_meta = {}
            lang = _detect_language(parent_text)
            if lang:
                parent_meta["language"] = lang
            if course:
                parent_meta["course"] = course

            # 写入父块（is_parent=True，不嵌入）
            parent_chunk = TextChunk(
                chunk_id=parent_id,
                text=parent_text,
                doc_id=doc_id,
                source_path=source_path,
                section=section_title,
                is_parent=True,
                metadata=parent_meta,
            )
            chunks.append(parent_chunk)

            # Step 2: 父块切分为子块，建立父子关系
            child_texts = split_text(parent_text, chunk_size=child_size)
            for child_text in child_texts:
                child_meta = {}
                lang = _detect_language(child_text)
                if lang:
                    child_meta["language"] = lang
                if course:
                    child_meta["course"] = course

                child_chunk = TextChunk(
                    chunk_id=f"{doc_id}_{chunk_index}",
                    text=child_text,
                    doc_id=doc_id,
                    source_path=source_path,
                    section=section_title,
                    parent_chunk_id=parent_id,
                    is_parent=False,
                    metadata=child_meta,
                )
                chunks.append(child_chunk)
                chunk_index += 1

    logger.info(
        f"[loader] 父子切割：{len(sections)} 个章节 → "
        f"{sum(1 for c in chunks if c.is_parent)} 个父块 + "
        f"{sum(1 for c in chunks if not c.is_parent)} 个子块"
    )
    return chunks


def _split_markdown_by_headers(text: str) -> list[tuple[str, str]]:
    """
    按 Markdown 标题（# / ## / ###）切分章节。

    - 优先按一级标题 # 分割；无 # 则按 ##；再按 ###
    - 标题层级越高（# 最少），划分出的章节粒度越粗
    - 代码块内的 # 不会被识别为标题

    :param text: 原始 Markdown 文本
    :return:     [(章节标题, 章节内容), ...] 列表
    """
    # 先提取并保存代码块，避免代码中的 # 被误解析为标题
    code_block_pattern = re.compile(r'```[\s\S]*?```')
    code_blocks: dict[str, str] = {}
    counter = [0]

    def _save_code(match: re.Match) -> str:
        key = f"__CODE_BLOCK_{counter[0]}__"
        code_blocks[key] = match.group(0)
        counter[0] += 1
        return key

    text_without_code = code_block_pattern.sub(_save_code, text)

    def _restore_code(text_section: str) -> str:
        for key, original in code_blocks.items():
            text_section = text_section.replace(key, original)
        return text_section

    # 匹配 #、##、### 三级标题行
    header_pattern = re.compile(r'^(#{1,3})\s+(.+)$', re.MULTILINE)
    headers = list(header_pattern.finditer(text_without_code))

    if not headers:
        # 没有标题，返回整篇作为"无标题"章节
        return [("无标题", _restore_code(text_without_code).strip())]

    # 选择实际出现的最小标题级别作为分割基准
    min_level = min(len(m.group(1)) for m in headers)
    effective_headers = [m for m in headers if len(m.group(1)) == min_level]

    # 检查是否分割过细：若按 min_level 切出超过 50 个 section，
    # 则回退到下一级标题（更粗粒度）
    if len(effective_headers) > config.rag.max_sections_before_coarse_split and min_level < 3:
        next_level = min_level + 1
        effective_headers = [m for m in headers if len(m.group(1)) == next_level]
        if not effective_headers:
            effective_headers = [m for m in headers if len(m.group(1)) == min_level]

    sections: list[tuple[str, str]] = []

    for i, match in enumerate(effective_headers):
        title = match.group(2).strip()
        start = match.end()
        # 下一个同级标题之前，或文件末尾
        end = (
            effective_headers[i + 1].start()
            if i + 1 < len(effective_headers)
            else len(text_without_code)
        )

        section_text = text_without_code[start:end].strip()
        # 还原代码块（恢复原始内容）
        section_text = _restore_code(section_text)

        if section_text:
            sections.append((title, section_text))

    return sections


# ----------------------------------------------------------
# 便捷函数：直接返回 LangChain Document
# ----------------------------------------------------------

def load_file_as_documents(file_path: str | Path, doc_id: str | None = None) -> list[Document]:
    """
    加载文件并直接返回 LangChain Document 列表（跳过 TextChunk 转换）。

    适用于需要保留 LangChain Document 完整元数据的场景。

    :param file_path: 文件路径
    :param doc_id:    文档 ID
    :return:          Document 列表
    """
    chunks = load_file(file_path, doc_id)
    return [chunk.to_langchain_doc() for chunk in chunks]


def load_directory_as_documents(
    dir_path: str | Path,
    glob_pattern: str = "**/*",
    recursive: bool = True,
) -> list[Document]:
    """
    加载目录并直接返回 LangChain Document 列表。

    :param dir_path:      目录路径
    :param glob_pattern:  文件匹配模式
    :param recursive:      是否递归
    :return:               Document 列表
    """
    chunks = load_directory(dir_path, glob_pattern, recursive)
    return [chunk.to_langchain_doc() for chunk in chunks]


def _detect_language(text: str) -> str:
    """
    检测文本语言：统计中文字符和英文字符比例。

    :return: "zh" | "en" | "mixed"
    """
    cn_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    en_chars = len(re.findall(r'[a-zA-Z]', text))
    total = cn_chars + en_chars
    if total == 0:
        return "unknown"
    cn_ratio = cn_chars / total
    if cn_ratio > 0.75:
        return "zh"
    elif cn_ratio < 0.25:
        return "en"
    return "mixed"
