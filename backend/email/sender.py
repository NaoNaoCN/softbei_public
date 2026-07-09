"""
backend/email/sender.py
异步邮件发送器，封装 aiosmtplib，支持自动重试。
SMTP 未配置时自动切换为本地文件模式（保存 .html 到 debug_emails/ 目录）。
"""

from __future__ import annotations

import os
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr, parseaddr
from pathlib import Path

import aiosmtplib
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import config


class EmailSender:
    """异步邮件发送器。SMTP 未配置时自动降级为本地文件保存，方便开发调试。"""

    def __init__(self):
        cfg = config.email
        self._host = cfg.smtp_host
        self._port = cfg.smtp_port
        self._username = cfg.smtp_username
        self._password = cfg.smtp_password
        self._from = cfg.smtp_from
        self._use_tls = cfg.smtp_use_tls
        self._timeout = cfg.smtp_timeout
        self._enabled = bool(cfg.smtp_host and cfg.smtp_password)
        # 调试模式：SMTP 未配置时，邮件保存到本地文件
        self._debug_dir = Path(__file__).resolve().parent.parent.parent / "debug_emails"

    @property
    def enabled(self) -> bool:
        return self._enabled

    @retry(
        stop=stop_after_attempt(config.email.max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def send(self, to: str, subject: str, html: str) -> bool:
        """发送 HTML 邮件，失败时自动重试。SMTP 未配置时保存到本地文件。"""
        if not self._enabled:
            return self._save_to_file(to, subject, html)

        # Extract bare email for SMTP envelope; always use project name as display name.
        # QQ/163 mail servers reject messages with malformed From headers.
        _, from_addr = parseaddr(self._from)
        if not from_addr:
            logger.warning("[Email] From 地址解析失败: raw={}", self._from)
            from_addr = self._from
        msg_from = formataddr(("智学工坊", from_addr))

        msg = MIMEMultipart("alternative")
        msg["From"] = msg_from
        msg["To"] = to
        msg["Subject"] = subject
        msg.attach(MIMEText(html, "html", "utf-8"))

        try:
            await aiosmtplib.send(
                msg,
                sender=from_addr,
                hostname=self._host,
                port=self._port,
                username=self._username,
                password=self._password,
                use_tls=self._use_tls,
                timeout=self._timeout,
                local_hostname="localhost",
            )
            logger.success("[Email] 发送成功: to={}, subject={}", to, subject)
            return True
        except Exception as e:
            logger.exception("[Email] 发送失败: to={}, subject={}, error={}", to, subject, e)
            raise

    def _save_to_file(self, to: str, subject: str, html: str) -> bool:
        """调试模式：将邮件保存为本地 HTML 文件。"""
        self._debug_dir.mkdir(parents=True, exist_ok=True)
        safe_subject = "".join(c if c.isalnum() or c in "._- " else "_" for c in subject)[:50]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_{safe_subject}.html"
        filepath = self._debug_dir / filename
        filepath.write_text(html, encoding="utf-8")
        logger.info("[Email] [DEBUG] 邮件已保存到本地: {}", filepath)
        return True

    async def send_verification(self, to: str, username: str, token: str) -> bool:
        """发送邮箱验证邮件。"""
        from backend.email.templates import render_verify_email

        frontend_base = "http://localhost:8000/app"
        verify_url = f"{frontend_base}/verify-email.html?token={token}"

        html = render_verify_email(username, verify_url)
        return await self.send(to, "[学习系统] 请验证您的邮箱地址", html)

    async def send_password_reset(self, to: str, username: str, token: str) -> bool:
        """发送密码重置邮件。"""
        from backend.email.templates import render_reset_password

        frontend_base = "http://localhost:8000/app"
        reset_url = f"{frontend_base}/reset-password.html?token={token}"

        html = render_reset_password(username, reset_url)
        return await self.send(to, "[学习系统] 密码重置请求", html)

    async def send_learning_report(self, to: str, username: str, report_data: dict) -> bool:
        """发送学习报告邮件。"""
        from backend.email.templates import render_learning_report
        from datetime import date

        report_data["date"] = date.today().isoformat()
        html = render_learning_report(username, report_data)
        return await self.send(to, f"[学习系统] 您的学习报告（{report_data['date']}）", html)

    async def send_study_plan(self, to: str, username: str, plan_data: dict) -> bool:
        """发送学习计划表邮件。"""
        from backend.email.templates import render_study_plan

        html = render_study_plan(username, plan_data)
        subject = f"[学习系统] 您的学习计划表 · {plan_data.get('title', '')}"
        return await self.send(to, subject, html)


# 模块级单例
email_sender = EmailSender()
