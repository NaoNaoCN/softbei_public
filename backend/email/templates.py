"""
backend/email/templates.py
Jinja2 邮件模板渲染。
"""

from __future__ import annotations

from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path

_templates_dir = Path(__file__).resolve().parent.parent / "templates" / "email"
_env = Environment(
    loader=FileSystemLoader(str(_templates_dir)),
    autoescape=select_autoescape(["html"]),
)


def render_verify_email(username: str, verify_url: str) -> str:
    """渲染邮箱验证邮件 HTML。"""
    return _env.get_template("verify_email.html").render(
        username=username, verify_url=verify_url
    )


def render_reset_password(username: str, reset_url: str) -> str:
    """渲染密码重置邮件 HTML。"""
    return _env.get_template("reset_password.html").render(
        username=username, reset_url=reset_url
    )


def render_learning_report(username: str, report_data: dict) -> str:
    """渲染学习报告邮件 HTML。"""
    return _env.get_template("learning_report.html").render(
        username=username, **report_data
    )


def render_study_plan(username: str, plan_data: dict) -> str:
    """渲染学习计划表邮件 HTML。"""
    return _env.get_template("study_plan.html").render(
        username=username, **plan_data
    )
