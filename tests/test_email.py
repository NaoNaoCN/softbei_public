"""
Email 功能测试脚本
测试所有邮件相关 API 端点，包含正常流程和异常场景。
SMTP 未配置时邮件自动保存到 debug_emails/ 目录。
"""
import asyncio
import os
import sys
import json
from pathlib import Path

# 确保项目在 path 中
sys.path.insert(0, str(Path(__file__).parent))

from httpx import AsyncClient, ASGITransport
from backend.main import app
from backend.config import config
from backend.email.sender import email_sender
from backend.db.database import init_db

BASE = "http://test"
DEBUG_DIR = Path(__file__).parent / "debug_emails"

# 测试数据
TEST_USER = "email_test_user_" + os.urandom(4).hex()
TEST_EMAIL = f"test_{os.urandom(4).hex()}@example.com"
TEST_PASS = "test123456"

# 收集测试结果
results: list[dict] = []


def record(name: str, ok: bool, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    results.append({"name": name, "status": status, "detail": detail})
    icon = "[OK]" if ok else "[FAIL]"
    print(f"  {icon} {name}" + (f" — {detail}" if detail else ""))


async def test_all():
    await init_db()
    transport = ASGITransport(app=app)
    token = None
    user_id = None

    # 清空调试目录
    if DEBUG_DIR.exists():
        for f in DEBUG_DIR.iterdir():
            f.unlink()
    print(f"\n当前 SMTP 状态: enabled={email_sender.enabled}, debug_dir={DEBUG_DIR}")
    print("-" * 60)

    async with AsyncClient(transport=transport, base_url=BASE) as client:
        # ────────────────────────────────────────────────
        # 1. 注册 (无 email)
        # ────────────────────────────────────────────────
        print("\n[1] 注册 (无 email)")
        resp = await client.post("/auth/register", json={
            "username": TEST_USER + "_noemail",
            "password": TEST_PASS,
        })
        record("注册(无email) → 返回200", resp.status_code == 200)
        data = resp.json()
        record("注册(无email) → email为null", data.get("email") is None)
        record("注册(无email) → email_verified=False", data.get("email_verified") == False)

        # ────────────────────────────────────────────────
        # 2. 注册 (有 email)
        # ────────────────────────────────────────────────
        print("\n[2] 注册 (有 email)")
        resp = await client.post("/auth/register", json={
            "username": TEST_USER,
            "password": TEST_PASS,
            "email": TEST_EMAIL,
        })
        record("注册(有email) → 返回200", resp.status_code == 200)
        data = resp.json()
        record("注册(有email) → email字段正确", data.get("email") == TEST_EMAIL)
        record("注册(有email) → email_verified=False", data.get("email_verified") == False)
        user_id = int(data["id"])

        # 等后台任务完成
        await asyncio.sleep(0.5)

        # 检查验证邮件是否生成
        email_files = list(DEBUG_DIR.glob("*.html"))
        verify_files = [f for f in email_files if "验证" in f.name]
        record(f"注册 → debug_emails/ 有验证邮件 ({len(verify_files)} 封)", len(verify_files) > 0)

        # ────────────────────────────────────────────────
        # 3. 重复用户名注册
        # ────────────────────────────────────────────────
        print("\n[3] 重复用户名注册")
        resp = await client.post("/auth/register", json={
            "username": TEST_USER,
            "password": TEST_PASS,
        })
        record("重复注册 → 返回400", resp.status_code == 400)
        record("重复注册 → 'Username already exists'", "already exists" in resp.json().get("detail", "").lower())

        # ────────────────────────────────────────────────
        # 4. 重复邮箱注册
        # ────────────────────────────────────────────────
        print("\n[4] 重复邮箱注册")
        resp = await client.post("/auth/register", json={
            "username": TEST_USER + "_another",
            "password": TEST_PASS,
            "email": TEST_EMAIL,
        })
        record("重复邮箱 → 返回400", resp.status_code == 400)

        # ────────────────────────────────────────────────
        # 5. 登录
        # ────────────────────────────────────────────────
        print("\n[5] 登录")
        resp = await client.post("/auth/login", json={
            "username": TEST_USER,
            "password": TEST_PASS,
        })
        record("登录 → 返回200", resp.status_code == 200)
        data = resp.json()
        token = data.get("access_token")
        record("登录 → 返回access_token", bool(token))

        # ────────────────────────────────────────────────
        # 6. 发送验证邮件
        # ────────────────────────────────────────────────
        print("\n[6] POST /auth/send-verification")
        headers = {"Authorization": f"Bearer {token}"}
        resp = await client.post("/auth/send-verification", headers=headers)
        record("发验证邮件 → 返回200", resp.status_code == 200)
        record("发验证邮件 → '验证邮件已发送'", resp.json().get("message") == "验证邮件已发送")

        await asyncio.sleep(0.5)

        # ────────────────────────────────────────────────
        # 7. 验证邮箱 (无效 token)
        # ────────────────────────────────────────────────
        print("\n[7] GET /auth/verify-email (无效token)")
        resp = await client.get("/auth/verify-email?token=invalid_token_xxx")
        # 400 error 返回 JSON
        record("无效token → 返回4xx", resp.status_code >= 400)

        # ────────────────────────────────────────────────
        # 8. 验证邮箱 (有效 token)
        # ────────────────────────────────────────────────
        print("\n[8] GET /auth/verify-email (有效token)")
        # 从 debug_emails 中提取 token
        from backend.db.database import _session_factory
        from backend.db.models import EmailVerification
        from backend.email.utils import hash_token
        import sqlalchemy as sa

        raw_token = None
        if _session_factory:
            async with _session_factory() as db:
                row = await db.execute(
                    sa.select(EmailVerification).where(
                        EmailVerification.user_id == user_id,
                        EmailVerification.purpose == "email_verify",
                        EmailVerification.used == False,
                    ).order_by(EmailVerification.created_at.desc()).limit(1)
                )
                ver_record = row.scalar_one_or_none()
                if ver_record:
                    # 从 debug_emails 邮件中提取原始 token
                    for f in sorted(DEBUG_DIR.glob("*.html"), key=lambda x: x.stat().st_mtime, reverse=True):
                        content = f.read_text(encoding="utf-8")
                        if "token=" in content:
                            import re
                            m = re.search(r"token=([a-zA-Z0-9_-]+)", content)
                            if m:
                                raw_token = m.group(1)
                                break

        if raw_token:
            resp = await client.get(f"/auth/verify-email?token={raw_token}")
            # 成功时返回 302 重定向
            record("有效token → 返回重定向", resp.status_code in (200, 302, 307))
            if resp.status_code in (302, 307):
                record("重定向到verify-email.html?status=success", "verify-email.html" in resp.headers.get("location", ""))
        else:
            record("有效token → 提取到token", False, "无法从debug_emails中提取token")

        # ────────────────────────────────────────────────
        # 9. 忘记密码 (不存在的邮箱)
        # ────────────────────────────────────────────────
        print("\n[9] POST /auth/forgot-password (不存在的邮箱)")
        resp = await client.post("/auth/forgot-password", json={
            "email": "nonexistent@example.com",
        })
        record("不存在邮箱 → 返回200", resp.status_code == 200)
        record("不存在邮箱 → 统一成功消息", resp.json().get("message"))

        # ────────────────────────────────────────────────
        # 10. 忘记密码 (存在的邮箱)
        # ────────────────────────────────────────────────
        print("\n[10] POST /auth/forgot-password (存在的邮箱)")
        resp = await client.post("/auth/forgot-password", json={
            "email": TEST_EMAIL,
        })
        record("存在邮箱 → 返回200", resp.status_code == 200)
        record("存在邮箱 → 统一成功消息", resp.json().get("message"))

        await asyncio.sleep(0.5)

        # 检查密码重置邮件
        email_files = list(DEBUG_DIR.glob("*.html"))
        reset_files = [f for f in email_files if "密码重置" in f.name]
        record(f"忘记密码 → debug_emails/ 有重置邮件 ({len(reset_files)} 封)", len(reset_files) > 0)

        # ────────────────────────────────────────────────
        # 11. 重置密码 (无效 token)
        # ────────────────────────────────────────────────
        print("\n[11] POST /auth/reset-password (无效token)")
        resp = await client.post("/auth/reset-password", json={
            "token": "invalid_token_xxx",
            "new_password": "newpass123",
        })
        record("无效token → 返回400", resp.status_code == 400)

        # ────────────────────────────────────────────────
        # 12. 重置密码 (有效 token)
        # ────────────────────────────────────────────────
        print("\n[12] POST /auth/reset-password (有效token)")
        # 从 debug_emails 提取密码重置 token
        reset_token = None
        for f in sorted(DEBUG_DIR.glob("*.html"), key=lambda x: x.stat().st_mtime, reverse=True):
            content = f.read_text(encoding="utf-8")
            if "reset-password.html?token=" in content:
                import re
                m = re.search(r"token=([a-zA-Z0-9_-]+)", content)
                if m:
                    reset_token = m.group(1)
                    break

        if reset_token:
            resp = await client.post("/auth/reset-password", json={
                "token": reset_token,
                "new_password": "newpass456",
            })
            record("有效token → 返回200", resp.status_code == 200)
            record("有效token → '密码重置成功'", resp.json().get("message") == "密码重置成功")

            # 用新密码登录
            resp = await client.post("/auth/login", json={
                "username": TEST_USER,
                "password": "newpass456",
            })
            record("新密码登录 → 返回200", resp.status_code == 200)
            token = resp.json().get("access_token")
        else:
            record("有效token → 提取到token", False, "无法从debug_emails中提取token")

        # ────────────────────────────────────────────────
        # 13. 发送学习报告
        # ────────────────────────────────────────────────
        print("\n[13] POST /email/learning-report")
        if token:
            headers = {"Authorization": f"Bearer {token}"}
            resp = await client.post("/email/learning-report", headers=headers)
            record("学习报告 → 返回200", resp.status_code == 200)
            record("学习报告 → 提示已发送", "已发送" in resp.json().get("message", ""))

            await asyncio.sleep(0.5)
            report_files = [f for f in DEBUG_DIR.glob("*.html") if "学习报告" in f.name]
            record(f"学习报告 → debug_emails/ 有报告邮件 ({len(report_files)} 封)", len(report_files) > 0)

    # ────────────────────────────────────────────────
    # 汇总
    # ────────────────────────────────────────────────
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    total = len(results)
    email_count = len(list(DEBUG_DIR.glob("*.html")))

    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{total} 通过, {failed} 失败")
    print(f"本地邮件文件: {email_count} 封 → {DEBUG_DIR}")
    print("=" * 60)

    if failed > 0:
        print("\n失败详情:")
        for r in results:
            if r["status"] == "FAIL":
                print(f"  [FAIL] {r['name']}: {r['detail']}")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(test_all())
    sys.exit(0 if success else 1)
