/**
 * 页面停留时长追踪器
 * 自动记录用户在每个页面的活跃停留时间，离开页面时上报到后端。
 * 使用方式：在页面底部添加 <script src="./assets/tracker.js"></script>
 */
(function() {
    'use strict';

    const API_BASE = window.location.origin;
    const MIN_DURATION = 5; // 最少停留 5 秒才上报（过滤意外跳转）
    const HEARTBEAT_INTERVAL = 60; // 每 60 秒发一次心跳保底上报

    let startTime = Date.now();
    let activeTime = 0;     // 累计活跃时间（毫秒）
    let lastActive = Date.now();
    let isVisible = true;
    let reported = false;
    let heartbeatTimer = null;

    // 获取用户 ID
    function getUserId() {
        return localStorage.getItem('user_id');
    }

    function getToken() {
        return localStorage.getItem('access_token');
    }

    // 获取当前页面名称
    function getPageName() {
        const path = window.location.pathname;
        const file = path.split('/').pop() || 'index.html';
        return file.replace('.html', '');
    }

    // 上报停留时长
    function reportDuration(durationSec) {
        const userId = getUserId();
        if (!userId || durationSec < MIN_DURATION) return;
        if (reported) return;
        reported = true;

        const data = {
            action: 'stay',
            duration_seconds: durationSec,
        };

        // 使用 sendBeacon 确保页面关闭时也能发送
        const url = `${API_BASE}/records?user_id=${encodeURIComponent(userId)}`;
        const headers = { 'Content-Type': 'application/json' };
        const token = getToken();
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }

        // sendBeacon 不支持自定义 headers，用 fetch keepalive 代替
        try {
            fetch(url, {
                method: 'POST',
                headers: headers,
                body: JSON.stringify(data),
                keepalive: true,
            }).catch(() => {});
        } catch (e) {
            // 静默失败
        }
    }

    // 心跳上报：每隔一段时间上报一次，防止用户长时间停留但从不离开
    function heartbeat() {
        const userId = getUserId();
        if (!userId) return;

        const elapsed = Math.round(activeTime / 1000);
        if (elapsed < MIN_DURATION) return;

        const data = {
            action: 'stay',
            duration_seconds: elapsed,
        };

        const url = `${API_BASE}/records?user_id=${encodeURIComponent(userId)}`;
        const token = getToken();
        const headers = { 'Content-Type': 'application/json' };
        if (token) headers['Authorization'] = `Bearer ${token}`;

        fetch(url, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify(data),
            keepalive: true,
        }).catch(() => {});

        // 重置计时
        activeTime = 0;
        lastActive = Date.now();
        reported = false;
    }

    // 页面可见性变化
    function handleVisibility() {
        if (document.hidden) {
            // 页面隐藏，累计活跃时间
            if (isVisible) {
                activeTime += Date.now() - lastActive;
                isVisible = false;
            }
        } else {
            // 页面恢复可见
            lastActive = Date.now();
            isVisible = true;
            reported = false;
        }
    }

    // 页面卸载
    function handleUnload() {
        if (isVisible) {
            activeTime += Date.now() - lastActive;
        }
        const totalSec = Math.round(activeTime / 1000);
        reportDuration(totalSec);
    }

    // 初始化
    function init() {
        if (!getUserId()) return; // 未登录不追踪

        document.addEventListener('visibilitychange', handleVisibility);
        window.addEventListener('beforeunload', handleUnload);
        window.addEventListener('pagehide', handleUnload);

        // 心跳定时器：每 60 秒上报一次累计时间
        heartbeatTimer = setInterval(heartbeat, HEARTBEAT_INTERVAL * 1000);
    }

    // DOM ready 后初始化
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
