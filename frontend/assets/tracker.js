/**
 * 页面停留时长追踪器 — 记录活跃停留时间并在离开时上报。
 * 用法：<script src="./assets/tracker.js"></script>
 */
(function() {
    'use strict';

    const API_BASE = window.location.origin;
    const MIN_DURATION = 5; // 最少停留 5 秒才上报（过滤意外跳转）
    const HEARTBEAT_INTERVAL = 60; // 每 60 秒发一次心跳保底上报

    let startTime = Date.now();
    let activeTime = 0;
    let lastActive = Date.now();
    let isVisible = true;
    let reported = false;
    let heartbeatTimer = null;

    function getUserId() {
        return localStorage.getItem('user_id');
    }

    function getToken() {
        return localStorage.getItem('access_token');
    }

    function getPageName() {
        const path = window.location.pathname;
        const file = path.split('/').pop() || 'index.html';
        return file.replace('.html', '');
    }

    function reportDuration(durationSec) {
        const userId = getUserId();
        if (!userId || durationSec < MIN_DURATION) return;
        if (reported) return;
        reported = true;

        const data = {
            action: 'stay',
            duration_seconds: durationSec,
        };

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
        }
    }

    // 心跳上报：防止用户长时间停留但从不离开导致数据丢失
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

        activeTime = 0;
        lastActive = Date.now();
        reported = false;
    }

    function handleVisibility() {
        if (document.hidden) {
            if (isVisible) {
                activeTime += Date.now() - lastActive;
                isVisible = false;
            }
        } else {
            lastActive = Date.now();
            isVisible = true;
            reported = false;
        }
    }

    function handleUnload() {
        if (isVisible) {
            activeTime += Date.now() - lastActive;
        }
        const totalSec = Math.round(activeTime / 1000);
        reportDuration(totalSec);
    }

    function init() {
        if (!getUserId()) return; // 未登录不追踪

        document.addEventListener('visibilitychange', handleVisibility);
        window.addEventListener('beforeunload', handleUnload);
        window.addEventListener('pagehide', handleUnload);

        heartbeatTimer = setInterval(heartbeat, HEARTBEAT_INTERVAL * 1000);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
