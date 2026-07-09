/**
 * assistant.js — AI 学习伴侣悬浮面板
 *
 * 功能：
 *   1. 悬浮机器人按钮（右下角）
 *   2. 多 Tab 面板：今日任务 / 学情摘要 / 番茄钟
 *   3. 每日学习提醒弹窗（每天首次进入首页时显示）
 *   4. 随机气泡提示
 */

import { getUserId, isLoggedIn, getLearningAnalytics, getProfile } from './api.js';

// ============================================================
// 常量
// ============================================================

const MOTIVATIONS = [
    '坚持就是胜利，今天也要加油学习哦！',
    '每天进步一点点，日积月累成就大不同。',
    '学而不思则罔，思而不学则殆。',
    '今天的努力是明天的底气。',
    '知识是最好的投资，开始今天的学习吧！',
    '不积跬步，无以至千里。一起加油！',
    '天道酬勤，越努力越幸运！',
    '学习使人充实，坚持让你出众。',
];

const BUBBLE_MESSAGES = [
    '有什么我可以帮你的吗？',
    '今天复习了吗？别让知识溜走哦~',
    '试试番茄钟，保持专注！',
    '点击我可以随时提问哦~',
    '你已经很棒了，继续保持！',
    '来个番茄钟，专注25分钟吧！',
];

const GREETINGS = {
    morning: '早上好',
    afternoon: '下午好',
    evening: '晚上好',
    night: '夜深了',
};

// ============================================================
// 工具函数
// ============================================================

function getTimeGreeting() {
    const h = new Date().getHours();
    if (h >= 5 && h < 12) return GREETINGS.morning;
    if (h >= 12 && h < 18) return GREETINGS.afternoon;
    if (h >= 18 && h < 22) return GREETINGS.evening;
    return GREETINGS.night;
}

function getMotivation() {
    return MOTIVATIONS[Math.floor(Math.random() * MOTIVATIONS.length)];
}

function getTodayKey(userId) {
    const today = new Date().toISOString().slice(0, 10);
    return `softbei_daily_reminder_${userId}_${today}`;
}

// ============================================================
// 创建 DOM
// ============================================================

function createAssistantDOM() {
    // 悬浮按钮
    const fab = document.createElement('button');
    fab.className = 'ai-bot-fab';
    fab.id = 'aiBotFab';
    fab.innerHTML = `
        <img class="xiaozhi-eye" src="/app/assets/xiaozhi-mascot.svg" alt="" aria-hidden="true" draggable="false">
        <span class="bot-close">✕</span>
        <span class="bot-badge" id="aiBotBadge"></span>
    `;
    fab.setAttribute('aria-label', '小知 · AI 学习助手');

    // 气泡
    const bubble = document.createElement('div');
    bubble.className = 'ai-bot-bubble';
    bubble.id = 'aiBotBubble';

    // 主面板
    const panel = document.createElement('div');
    panel.className = 'ai-bot-panel';
    panel.id = 'aiBotPanel';
    panel.innerHTML = `
        <div class="ai-panel-header">
            <span class="panel-bot-icon"><img class="xiaozhi-eye xz-mini" src="/app/assets/xiaozhi-mascot.svg" alt="" aria-hidden="true" draggable="false"></span>
            <div>
                <div class="panel-title">学习小助手</div>
                <div class="panel-subtitle">你的专属学习伴侣</div>
            </div>
            <div class="ai-panel-today-time">
                <span class="today-time-icon"><i data-lucide="clock" style="width:14px;height:14px;"></i></span>
                <span class="today-time-value" id="aiTodayTime">0分钟</span>
                <span class="today-time-label">今日已学习</span>
            </div>
        </div>
        <div class="ai-panel-tabs">
            <div class="ai-panel-tab active" data-tab="tasks">
                <span class="tab-icon"><i data-lucide="calendar-check" style="width:16px;height:16px;"></i></span>
                <span>今日任务</span>
            </div>
            <div class="ai-panel-tab" data-tab="stats">
                <span class="tab-icon"><i data-lucide="bar-chart-3" style="width:16px;height:16px;"></i></span>
                <span>学情</span>
            </div>
            <div class="ai-panel-tab" data-tab="pomodoro">
                <span class="tab-icon"><i data-lucide="timer" style="width:16px;height:16px;"></i></span>
                <span>番茄钟</span>
            </div>
        </div>
        <div class="ai-panel-content">
            <!-- Tab: 今日任务 -->
            <div class="ai-tab-pane active" id="aiTabTasks">
                <div id="aiTasksContent">
                    <div class="ai-empty-state">加载中...</div>
                </div>
            </div>
            <!-- Tab: 学情 -->
            <div class="ai-tab-pane" id="aiTabStats">
                <div id="aiStatsContent">
                    <div class="ai-empty-state">加载中...</div>
                </div>
            </div>
            <!-- Tab: 番茄钟 -->
            <div class="ai-tab-pane" id="aiTabPomodoro">
                <div class="ai-pomodoro">
                    <div class="ai-pomo-circle" id="aiPomoCircle">
                        <div class="ai-pomo-time" id="aiPomoTime">25:00</div>
                        <div class="ai-pomo-label" id="aiPomoLabel">专注时间</div>
                    </div>
                    <div class="ai-pomo-controls" id="aiPomoControls">
                        <button class="ai-pomo-btn primary" id="aiPomoStart">开始专注</button>
                    </div>
                    <div class="ai-pomo-stats">
                        <span><i data-lucide="timer" style="width:13px;height:13px;vertical-align:-2px;margin-right:2px;"></i>今日完成 <span class="pomo-count" id="aiPomoCount">0</span> 个</span>
                        <span><i data-lucide="clock" style="width:13px;height:13px;vertical-align:-2px;margin-right:2px;"></i>共 <span class="pomo-count" id="aiPomoMinutes">0</span> 分钟</span>
                    </div>
                    <div class="ai-pomo-settings">
                        <div class="ai-pomo-setting-row">
                            <span>专注时长（分钟）</span>
                            <input type="number" id="aiPomoWorkMin" value="25" min="5" max="60">
                        </div>
                        <div class="ai-pomo-setting-row">
                            <span>休息时长（分钟）</span>
                            <input type="number" id="aiPomoBreakMin" value="5" min="1" max="30">
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;

    // 每日提醒弹窗
    const reminder = document.createElement('div');
    reminder.className = 'daily-reminder-overlay';
    reminder.id = 'dailyReminder';
    reminder.innerHTML = `
        <div class="daily-reminder-card">
            <div class="daily-reminder-icon" id="reminderIcon"><i data-lucide="sunrise" style="width:40px;height:40px;color:#C77B3C;"></i></div>
            <div class="daily-reminder-greeting" id="reminderGreeting">早上好！</div>
            <div class="daily-reminder-time" id="reminderTime"></div>
            <div id="reminderContent"></div>
            <div class="daily-reminder-tip" id="reminderTip"></div>
            <div class="daily-reminder-actions">
                <button class="daily-reminder-btn primary" id="reminderStartBtn">开始学习</button>
                <button class="daily-reminder-btn secondary" id="reminderDismissBtn">稍后再说</button>
            </div>
            <div class="daily-motivation" id="reminderMotivation"></div>
        </div>
    `;

    document.body.appendChild(fab);
    document.body.appendChild(bubble);
    document.body.appendChild(panel);
    document.body.appendChild(reminder);

    // 渲染 Lucide 图标
    if (typeof lucide !== 'undefined') lucide.createIcons();
}

// ============================================================
// 面板 & Tab 交互
// ============================================================

let panelOpen = false;
let bubbleTimeout = null;
let currentTab = 'tasks';

// 全局状态持久化
function getPanelStateKey() {
    return `softbei_panel_state_${getUserId()}`;
}

function savePanelState() {
    const data = { tab: currentTab, open: panelOpen };
    localStorage.setItem(getPanelStateKey(), JSON.stringify(data));
}

function restorePanelState() {
    try {
        const raw = localStorage.getItem(getPanelStateKey());
        if (!raw) return;
        const data = JSON.parse(raw);
        if (data.tab) currentTab = data.tab;
        if (data.open) panelOpen = true;
    } catch (e) { /* ignore */ }
}

function togglePanel() {
    const fab = document.getElementById('aiBotFab');
    const panel = document.getElementById('aiBotPanel');
    const bubble = document.getElementById('aiBotBubble');

    panelOpen = !panelOpen;
    savePanelState();

    if (panelOpen) {
        fab.classList.add('open');
        panel.classList.add('open');
        bubble.classList.remove('show');
        if (currentTab === 'stats') loadStatsData();
        if (currentTab === 'tasks') renderPlanTab();
        if (currentTab === 'pomodoro') updatePomoDisplay();
    } else {
        fab.classList.remove('open');
        panel.classList.remove('open');
    }
}

function switchTab(tabName) {
    currentTab = tabName;
    savePanelState();
    document.querySelectorAll('.ai-panel-tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tabName));
    document.querySelectorAll('.ai-tab-pane').forEach(p => p.classList.remove('active'));
    const targetPane = document.getElementById(tabName === 'tasks' ? 'aiTabTasks' : tabName === 'stats' ? 'aiTabStats' : 'aiTabPomodoro');
    if (targetPane) targetPane.classList.add('active');

    if (tabName === 'stats') loadStatsData();
    if (tabName === 'tasks') renderPlanTab();
    if (tabName === 'pomodoro') updatePomoDisplay();
}

function showBubble(msg) {
    const bubble = document.getElementById('aiBotBubble');
    if (!bubble || panelOpen) return;
    bubble.textContent = msg;
    bubble.classList.add('show');
    clearTimeout(bubbleTimeout);
    bubbleTimeout = setTimeout(() => bubble.classList.remove('show'), 5000);
}

function scheduleBubbles() {
    setTimeout(() => {
        if (!panelOpen) showBubble(BUBBLE_MESSAGES[Math.floor(Math.random() * BUBBLE_MESSAGES.length)]);
    }, 6000);

    setInterval(() => {
        if (!panelOpen && Math.random() > 0.5) {
            showBubble(BUBBLE_MESSAGES[Math.floor(Math.random() * BUBBLE_MESSAGES.length)]);
        }
    }, 90000);
}

// ============================================================
// Tab1: 学习计划
// ============================================================

let _sharedForgettingItems = null; // 全局共享遗忘知识点缓存

async function getSharedForgettingItems() {
    if (_sharedForgettingItems !== null) return _sharedForgettingItems;
    try {
        const userId = getUserId();
        if (userId) {
            const data = await getLearningAnalytics(userId);
            if (data && data.forgetting_curve) {
                _sharedForgettingItems = data.forgetting_curve.filter(i => i.needs_review);
                return _sharedForgettingItems;
            }
        }
    } catch (e) { /* ignore */ }
    _sharedForgettingItems = [];
    return _sharedForgettingItems;
}

function getPlanStorageKey() {
    const userId = getUserId();
    const today = new Date().toISOString().slice(0, 10);
    return `softbei_plan_${userId}_${today}`;
}

function getTodayPlan() {
    const key = getPlanStorageKey();
    const raw = localStorage.getItem(key);
    if (!raw) return [];
    try { return JSON.parse(raw); } catch { return []; }
}

function saveTodayPlan(plans) {
    const key = getPlanStorageKey();
    localStorage.setItem(key, JSON.stringify(plans));
}

async function renderPlanTab() {
    const container = document.getElementById('aiTasksContent');
    if (!container) return;

    // 加载遗忘知识点（共享缓存）
    const forgettingItems = await getSharedForgettingItems();

    const plans = getTodayPlan();
    const total = plans.length;
    const done = plans.filter(p => p.done).length;
    const pct = total > 0 ? Math.round(done / total * 100) : 0;

    let html = '';

    // 进度概览
    html += `<div style="display:flex;align-items:center;gap:12px;padding:10px 12px;background:#FFF5EC;border-radius:12px;margin-bottom:14px;">
        <div style="position:relative;width:44px;height:44px;flex-shrink:0;">
            <svg viewBox="0 0 36 36" width="44" height="44" style="transform:rotate(-90deg);">
                <path fill="none" stroke="rgba(199,123,60,0.15)" stroke-width="3.5" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                <path fill="none" stroke="#C77B3C" stroke-width="3.5" stroke-linecap="round" stroke-dasharray="${pct}, 100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
            </svg>
            <span style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;color:#C77B3C;">${pct}%</span>
        </div>
        <div style="flex:1;">
            <div style="font-size:13px;font-weight:600;color:#1E1E2E;">今日计划</div>
            <div style="font-size:11px;color:#6B7280;margin-top:2px;">${total > 0 ? `已完成 ${done}/${total} 项` : '还没有计划，添加一个吧'}</div>
        </div>
    </div>`;

    // 计划列表
    if (total > 0) {
        html += '<div style="display:flex;flex-direction:column;gap:4px;margin-bottom:12px;">';
        plans.forEach((item, idx) => {
            html += `<div class="plan-item" data-idx="${idx}" style="display:flex;align-items:center;gap:8px;padding:8px 10px;border-radius:8px;background:${item.done ? '#F9FAFB' : '#fff'};border:1px solid #E5E7EB;cursor:pointer;transition:all 0.15s;">
                <div class="plan-check" style="width:18px;height:18px;border-radius:50%;border:2px solid ${item.done ? '#10B981' : '#D1D5DB'};display:flex;align-items:center;justify-content:center;flex-shrink:0;background:${item.done ? '#10B981' : 'transparent'};transition:all 0.2s;">
                    ${item.done ? '<svg width="10" height="10" viewBox="0 0 12 12" fill="none"><path d="M2 6l3 3 5-5" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>' : ''}
                </div>
                <span style="flex:1;font-size:12px;color:${item.done ? '#9CA3AF' : '#1E1E2E'};${item.done ? 'text-decoration:line-through;' : ''}overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${item.text}</span>
                <span class="plan-del" data-idx="${idx}" style="width:16px;height:16px;display:flex;align-items:center;justify-content:center;color:#D1D5DB;cursor:pointer;font-size:14px;border-radius:4px;">&times;</span>
            </div>`;
        });
        html += '</div>';
    } else {
        html += `<div style="text-align:center;padding:16px 12px;font-size:12px;color:#9CA3AF;background:#F9FAFB;border-radius:8px;margin-bottom:12px;">
            制定今天的学习计划，开启高效学习之旅
        </div>`;
    }

    // 自定义添加输入框
    html += `<div style="display:flex;gap:6px;margin-bottom:10px;">
        <input id="aiPlanInput" type="text" placeholder="输入学习计划..." maxlength="50" style="flex:1;border:1px solid #E5E7EB;border-radius:8px;padding:8px 10px;font-size:12px;outline:none;transition:border-color 0.2s;">
        <button id="aiPlanAddBtn" style="padding:8px 12px;border-radius:8px;border:none;background:linear-gradient(135deg,#C77B3C 0%,#A8652E 100%);color:#fff;font-size:12px;font-weight:500;cursor:pointer;white-space:nowrap;">添加</button>
    </div>`;

    // 选择复习知识点快捷添加
    const existingTexts = plans.map(p => p.text);
    const available = forgettingItems.filter(item => !existingTexts.includes(`复习：${item.kp_name}`));
    if (available.length > 0) {
        html += `<div style="padding:10px 12px;border-radius:10px;background:linear-gradient(135deg,#FFFBF5 0%,#FFF0E3 100%);border:1px solid #F5E6D3;margin-bottom:12px;">`;
        html += `<div style="font-size:10.5px;color:#A8652E;margin-bottom:8px;font-weight:500;display:flex;align-items:center;justify-content:space-between;"><span>\u{1F4CB} 待复习 ${forgettingItems.length} 项</span><span style="font-size:9.5px;color:#C4A882;font-weight:400;">点击加入计划</span></div>`;
        html += '<div style="display:flex;flex-direction:column;gap:4px;max-height:96px;overflow-y:auto;">';
        available.forEach(item => {
            const urgLabel = item.urgency === 'high' ? '紧急' : item.urgency === 'medium' ? '需关注' : '稳定';
            const urgBg = item.urgency === 'high' ? '#FEE2E2' : item.urgency === 'medium' ? '#FEF3C7' : '#D1FAE5';
            const urgColor = item.urgency === 'high' ? '#DC2626' : item.urgency === 'medium' ? '#D97706' : '#059669';
            html += `<div class="plan-kp-tag" data-kp="${item.kp_name}" style="display:flex;align-items:center;gap:8px;padding:6px 8px;border-radius:8px;background:#fff;cursor:pointer;transition:all 0.15s;border:1px solid #F0E6DA;" title="${item.kp_name}">
                <span style="flex:1;font-size:11px;color:#4B3621;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${item.kp_name}</span>
                <span style="font-size:9px;padding:1px 5px;border-radius:4px;background:${urgBg};color:${urgColor};white-space:nowrap;flex-shrink:0;">${urgLabel}</span>
                <span style="font-size:9.5px;color:#B8977A;white-space:nowrap;flex-shrink:0;">${item.days_since_last}天</span>
            </div>`;
        });
        html += '</div></div>';
    }



    container.innerHTML = html;
    if (typeof lucide !== 'undefined') lucide.createIcons();

    // 绑定事件
    bindPlanEvents();
}

function bindPlanEvents() {
    const container = document.getElementById('aiTasksContent');
    if (!container) return;

    // 自定义添加
    const addBtn = document.getElementById('aiPlanAddBtn');
    const input = document.getElementById('aiPlanInput');
    if (addBtn && input) {
        const doAdd = () => {
            const text = input.value.trim();
            if (!text) return;
            const plans = getTodayPlan();
            plans.push({ text, done: false });
            saveTodayPlan(plans);
            renderPlanTab();
        };
        addBtn.addEventListener('click', doAdd);
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') { e.preventDefault(); doAdd(); }
        });
        input.addEventListener('focus', () => input.style.borderColor = '#C77B3C');
        input.addEventListener('blur', () => input.style.borderColor = '#E5E7EB');
    }

    // 知识点标签点击添加
    container.querySelectorAll('.plan-kp-tag').forEach(tag => {
        tag.addEventListener('click', () => {
            const kpName = tag.dataset.kp;
            const plans = getTodayPlan();
            plans.push({ text: `复习：${kpName}`, done: false });
            saveTodayPlan(plans);
            renderPlanTab();
        });
    });

    // 勾选完成
    container.querySelectorAll('.plan-item').forEach(el => {
        el.addEventListener('click', (e) => {
            if (e.target.closest('.plan-del')) return;
            const idx = parseInt(el.dataset.idx);
            const plans = getTodayPlan();
            if (plans[idx]) {
                plans[idx].done = !plans[idx].done;
                saveTodayPlan(plans);
                renderPlanTab();
            }
        });
    });

    // 删除
    container.querySelectorAll('.plan-del').forEach(el => {
        el.addEventListener('click', (e) => {
            e.stopPropagation();
            const idx = parseInt(el.dataset.idx);
            const plans = getTodayPlan();
            plans.splice(idx, 1);
            saveTodayPlan(plans);
            renderPlanTab();
        });
    });
}

// ============================================================
// Tab2: 学情摘要
// ============================================================

let statsLoaded = false;

async function loadStatsData() {
    if (statsLoaded) return;
    const userId = getUserId();
    if (!userId) return;

    const container = document.getElementById('aiStatsContent');
    if (!container) return;

    try {
        const data = await getLearningAnalytics(userId);
        if (!data) {
            container.innerHTML = '<div class="ai-empty-state">暂无学习数据，快去学习吧！</div>';
            statsLoaded = true;
            return;
        }

        const behavior = data.learning_behavior || {};
        const forgetting = (data.forgetting_curve || []).filter(i => i.needs_review);
        // 同步到共享缓存
        _sharedForgettingItems = forgetting;
        const mastery = data.quiz_mastery || [];

        // 统计卡片
        let html = `<div class="ai-stats-grid">
            <div class="ai-stat-card">
                <div class="ai-stat-value">${behavior.streak_days || 0}</div>
                <div class="ai-stat-label">连续学习天数</div>
            </div>
            <div class="ai-stat-card">
                <div class="ai-stat-value">${behavior.total_actions || 0}</div>
                <div class="ai-stat-label">总学习次数</div>
            </div>
            <div class="ai-stat-card">
                <div class="ai-stat-value">${Math.round(behavior.total_minutes || 0)}</div>
                <div class="ai-stat-label">总学习(分钟)</div>
            </div>
            <div class="ai-stat-card">
                <div class="ai-stat-value">${behavior.active_days || 0}</div>
                <div class="ai-stat-label">活跃天数</div>
            </div>
        </div>`;

        // 掌握度 Top 5
        if (mastery.length > 0) {
            html += `<div class="ai-section-title">知识掌握度</div>`;
            const top5 = mastery.sort((a, b) => b.mastery_score - a.mastery_score).slice(0, 5);
            html += top5.map(m => {
                const pct = m.mastery_score;
                const color = pct >= 80 ? '#43A047' : pct >= 60 ? '#F57C00' : '#E53935';
                return `<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:12px;">
                    <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${m.kp_name}</span>
                    <div style="width:80px;height:6px;background:#E0E6ED;border-radius:3px;overflow:hidden;">
                        <div style="width:${pct}%;height:100%;background:${color};border-radius:3px;"></div>
                    </div>
                    <span style="width:30px;text-align:right;color:${color};font-weight:600;">${pct}%</span>
                </div>`;
            }).join('');
        }

        // 遗忘知识点复习
        if (forgetting.length > 0) {
            html += `<div class="ai-section-title">需要复习 (${forgetting.length})</div>`;
            html += '<div style="display:flex;flex-direction:column;gap:4px;max-height:200px;overflow-y:auto;">';
            html += forgetting.map(item => {
                const urgColor = item.urgency === 'high' ? '#EF4444' : item.urgency === 'medium' ? '#F59E0B' : '#10B981';
                return `<div style="display:flex;align-items:center;gap:8px;padding:8px 10px;border-radius:8px;border:1px solid #E5E7EB;background:#fff;border-left:3px solid ${urgColor};cursor:pointer;" onclick="window.location.href='generate.html?kp=${encodeURIComponent(item.kp_id || item.kp_name)}&type=doc'">
                    <span style="flex:1;font-size:12px;color:#1E1E2E;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${item.kp_name}</span>
                    <span style="font-size:11px;color:#9CA3AF;white-space:nowrap;">${item.days_since_last}天前</span>
                </div>`;
            }).join('');
            html += '</div>';
        } else {
            html += '<div class="ai-section-title">复习状态</div>';
            html += '<div style="font-size:12px;color:#6B7280;line-height:1.6;padding:8px 10px;background:#F0FDF4;border-radius:8px;">所有知识点记忆状态良好，继续保持！</div>';
        }

        container.innerHTML = html;
        statsLoaded = true;
    } catch (e) {
        console.warn('[assistant] 加载学情数据失败', e);
        container.innerHTML = '<div class="ai-empty-state">加载失败，请稍后重试</div>';
    }
}

// ============================================================
// Tab3: 番茄钟
// ============================================================

let pomoState = 'idle'; // idle | running | paused | resting
let pomoInterval = null;
let pomoRemaining = 25 * 60; // seconds
let pomoWorkMin = 25;
let pomoBreakMin = 5;
let pomoCount = 0;
let pomoTotalMin = 0;

// 番茄钟状态持久化
function getPomoStateKey() {
    const userId = getUserId();
    return `softbei_pomo_state_${userId}`;
}

function savePomoState() {
    const data = {
        state: pomoState,
        remaining: pomoRemaining,
        workMin: pomoWorkMin,
        breakMin: pomoBreakMin,
        timestamp: Date.now()
    };
    localStorage.setItem(getPomoStateKey(), JSON.stringify(data));
}

function clearPomoState() {
    localStorage.removeItem(getPomoStateKey());
}

function restorePomoState() {
    const raw = localStorage.getItem(getPomoStateKey());
    if (!raw) return;
    try {
        const data = JSON.parse(raw);
        pomoWorkMin = data.workMin || 25;
        pomoBreakMin = data.breakMin || 5;

        if (data.state === 'running' || data.state === 'resting') {
            // 计算离开期间流逝的时间
            const elapsed = Math.floor((Date.now() - data.timestamp) / 1000);
            pomoRemaining = (data.remaining || 0) - elapsed;

            if (data.state === 'running') {
                if (pomoRemaining <= 0) {
                    // 专注已完成，记录并进入休息
                    pomoCount++;
                    pomoTotalMin += pomoWorkMin;
                    savePomodoroStats();
                    pomoState = 'resting';
                    // 计算休息剩余时间
                    const restElapsed = Math.abs(pomoRemaining); // 超出的时间
                    pomoRemaining = pomoBreakMin * 60 - restElapsed;
                    if (pomoRemaining <= 0) {
                        // 休息也已结束
                        pomoState = 'idle';
                        pomoRemaining = pomoWorkMin * 60;
                        clearPomoState();
                    } else {
                        pomoInterval = setInterval(pomoTick, 1000);
                        savePomoState();
                    }
                } else {
                    // 还在专注中，继续计时
                    pomoState = 'running';
                    pomoInterval = setInterval(pomoTick, 1000);
                }
            } else if (data.state === 'resting') {
                if (pomoRemaining <= 0) {
                    // 休息已结束
                    pomoState = 'idle';
                    pomoRemaining = pomoWorkMin * 60;
                    clearPomoState();
                } else {
                    pomoState = 'resting';
                    pomoInterval = setInterval(pomoTick, 1000);
                }
            }
        } else if (data.state === 'paused') {
            // 暂停状态直接恢复
            pomoState = 'paused';
            pomoRemaining = data.remaining || pomoWorkMin * 60;
        }
    } catch (e) { /* ignore */ }
}

function updatePomoDisplay() {
    const timeEl = document.getElementById('aiPomoTime');
    const labelEl = document.getElementById('aiPomoLabel');
    const circleEl = document.getElementById('aiPomoCircle');
    const controlsEl = document.getElementById('aiPomoControls');

    if (!timeEl) return;

    const min = Math.floor(pomoRemaining / 60);
    const sec = pomoRemaining % 60;
    timeEl.textContent = `${String(min).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;

    circleEl.className = 'ai-pomo-circle' + (pomoState === 'running' ? ' running' : pomoState === 'resting' ? ' resting' : '');

    if (pomoState === 'resting') {
        labelEl.textContent = '休息时间';
    } else if (pomoState === 'running') {
        labelEl.textContent = '专注中...';
    } else if (pomoState === 'paused') {
        labelEl.textContent = '已暂停';
    } else {
        labelEl.textContent = '专注时间';
    }

    // 按钮
    if (pomoState === 'idle') {
        controlsEl.innerHTML = `<button class="ai-pomo-btn primary" id="aiPomoStart">开始专注</button>`;
    } else if (pomoState === 'running') {
        controlsEl.innerHTML = `
            <button class="ai-pomo-btn secondary" id="aiPomoPause">暂停</button>
            <button class="ai-pomo-btn danger" id="aiPomoStop">放弃</button>
        `;
    } else if (pomoState === 'paused') {
        controlsEl.innerHTML = `
            <button class="ai-pomo-btn primary" id="aiPomoResume">继续</button>
            <button class="ai-pomo-btn danger" id="aiPomoStop">放弃</button>
        `;
    } else if (pomoState === 'resting') {
        controlsEl.innerHTML = `<button class="ai-pomo-btn secondary" id="aiPomoSkipRest">跳过休息</button>`;
    }

    // 绑定事件
    const startBtn = document.getElementById('aiPomoStart');
    const pauseBtn = document.getElementById('aiPomoPause');
    const resumeBtn = document.getElementById('aiPomoResume');
    const stopBtn = document.getElementById('aiPomoStop');
    const skipBtn = document.getElementById('aiPomoSkipRest');

    if (startBtn) startBtn.onclick = startPomodoro;
    if (pauseBtn) pauseBtn.onclick = pausePomodoro;
    if (resumeBtn) resumeBtn.onclick = resumePomodoro;
    if (stopBtn) stopBtn.onclick = stopPomodoro;
    if (skipBtn) skipBtn.onclick = skipRest;

    // 统计
    document.getElementById('aiPomoCount').textContent = pomoCount;
    document.getElementById('aiPomoMinutes').textContent = pomoTotalMin;
}

function startPomodoro() {
    pomoWorkMin = parseInt(document.getElementById('aiPomoWorkMin')?.value) || 25;
    pomoBreakMin = parseInt(document.getElementById('aiPomoBreakMin')?.value) || 5;
    pomoRemaining = pomoWorkMin * 60;
    pomoState = 'running';
    savePomoState();
    updatePomoDisplay();
    pomoInterval = setInterval(pomoTick, 1000);
}

function pausePomodoro() {
    pomoState = 'paused';
    clearInterval(pomoInterval);
    savePomoState();
    updatePomoDisplay();
}

function resumePomodoro() {
    pomoState = 'running';
    savePomoState();
    updatePomoDisplay();
    pomoInterval = setInterval(pomoTick, 1000);
}

function stopPomodoro() {
    pomoState = 'idle';
    clearInterval(pomoInterval);
    pomoRemaining = pomoWorkMin * 60;
    clearPomoState();
    updatePomoDisplay();
}

function skipRest() {
    clearInterval(pomoInterval);
    pomoState = 'idle';
    pomoRemaining = pomoWorkMin * 60;
    clearPomoState();
    updatePomoDisplay();
}

function pomoTick() {
    pomoRemaining--;
    if (pomoRemaining <= 0) {
        clearInterval(pomoInterval);
        if (pomoState === 'running') {
            // 专注结束
            pomoCount++;
            pomoTotalMin += pomoWorkMin;
            savePomodoroStats();
            showBubble('太棒了！一个番茄钟完成！休息一下吧～');
            // 进入休息
            pomoState = 'resting';
            pomoRemaining = pomoBreakMin * 60;
            savePomoState();
            updatePomoDisplay();
            pomoInterval = setInterval(pomoTick, 1000);
            // 尝试通知
            tryNotification('番茄钟完成！', '你已经专注了 ' + pomoWorkMin + ' 分钟，休息一下吧');
        } else if (pomoState === 'resting') {
            // 休息结束
            pomoState = 'idle';
            pomoRemaining = pomoWorkMin * 60;
            clearPomoState();
            updatePomoDisplay();
            showBubble('休息结束，准备好开始下一轮了吗？');
            tryNotification('休息结束！', '该开始下一个番茄钟了');
        }
    } else {
        // 每10秒保存一次状态，避免频繁写入
        if (pomoRemaining % 10 === 0) savePomoState();
    }
    updatePomoDisplay();
}

function savePomodoroStats() {
    const userId = getUserId();
    const today = new Date().toISOString().slice(0, 10);
    const key = `softbei_pomo_${userId}_${today}`;
    const stored = JSON.parse(localStorage.getItem(key) || '{"count":0,"minutes":0}');
    stored.count++;
    stored.minutes += pomoWorkMin;
    localStorage.setItem(key, JSON.stringify(stored));
}

function loadPomodoroStats() {
    const userId = getUserId();
    const today = new Date().toISOString().slice(0, 10);
    const key = `softbei_pomo_${userId}_${today}`;
    const stored = JSON.parse(localStorage.getItem(key) || '{"count":0,"minutes":0}');
    pomoCount = stored.count;
    pomoTotalMin = stored.minutes;
}

function tryNotification(title, body) {
    if ('Notification' in window && Notification.permission === 'granted') {
        new Notification(title, { body });
    }
}

// ============================================================
// 每日学习提醒
// ============================================================

async function showDailyReminder() {
    console.log('[daily-reminder] 开始执行');
    const userId = getUserId();
    if (!userId) { console.log('[daily-reminder] 无userId，退出'); return; }

    const path = window.location.pathname;
    console.log('[daily-reminder] pathname:', path);
    const isIndex = path.endsWith('index.html') || path.endsWith('/') || path === '' || path.endsWith('/frontend/') || path.endsWith('/app/');
    if (!isIndex) { console.log('[daily-reminder] 非首页，退出'); return; }

    // 引导正在展示或即将开始，不弹每日提醒
    const guideVisible = document.querySelector('.guide-pending') || document.querySelector('.guide-welcome') || document.querySelector('.guide-overlay');
    if (guideVisible) { console.log('[daily-reminder] 引导展示中，退出'); return; }

    // 本次登录会话已经弹过，不再重复弹
    if (sessionStorage.getItem('softbei_daily_shown')) { console.log('[daily-reminder] 本次会话已弹过，退出'); return; }

    console.log('[daily-reminder] 条件检查通过，准备获取数据');

    let forgettingItems = [];
    let streakDays = 0;

    try {
        const analytics = await getLearningAnalytics(userId);
        if (analytics) {
            forgettingItems = (analytics.forgetting_curve || []).filter(i => i.needs_review);
            streakDays = analytics.learning_behavior?.streak_days || 0;
        }
    } catch (e) {
        console.warn('[daily-reminder] 获取提醒数据失败', e);
    }

    console.log('[daily-reminder] 数据获取完成，准备显示弹窗');

    const overlay = document.getElementById('dailyReminder');
    if (!overlay) { console.log('[daily-reminder] overlay元素不存在！'); return; }
    const greeting = document.getElementById('reminderGreeting');
    const timeEl = document.getElementById('reminderTime');
    const content = document.getElementById('reminderContent');
    const tip = document.getElementById('reminderTip');
    const motivation = document.getElementById('reminderMotivation');
    const iconEl = document.getElementById('reminderIcon');

    const now = new Date();
    const hour = now.getHours();
    const timeIcon = hour < 12 ? 'sunrise' : (hour < 18 ? 'sun' : 'moon');
    if (iconEl) iconEl.innerHTML = `<i data-lucide="${timeIcon}" style="width:40px;height:40px;color:var(--accent,#C77B3C);"></i>`;
    greeting.textContent = `${getTimeGreeting()}！`;
    timeEl.textContent = `${now.getFullYear()}年${now.getMonth()+1}月${now.getDate()}日 星期${['日','一','二','三','四','五','六'][now.getDay()]}`;

    let streakHTML = '';
    if (streakDays > 0) {
        streakHTML = `<div class="daily-reminder-section">
            <div class="daily-reminder-section-title"><i data-lucide="flame" style="width:14px;height:14px;vertical-align:-2px;margin-right:4px;color:#C77B3C;"></i>连续学习 ${streakDays} 天，真棒！</div>
        </div>`;
    }

    let reviewHTML = '';
    if (forgettingItems.length > 0) {
        const tags = forgettingItems.slice(0, 6).map(item => {
            const cls = item.urgency === 'high' ? 'urgent' : (item.urgency === 'medium' ? 'medium' : 'normal');
            return `<span class="daily-reminder-tag ${cls}" style="cursor:pointer;" onclick="window.location.href='generate.html?kp=${encodeURIComponent(item.kp_id || item.kp_name)}&type=doc'">${item.kp_name}</span>`;
        }).join('');
        reviewHTML = `<div class="daily-reminder-section">
            <div class="daily-reminder-section-title"><i data-lucide="clipboard-list" style="width:14px;height:14px;vertical-align:-2px;margin-right:4px;color:#C77B3C;"></i>以下知识点需要复习：</div>
            <div class="daily-reminder-tags">${tags}</div>
        </div>`;
    }

    content.innerHTML = streakHTML + reviewHTML;

    let tipText = '';
    if (forgettingItems.length > 0) {
        tipText = `根据艾宾浩斯遗忘曲线，你有 ${forgettingItems.length} 个知识点即将遗忘，建议今天花 10-15 分钟进行针对性复习。`;
    } else if (hour < 12) {
        tipText = '早上是记忆力最好的时段，适合学习新知识。试试开启番茄钟，专注25分钟！';
    } else if (hour < 18) {
        tipText = '下午适合做练习和复习。试试做几道测验题保持手感吧！';
    } else {
        tipText = '晚间适合轻度复习和总结。回顾今天的学习内容，巩固记忆。';
    }
    tip.textContent = tipText;
    motivation.textContent = `"${getMotivation()}"`;

    // 弹窗显示前再次检查引导是否在展示（因为上面有 await，引导可能在等待期间已开始）
    if (document.querySelector('.guide-pending') || document.querySelector('.guide-welcome') || document.querySelector('.guide-overlay')) return;

    // 标记本次会话已弹过
    sessionStorage.setItem('softbei_daily_shown', '1');

    overlay.classList.add('show');

    // 渲染弹窗内的 Lucide 图标
    if (typeof lucide !== 'undefined') lucide.createIcons();

    document.getElementById('reminderStartBtn').onclick = () => {
        overlay.classList.remove('show');
    };
    document.getElementById('reminderDismissBtn').onclick = () => {
        overlay.classList.remove('show');
    };
}

// ============================================================
// 初始化
// ============================================================

function init() {
    if (!isLoggedIn()) return;

    createAssistantDOM();

    // 悬浮按钮
    document.getElementById('aiBotFab').addEventListener('click', togglePanel);

    // 点击面板外关闭
    document.addEventListener('click', (e) => {
        if (!panelOpen) return;
        // 如果点击的元素已被移除（重新渲染导致），不关闭面板
        if (!e.target.isConnected) return;
        const panel = document.getElementById('aiBotPanel');
        const fab = document.getElementById('aiBotFab');
        if (!panel.contains(e.target) && !fab.contains(e.target)) {
            togglePanel();
        }
    });

    // Tab 切换
    document.querySelectorAll('.ai-panel-tab').forEach(tab => {
        tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });

    // 番茄钟：恢复状态 + 加载今日统计
    loadPomodoroStats();
    restorePomoState();
    updatePomoDisplay();

    // 恢复面板状态（Tab + 打开/关闭）
    restorePanelState();
    // 应用恢复的 Tab
    switchTab(currentTab);
    // 应用恢复的打开状态
    if (panelOpen) {
        const fab = document.getElementById('aiBotFab');
        const panel = document.getElementById('aiBotPanel');
        fab.classList.add('open');
        panel.classList.add('open');
    }

    // 预加载学习计划（仅当当前 Tab 是 tasks 时）
    if (currentTab === 'tasks') renderPlanTab();

    // 请求通知权限
    if ('Notification' in window && Notification.permission === 'default') {
        // 延迟请求，不打扰用户
        setTimeout(() => Notification.requestPermission(), 30000);
    }

    // 气泡
    scheduleBubbles();

    // 今日已学习时长计时
    initTodayTimer();

    // 每日提醒（仅首页）
    showDailyReminder();

    // 如果有遗忘知识点，显示红点提示
    checkBadge();
}

// ============================================================
// 今日已学习时长
// ============================================================

let todayTimerInterval = null;
let lastSavedSessionSec = 0; // 上次已保存到 localStorage 的本次会话秒数
const SESSION_START = Date.now();

function getTodayOnlineKey(userId) {
    const today = new Date().toISOString().slice(0, 10);
    return `softbei_online_${userId}_${today}`;
}

function getStoredTodaySeconds() {
    const userId = getUserId();
    if (!userId) return 0;
    const key = getTodayOnlineKey(userId);
    return parseInt(localStorage.getItem(key) || '0', 10);
}

function saveTodaySeconds(totalSec) {
    const userId = getUserId();
    if (!userId) return;
    const key = getTodayOnlineKey(userId);
    localStorage.setItem(key, String(totalSec));
}

function getCurrentSessionSeconds() {
    return Math.floor((Date.now() - SESSION_START) / 1000);
}

function formatOnlineTime(totalSec) {
    const totalMin = Math.floor(totalSec / 60);
    if (totalMin < 60) return `${totalMin}分钟`;
    const h = Math.floor(totalMin / 60);
    const m = totalMin % 60;
    return `${h}小时${m}分`;
}

function updateTodayTimeDisplay() {
    const el = document.getElementById('aiTodayTime');
    if (!el) return;
    const stored = getStoredTodaySeconds();
    const sessionDelta = getCurrentSessionSeconds() - lastSavedSessionSec;
    const total = stored + sessionDelta;
    el.textContent = formatOnlineTime(total);
}

function initTodayTimer() {
    // 立即更新一次
    updateTodayTimeDisplay();
    // 每 30 秒更新显示 + 增量保存
    todayTimerInterval = setInterval(() => {
        const sessionNow = getCurrentSessionSeconds();
        const delta = sessionNow - lastSavedSessionSec;
        if (delta > 0) {
            const stored = getStoredTodaySeconds();
            saveTodaySeconds(stored + delta);
            lastSavedSessionSec = sessionNow;
        }
        updateTodayTimeDisplay();
    }, 30000);

    // 页面关闭时保存增量
    window.addEventListener('beforeunload', () => {
        const sessionNow = getCurrentSessionSeconds();
        const delta = sessionNow - lastSavedSessionSec;
        if (delta > 0) {
            const stored = getStoredTodaySeconds();
            saveTodaySeconds(stored + delta);
        }
    });

    // 每小时休息提醒
    initHourlyRestReminder();
}

// ============================================================
// 每小时休息提醒
// ============================================================

let restReminderCreated = false;
let lastRestReminderHour = 0; // 上次弹出提醒时的小时数

function getRestReminderKey(userId) {
    const today = new Date().toISOString().slice(0, 10);
    return `softbei_rest_reminded_${userId}_${today}`;
}

function createRestReminderDOM() {
    if (restReminderCreated) return;
    restReminderCreated = true;
    const overlay = document.createElement('div');
    overlay.className = 'rest-reminder-overlay';
    overlay.id = 'restReminderOverlay';
    overlay.innerHTML = `
        <div class="rest-reminder-card">
            <div class="rest-reminder-icon"><i data-lucide="coffee" style="width:44px;height:44px;color:#C77B3C;"></i></div>
            <div class="rest-reminder-title">该休息一下啦！</div>
            <div class="rest-reminder-hours-label">今日已学习</div>
            <div class="rest-reminder-hours" id="restReminderHours">1<span class="rest-reminder-hours-unit">小时</span></div>
            <div class="rest-reminder-desc">连续学习时间较长，让眼睛和大脑休息一下吧！<br>起来走动走动、看看远处，身体好才能学习好。</div>
            <button class="rest-reminder-btn" id="restReminderBtn">好的，我去休息</button>
        </div>
    `;
    document.body.appendChild(overlay);
    if (typeof lucide !== 'undefined') lucide.createIcons();
    document.getElementById('restReminderBtn').addEventListener('click', () => {
        overlay.classList.remove('show');
    });
}

function showRestReminder(hours) {
    createRestReminderDOM();
    const hoursEl = document.getElementById('restReminderHours');
    if (hoursEl) hoursEl.innerHTML = `${hours}<span class="rest-reminder-hours-unit">小时</span>`;
    const overlay = document.getElementById('restReminderOverlay');
    if (overlay) {
        setTimeout(() => overlay.classList.add('show'), 200);
    }
}

function initHourlyRestReminder() {
    const userId = getUserId();
    if (!userId) return;

    // 读取今天已提醒过的小时数
    const key = getRestReminderKey(userId);
    lastRestReminderHour = parseInt(localStorage.getItem(key) || '0', 10);

    // 每 60 秒检测一次是否达到新的整小时
    setInterval(() => {
        const stored = getStoredTodaySeconds();
        const sessionDelta = getCurrentSessionSeconds() - lastSavedSessionSec;
        const totalSec = stored + sessionDelta;
        const totalHours = Math.floor(totalSec / 3600);

        // 如果达到了新的整小时且未提醒过
        if (totalHours > 0 && totalHours > lastRestReminderHour) {
            lastRestReminderHour = totalHours;
            localStorage.setItem(key, String(totalHours));
            showRestReminder(totalHours);
        }
    }, 60000); // 每分钟检测一次
}

async function checkBadge() {
    try {
        const userId = getUserId();
        const data = await getLearningAnalytics(userId);
        if (data && data.forgetting_curve) {
            const needsReview = data.forgetting_curve.filter(i => i.needs_review);
            if (needsReview.length > 0) {
                const badge = document.getElementById('aiBotBadge');
                if (badge) badge.classList.add('show');
            }
        }
    } catch (e) { /* ignore */ }
}

// 页面加载后初始化
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
