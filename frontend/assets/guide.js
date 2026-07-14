/* guide.js — 新用户引导组件：首次登录弹出欢迎屏 + 11 步高亮引导。
   引导完成状态按用户 ID 记录在 localStorage。 */
(function () {
    function getUserId() {
        try {
            const raw = localStorage.getItem('user');
            if (raw) {
                const obj = JSON.parse(raw);
                return obj.user_id || obj.id || null;
            }
        } catch (e) {}
        return localStorage.getItem('user_id') || null;
    }

    const userId = getUserId();
    if (!userId) return;

    const guideKey = `softbei_guide_done_${userId}`;
    if (localStorage.getItem(guideKey)) return;

    // 立即创建隐藏标记元素，让 assistant.js 能检测到引导即将开始
    const marker = document.createElement('div');
    marker.className = 'guide-pending';
    marker.style.display = 'none';
    document.body.appendChild(marker);

    const steps = [
        {
            selector: '.topbar-nav',
            title: '导航栏',
            desc: '这是顶部导航栏，你可以快速切换到各个功能模块。接下来我们一个个介绍。',
            position: 'bottom'
        },
        {
            selector: '.home-stats',
            title: '学习数据概览',
            desc: '这里实时展示你的学习统计：连续学习天数、已掌握知识点、总学习时长。随着你的学习，数据会自动更新。',
            position: 'bottom'
        },
        {
            selector: '.home-heatmap',
            title: '学习热力图',
            desc: '这里以热力图的形式展示你每天的学习情况，颜色越深表示学习时间越长，让你直观感受学习节奏。',
            position: 'right'
        },
        {
            selector: '.home-panels',
            title: '学情分析面板',
            desc: '这里展示学习活跃度趋势、知识掌握度、遗忘预警和最近动态，帮助你全方位了解学习状况。',
            position: 'left'
        },
        {
            selector: '.topbar-link[data-key="chat"]',
            title: '智能对话',
            desc: 'AI 学习伴侣！你可以随时向 AI 提问、讨论知识点、让它帮你解答疑惑或整理笔记。',
            position: 'bottom'
        },
        {
            selector: '.topbar-link[data-key="generate"]',
            title: '资源生成',
            desc: '核心功能！选择知识点后，系统会为你自动生成学习文档、思维导图、测验题、代码示例等个性化资源。',
            position: 'bottom'
        },
        {
            selector: '.topbar-link[data-key="library"]',
            title: '资源库',
            desc: '所有生成的学习资源都保存在这里，支持分类浏览、搜索、预览和重新学习。',
            position: 'bottom'
        },
        {
            selector: '.topbar-link[data-key="pathway"]',
            title: '知识图谱',
            desc: '根据你的知识图谱，系统会规划个性化学习路径，帮你按顺序、有节奏地掌握知识。',
            position: 'bottom'
        },
        {
            selector: '.topbar-link[data-key="evaluate"]',
            title: '学习评估',
            desc: '定期评估你的学习效果！系统会分析薄弱知识点，并推荐针对性的复习计划。',
            position: 'bottom'
        },
        {
            selector: '.topbar-link[data-key="profile"]',
            title: '个人中心',
            desc: '在这里完善你的学习画像（专业、每日学习时间、认知风格等），系统会据此为你<strong>个性化推荐</strong>学习内容。',
            position: 'bottom'
        },
        {
            selector: null, // 居中展示
            title: '开始学习之旅！',
            desc: '引导完成！推荐你的下一步：<br><br>① 前往<strong>个人中心</strong>完善学习画像<br>② 到<strong>资源生成</strong>选择知识点开始学习<br>③ 学习后可到<strong>学习评估</strong>检测效果<br><br>祝你学习愉快！',
            position: 'center'
        }
    ];
    const style = document.createElement('style');
    style.textContent = `
        .guide-overlay {
            position: fixed; inset: 0; z-index: 99999;
            pointer-events: auto;
        }
        .guide-backdrop {
            position: fixed; inset: 0;
            background: rgba(0,0,0,0.55);
            transition: opacity 0.3s;
        }
        .guide-spotlight {
            position: absolute;
            border-radius: 12px;
            box-shadow: 0 0 0 9999px rgba(0,0,0,0.55);
            transition: all 0.35s cubic-bezier(0.4,0,0.2,1);
            z-index: 1;
        }
        .guide-tooltip {
            position: absolute;
            background: #fff;
            border-radius: 20px;
            padding: 28px;
            width: 360px;
            box-shadow: 0 16px 60px rgba(199, 123, 60, 0.18), 0 4px 12px rgba(0,0,0,0.06);
            z-index: 2;
            transition: all 0.35s cubic-bezier(0.4,0,0.2,1);
            border: 1px solid rgba(199, 123, 60, 0.08);
        }
        .guide-tooltip-title {
            font-size: 18px; font-weight: 700; color: #1E1E2E; margin-bottom: 10px;
        }
        .guide-tooltip-desc {
            font-size: 14px; color: #4B5563; line-height: 1.7; margin-bottom: 20px;
        }
        .guide-tooltip-desc strong {
            color: #C77B3C;
        }
        .guide-tooltip-actions {
            display: flex; gap: 10px; justify-content: flex-end; align-items: center;
        }
        .guide-btn {
            padding: 9px 22px; border-radius: 10px; font-size: 13px;
            font-weight: 600; cursor: pointer; border: none; transition: all 0.2s;
        }
        .guide-btn-primary {
            background: linear-gradient(135deg, #C77B3C 0%, #A8652E 100%);
            color: #fff;
            box-shadow: 0 4px 14px rgba(199, 123, 60, 0.3);
        }
        .guide-btn-primary:hover {
            box-shadow: 0 6px 20px rgba(168, 101, 46, 0.4);
            transform: translateY(-1px);
        }
        .guide-btn-skip { background: transparent; color: #9CA3AF; }
        .guide-btn-skip:hover { color: #6B7280; }
        .guide-dots {
            display: flex; gap: 6px; justify-content: center; margin-top: 16px;
        }
        .guide-dot {
            width: 8px; height: 8px; border-radius: 50%; background: #E5E7EB;
            transition: all 0.25s;
        }
        .guide-dot.active { background: linear-gradient(135deg, #C77B3C, #A8652E); transform: scale(1.2); }
        /* 欢迎屏 */
        .guide-welcome {
            position: fixed; inset: 0; z-index: 100000;
            display: flex; align-items: center; justify-content: center;
            background: rgba(0,0,0,0.45);
            backdrop-filter: blur(4px);
        }
        .guide-welcome-card {
            background: #fff; border-radius: 24px; padding: 44px 40px;
            text-align: center; max-width: 420px; width: 90%;
            box-shadow: 0 16px 60px rgba(199, 123, 60, 0.2), 0 4px 12px rgba(0,0,0,0.06);
            animation: guidePopIn 0.4s cubic-bezier(0.34,1.56,0.64,1);
        }
        @keyframes guidePopIn {
            from { transform: scale(0.9); opacity: 0; }
            to { transform: scale(1); opacity: 1; }
        }
        .guide-welcome-icon { display:flex;align-items:center;justify-content:center;margin-bottom: 16px; }
        .guide-welcome-title {
            font-size: 22px; font-weight: 700; color: #1E1E2E; margin-bottom: 10px;
            background: linear-gradient(135deg, #C77B3C, #A8652E);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .guide-welcome-desc { font-size: 14px; color: #4B5563; line-height: 1.7; margin-bottom: 28px; }
    `;
    document.head.appendChild(style);

    let currentStep = -1; // -1 = 欢迎屏

    function showWelcome() {
        const welcome = document.createElement('div');
        welcome.className = 'guide-welcome';
        welcome.id = 'guideWelcome';
        welcome.innerHTML = `
            <div class="guide-welcome-card">
                <div class="guide-welcome-icon"><i data-lucide="sparkles" style="width:48px;height:48px;color:#C77B3C;"></i></div>
                <div class="guide-welcome-title">欢迎使用智能学习助手！</div>
                <div class="guide-welcome-desc">这是你的专属学习平台，让我花 30 秒带你快速了解各个功能吧～</div>
                <div style="display:flex;gap:12px;justify-content:center;">
                    <button class="guide-btn guide-btn-skip" id="guideSkipBtn">跳过引导</button>
                    <button class="guide-btn guide-btn-primary" id="guideStartBtn">开始引导 →</button>
                </div>
            </div>
        `;
        document.body.appendChild(welcome);

        if (typeof lucide !== 'undefined') lucide.createIcons();

        document.getElementById('guideStartBtn').onclick = () => {
            welcome.remove();
            startGuide();
        };
        document.getElementById('guideSkipBtn').onclick = () => {
            welcome.remove();
            finishGuide();
        };
    }

    function startGuide() {
        const overlay = document.createElement('div');
        overlay.className = 'guide-overlay';
        overlay.id = 'guideOverlay';
        overlay.innerHTML = `
            <div class="guide-spotlight" id="guideSpotlight"></div>
            <div class="guide-tooltip" id="guideTooltip">
                <div class="guide-tooltip-title" id="guideTipTitle"></div>
                <div class="guide-tooltip-desc" id="guideTipDesc"></div>
                <div class="guide-dots" id="guideDots"></div>
                <div class="guide-tooltip-actions">
                    <button class="guide-btn guide-btn-skip" id="guideSkip2">跳过</button>
                    <button class="guide-btn guide-btn-primary" id="guideNext">下一步</button>
                </div>
            </div>
        `;
        document.body.appendChild(overlay);

        const dotsEl = document.getElementById('guideDots');
        dotsEl.innerHTML = steps.map((_, i) =>
            `<div class="guide-dot ${i === 0 ? 'active' : ''}" data-i="${i}"></div>`
        ).join('');

        document.getElementById('guideNext').onclick = nextStep;
        document.getElementById('guideSkip2').onclick = () => {
            document.getElementById('guideOverlay').remove();
            finishGuide();
        };

        currentStep = 0;
        showStep(0);
    }

    function nextStep() {
        currentStep++;
        if (currentStep >= steps.length) {
            document.getElementById('guideOverlay').remove();
            finishGuide();
        } else {
            showStep(currentStep);
        }
    }

    function showStep(idx) {
        const step = steps[idx];
        const spotlight = document.getElementById('guideSpotlight');
        const tooltip = document.getElementById('guideTooltip');
        const titleEl = document.getElementById('guideTipTitle');
        const descEl = document.getElementById('guideTipDesc');
        const nextBtn = document.getElementById('guideNext');

        titleEl.textContent = step.title;
        descEl.innerHTML = step.desc;
        nextBtn.textContent = idx === steps.length - 1 ? '完成 ✓' : '下一步 →';

        document.querySelectorAll('.guide-dot').forEach((dot, i) => {
            dot.classList.toggle('active', i === idx);
        });

        if (!step.selector) {
            spotlight.style.display = 'none';
            tooltip.style.left = '50%';
            tooltip.style.top = '50%';
            tooltip.style.transform = 'translate(-50%, -50%)';
        } else {
            spotlight.style.display = 'block';
            tooltip.style.transform = 'none';
            const el = document.querySelector(step.selector);
            if (el) {
                const rect = el.getBoundingClientRect();
                const pad = 8;
                spotlight.style.left = (rect.left - pad) + 'px';
                spotlight.style.top = (rect.top - pad) + 'px';
                spotlight.style.width = (rect.width + pad * 2) + 'px';
                spotlight.style.height = (rect.height + pad * 2) + 'px';

                positionTooltip(tooltip, rect, step.position);
            }
        }
    }

    function positionTooltip(tooltip, rect, pos) {
        const gap = 16;
        if (pos === 'right') {
            tooltip.style.left = (rect.right + gap) + 'px';
            tooltip.style.top = rect.top + 'px';
        } else if (pos === 'bottom') {
            tooltip.style.left = rect.left + 'px';
            tooltip.style.top = (rect.bottom + gap) + 'px';
        } else if (pos === 'top') {
            tooltip.style.left = rect.left + 'px';
            tooltip.style.top = (rect.top - gap - 180) + 'px';
        } else if (pos === 'left') {
            tooltip.style.left = (rect.left - 360 - gap) + 'px';
            tooltip.style.top = rect.top + 'px';
        }
        // 边界保护：定位后校正超出视口的 tooltip
        requestAnimationFrame(() => {
            const tRect = tooltip.getBoundingClientRect();
            if (tRect.left < 20) {
                tooltip.style.left = '20px';
            }
            if (tRect.right > window.innerWidth - 20) {
                tooltip.style.left = (window.innerWidth - tRect.width - 20) + 'px';
            }
            if (tRect.bottom > window.innerHeight - 20) {
                tooltip.style.top = (window.innerHeight - tRect.height - 20) + 'px';
            }
            if (tRect.top < 20) {
                tooltip.style.top = '20px';
            }
        });
    }

    function finishGuide() {
        localStorage.setItem(guideKey, '1');
        // 标记本次会话已弹过，避免引导完成后立即弹出每日提醒（下次登录时清除）
        sessionStorage.setItem('softbei_daily_shown', '1');
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => setTimeout(showWelcome, 500));
    } else {
        setTimeout(showWelcome, 500);
    }
})();

