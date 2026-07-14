/* nav.js — 持久化顶栏导航（全站共享外壳）。
   topbar 位于 #view 之外常驻，页面切换仅原地 swap #view 内容并配合 startViewTransition 过渡。 */

const NAV = [
    { key: 'index',    href: 'index.html',    icon: 'layout-dashboard', label: '主页' },
    { key: 'chat',     href: 'chat.html',     icon: 'sparkles',         label: 'AI 对话' },
    { key: 'generate', href: 'generate.html', icon: 'wand-2',           label: '资源生成' },
    { key: 'library',  href: 'library.html',  icon: 'library',          label: '资源库' },
    { key: 'pathway',  href: 'pathway.html',  icon: 'route',            label: '学习规划' },
    { key: 'evaluate', href: 'evaluate.html', icon: 'clipboard-check',  label: '学习评估' },
    { key: 'history',  href: 'history.html',  icon: 'scroll-text',      label: '历史记录' },
    { key: 'profile',  href: 'profile.html',  icon: 'circle-user',      label: '个人中心' },
];

const _registry = new Map();
let _currentKey = null;

export function registerPage(key, init, destroy) {
    _registry.set(key, { init, destroy });
}

const _pageCache = new Map();

async function navigateTo(href, pushState = true) {
    const pageName = href.split('/').pop().replace('.html', '');

    if (!_registry.has(pageName)) {
        window.location.href = href;
        return;
    }

    let html = _pageCache.get(href);
    if (!html) {
        try {
            const resp = await fetch(href);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            html = await resp.text();
            _pageCache.set(href, html);
        } catch (err) {
            console.error('[Nav] fetch failed:', err);
            window.location.href = href;
            return;
        }
    }

    const doc = new DOMParser().parseFromString(html, 'text/html');
    const newView = doc.querySelector('#view');
    if (!newView) { window.location.href = href; return; }

    if (_currentKey && _registry.has(_currentKey)) {
        try { await _registry.get(_currentKey).destroy?.(); } catch (e) {
            console.error('[Nav] destroy error:', e);
        }
    }

    const viewEl = document.getElementById('view');
    const newTitle = doc.querySelector('title')?.textContent;

    const update = () => {
        viewEl.innerHTML = newView.innerHTML;
        if (newTitle) document.title = newTitle;
        if (typeof lucide !== 'undefined') lucide.createIcons();
    };

    if (document.startViewTransition) {
        try { await document.startViewTransition(() => update()).finished; } catch (_) { update(); }
    } else {
        update();
    }

    _currentKey = pageName;
    if (pushState) history.pushState({ key: pageName }, '', href);

    if (_registry.has(pageName)) {
        try { await _registry.get(pageName).init?.(); } catch (e) {
            console.error('[Nav] init error:', e);
        }
    }

    window.scrollTo({ top: 0, behavior: 'instant' });
    updateTopbarActive(pageName);
}

function buildTopbar(current) {
    const links = NAV.map(n => `
        <a href="${n.href}" class="topbar-link${n.key === current ? ' active' : ''}" data-key="${n.key}">
            <i data-lucide="${n.icon}"></i><span>${n.label}</span>
        </a>`).join('');

    return `
        <header class="topbar">
            <a href="index.html" class="topbar-brand">
                <span class="topbar-brand-mark"><i data-lucide="atom"></i></span>
                <span>智学工坊</span>
            </a>
            <nav class="topbar-nav">${links}</nav>
        </header>`;
}

function updateTopbarActive(current) {
    const el = document.querySelector('.topbar');
    if (!el) return;
    el.querySelectorAll('.topbar-link').forEach(a => {
        a.classList.toggle('active', a.dataset.key === current);
    });
}

let _topbarEl = null;

export function initSidebar(current) {
    if (_topbarEl) {
        updateTopbarActive(current);
        _currentKey = current;
        history.replaceState({ key: current }, '', window.location.href);
        return _topbarEl;
    }

    const host = document.createElement('div');
    host.innerHTML = buildTopbar(current);
    _topbarEl = host.firstElementChild;
    document.body.insertBefore(_topbarEl, document.body.firstChild);

    _topbarEl.addEventListener('click', (e) => {
        const link = e.target.closest('a[href]');
        if (!link) return;
        const href = link.getAttribute('href');
        if (!href || /^(https?:|#|javascript:)/.test(href)) return;
        e.preventDefault();
        navigateTo(href);
    });

    if (typeof lucide !== 'undefined') lucide.createIcons();

    _currentKey = current;
    history.replaceState({ key: current }, '', window.location.href);

    window.addEventListener('popstate', (e) => {
        if (e.state?.key) navigateTo(e.state.key + '.html', false);
    });

    return _topbarEl;
}

// Backward compatibility
export { initSidebar as initCapNav };
export default initSidebar;
