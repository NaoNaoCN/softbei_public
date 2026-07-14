/* toast.js — 非阻塞通知系统，替代全站 alert()
   用法: showToast('消息', 'success', [{ label:'撤销', onClick:... }]) */

let toastContainer = null;
let toastTimer = null;

function ensureContainer() {
    if (!toastContainer || !document.body.contains(toastContainer)) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container';
        Object.assign(toastContainer.style, {
            position: 'fixed', top: '16px', right: '16px', zIndex: '10000',
            display: 'flex', flexDirection: 'column', gap: '8px',
            maxWidth: '400px', pointerEvents: 'none',
        });
        document.body.appendChild(toastContainer);
    }
    return toastContainer;
}

const TYPE_CONFIG = {
    success: { icon: 'check-circle', color: '#10B981', bg: '#ECFDF5', border: '#A7F3D0' },
    error:   { icon: 'x-circle', color: '#EF4444', bg: '#FEF2F2', border: '#FECACA' },
    warning: { icon: 'alert-triangle', color: '#F59E0B', bg: '#FFFBEB', border: '#FDE68A' },
    info:    { icon: 'info', color: '#4F6EF7', bg: '#EEF1FE', border: '#C7D2FE' },
};

/**
 * @param {string} message  通知文本
 * @param {'success'|'error'|'warning'|'info'} type
 * @param {Array<{label:string, onClick:Function}>} actions  操作按钮
 * @param {number} duration  自动消失毫秒数，0 表示不自动消失
 * @returns {HTMLElement} toast DOM
 */
export function showToast(message, type = 'info', actions = [], duration = 3500) {
    const cfg = TYPE_CONFIG[type] || TYPE_CONFIG.info;
    const container = ensureContainer();

    const el = document.createElement('div');
    el.className = 'toast-item';
    Object.assign(el.style, {
        background: '#fff', borderRadius: '12px', padding: '14px 18px',
        boxShadow: '0 4px 20px rgba(0,0,0,0.12), 0 1px 3px rgba(0,0,0,0.06)',
        borderLeft: '4px solid ' + cfg.color,
        display: 'flex', flexDirection: 'column', gap: '10px',
        pointerEvents: 'auto', opacity: '0', transform: 'translateX(40px)',
        transition: 'all 0.3s cubic-bezier(0.34,1.56,0.64,1)',
        position: 'relative', overflow: 'hidden',
    });

    const row = document.createElement('div');
    row.style.display = 'flex';
    row.style.alignItems = 'flex-start';
    row.style.gap = '10px';

    const iconWrap = document.createElement('div');
    iconWrap.innerHTML = '<i data-lucide="' + cfg.icon + '" style="width:18px;height:18px;color:' + cfg.color + ';"></i>';
    iconWrap.style.flexShrink = '0';
    iconWrap.style.marginTop = '1px';

    const msgEl = document.createElement('div');
    msgEl.textContent = message;
    msgEl.style.cssText = 'flex:1;font-size:14px;color:#1E1E2E;line-height:1.5;';

    const closeBtn = document.createElement('button');
    closeBtn.innerHTML = '<i data-lucide="x" style="width:14px;height:14px;"></i>';
    Object.assign(closeBtn.style, {
        background: 'none', border: 'none', cursor: 'pointer', color: '#9CA3AF',
        padding: '2px', borderRadius: '4px', flexShrink: '0',
    });
    closeBtn.addEventListener('click', () => dismiss(el));

    row.appendChild(iconWrap);
    row.appendChild(msgEl);
    row.appendChild(closeBtn);
    el.appendChild(row);

    if (actions.length > 0) {
        const actionsRow = document.createElement('div');
        actionsRow.style.cssText = 'display:flex;gap:8px;padding-left:28px;';
        actions.forEach(a => {
            const ab = document.createElement('button');
            ab.textContent = a.label;
            Object.assign(ab.style, {
                background: cfg.bg, border: '1px solid ' + cfg.border,
                color: cfg.color, borderRadius: '6px', padding: '4px 12px',
                fontSize: '12px', fontWeight: '600', cursor: 'pointer',
            });
            ab.addEventListener('click', () => {
                a.onClick();
                dismiss(el);
            });
            actionsRow.appendChild(ab);
        });
        el.appendChild(actionsRow);
    }

    if (duration > 0) {
        const bar = document.createElement('div');
        Object.assign(bar.style, {
            position: 'absolute', bottom: '0', left: '0', height: '3px',
            background: cfg.color, borderRadius: '0 0 0 3px',
            transition: 'width ' + duration + 'ms linear', width: '100%',
        });
        el.appendChild(bar);

        requestAnimationFrame(() => {
            requestAnimationFrame(() => { bar.style.width = '0%'; });
        });

        toastTimer = setTimeout(() => dismiss(el), duration);
    }

    container.appendChild(el);

    requestAnimationFrame(() => {
        el.style.opacity = '1';
        el.style.transform = 'translateX(0)';
    });

    if (typeof lucide !== 'undefined') lucide.createIcons({ icons: [cfg.icon, 'x'] });

    function dismiss(item) {
        clearTimeout(toastTimer);
        item.style.opacity = '0';
        item.style.transform = 'translateX(40px)';
        setTimeout(() => {
            if (item.parentNode) item.parentNode.removeChild(item);
        }, 300);
    }

    return el;
}

export default showToast;
