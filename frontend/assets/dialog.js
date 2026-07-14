/* dialog.js — 自定义确认对话框，替代原生 confirm()。
   三种模式 danger/normal/input，返回 Promise<boolean|string>。 */

let overlay = null;

function ensureOverlay() {
    if (!overlay || !document.body.contains(overlay)) {
        overlay = document.createElement('div');
        overlay.className = 'dialog-overlay';
        Object.assign(overlay.style, {
            position: 'fixed', inset: '0', zIndex: '9999',
            background: 'rgba(0,0,0,0.35)', backdropFilter: 'blur(2px)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            opacity: '0', pointerEvents: 'none', transition: 'opacity 0.2s ease',
        });
        document.body.appendChild(overlay);
    }
    return overlay;
}

const ICON_MAP = {
    danger: { icon: 'alert-triangle', color: '#EF4444', bg: '#FEF2F2' },
    normal: { icon: 'info', color: '#4F6EF7', bg: '#EEF1FE' },
    input:  { icon: 'alert-triangle', color: '#F59E0B', bg: '#FFFBEB' },
};

/**
 * @param {Object} opts
 * @param {'danger'|'normal'|'input'} opts.type
 * @param {string} opts.title  标题
 * @param {string} opts.desc  描述
 * @param {string} opts.confirmText  确认按钮文字
 * @param {string} opts.cancelText  取消按钮文字
 * @param {string} opts.inputPlaceholder  type=input 时的 placeholder
 * @param {string} opts.inputMatch  type=input 时需匹配的文字
 * @returns {Promise<boolean|string>} 确认=true/输入文字，取消=false
 */
export function showDialog(opts = {}) {
    const {
        type = 'normal',
        title = '确定执行此操作？',
        desc = '',
        confirmText = '确认',
        cancelText = '取消',
        inputPlaceholder = '',
        inputMatch = '',
    } = opts;

    return new Promise((resolve) => {
        const overlayEl = ensureOverlay();

        const cfg = ICON_MAP[type] || ICON_MAP.normal;

        const box = document.createElement('div');
        Object.assign(box.style, {
            background: '#fff', borderRadius: '16px', padding: '32px',
            maxWidth: '420px', width: '90%',
            boxShadow: '0 8px 40px rgba(0,0,0,0.15)',
            transform: 'scale(0.95)', transition: 'transform 0.25s cubic-bezier(0.34,1.56,0.64,1)',
            display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px',
        });

        const iconWrap = document.createElement('div');
        Object.assign(iconWrap.style, {
            width: '52px', height: '52px', borderRadius: '50%',
            background: cfg.bg, display: 'flex', alignItems: 'center', justifyContent: 'center',
        });
        iconWrap.innerHTML = '<i data-lucide="' + cfg.icon + '" style="width:26px;height:26px;color:' + cfg.color + ';"></i>';
        box.appendChild(iconWrap);

        const titleEl = document.createElement('div');
        titleEl.textContent = title;
        titleEl.style.cssText = 'font-size:17px;font-weight:600;color:#1E1E2E;text-align:center;';
        box.appendChild(titleEl);

        if (desc) {
            const descEl = document.createElement('div');
            descEl.textContent = desc;
            descEl.style.cssText = 'font-size:14px;color:#6B7280;text-align:left;line-height:1.8;white-space:pre-line;';
            box.appendChild(descEl);
        }

        let inputEl = null;
        if (type === 'input') {
            inputEl = document.createElement('input');
            inputEl.type = 'text';
            inputEl.placeholder = inputPlaceholder;
            Object.assign(inputEl.style, {
                width: '100%', padding: '12px 16px', border: '1px solid #E5E7EB',
                borderRadius: '10px', fontSize: '15px', outline: 'none',
                textAlign: 'center', fontFamily: 'inherit',
            });
            inputEl.addEventListener('input', () => {
                confirmBtn.disabled = inputMatch ? inputEl.value.trim() !== inputMatch : !inputEl.value.trim();
            });
            box.appendChild(inputEl);
        }

        const btnRow = document.createElement('div');
        btnRow.style.cssText = 'display:flex;gap:12px;width:100%;margin-top:4px;';

        const cancelBtn = document.createElement('button');
        cancelBtn.textContent = cancelText;
        Object.assign(cancelBtn.style, {
            flex: '1', height: '42px', borderRadius: '10px', border: '1px solid #E5E7EB',
            background: '#fff', color: '#4B5563', fontSize: '14px', fontWeight: '500',
            cursor: 'pointer', transition: 'all 0.15s',
        });
        cancelBtn.addEventListener('click', () => close(false));
        cancelBtn.addEventListener('mouseenter', () => { cancelBtn.style.background = '#F0F1F5'; });
        cancelBtn.addEventListener('mouseleave', () => { cancelBtn.style.background = '#fff'; });

        const confirmBtn = document.createElement('button');
        confirmBtn.textContent = confirmText;
        const confirmBg = type === 'danger'
            ? 'linear-gradient(135deg, #EF4444, #DC2626)'
            : 'linear-gradient(135deg, #4F6EF7, #7C5CFC)';
        Object.assign(confirmBtn.style, {
            flex: '1', height: '42px', borderRadius: '10px', border: 'none',
            background: confirmBg, color: '#fff', fontSize: '14px', fontWeight: '600',
            cursor: 'pointer', boxShadow: '0 2px 8px ' + (type === 'danger' ? 'rgba(239,68,68,0.3)' : 'rgba(79,110,247,0.3)'),
            transition: 'all 0.15s',
        });
        confirmBtn.addEventListener('click', () => {
            if (type === 'input' && inputEl) {
                close(inputEl.value.trim());
            } else {
                close(true);
            }
        });
        if (type === 'input') confirmBtn.disabled = true;

        btnRow.appendChild(cancelBtn);
        btnRow.appendChild(confirmBtn);
        box.appendChild(btnRow);

        overlayEl.innerHTML = '';
        overlayEl.appendChild(box);
        overlayEl.style.opacity = '1';
        overlayEl.style.pointerEvents = 'auto';

        requestAnimationFrame(() => { box.style.transform = 'scale(1)'; });

        const escHandler = (e) => { if (e.key === 'Escape') { close(false); document.removeEventListener('keydown', escHandler); } };
        document.addEventListener('keydown', escHandler);

        overlayEl.onclick = (e) => { if (e.target === overlayEl) close(false); };

        if (typeof lucide !== 'undefined') lucide.createIcons({ attrs: { 'data-lucide': '' } });

        function close(value) {
            box.style.transform = 'scale(0.95)';
            overlayEl.style.opacity = '0';
            overlayEl.style.pointerEvents = 'none';
            setTimeout(() => {
                overlayEl.innerHTML = '';
                if (inputEl) inputEl.value = '';
                resolve(value);
            }, 200);
        }
    });
}

export default showDialog;
