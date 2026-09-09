/* clipboard_helper.js
 * 打包版（pywebview/WebView2）恢复复制能力：
 *  - WebView2 关闭了浏览器快捷键（Ctrl+C/Ctrl+A）与默认右键菜单，
 *    此脚本自行实现：Ctrl+C 复制选区、Ctrl+A 全选、右键弹出"复制/全选"菜单。
 *  - 开发模式（真浏览器）下同样可用，行为一致。
 */
(function () {
    'use strict';

    if (window.__clipboardHelperInstalled) return;
    window.__clipboardHelperInstalled = true;

    function isEditable(target) {
        if (!target) return null;
        var tag = (target.tagName || '').toLowerCase();
        if (tag === 'textarea' || tag === 'input') return target;
        if (target.isContentEditable) return target;
        return null;
    }

    function getSelectionText() {
        var sel = window.getSelection();
        return sel ? sel.toString() : '';
    }

    function copyTextToClipboard(text) {
        if (navigator.clipboard && navigator.clipboard.writeText) {
            return navigator.clipboard.writeText(text).catch(function () {
                return legacyCopy(text);
            });
        }
        return Promise.resolve(legacyCopy(text));
    }

    function legacyCopy(text) {
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.setAttribute('readonly', '');
        ta.style.position = 'fixed';
        ta.style.left = '-9999px';
        ta.style.top = '0';
        document.body.appendChild(ta);
        ta.focus();
        ta.select();
        var ok = false;
        try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
        document.body.removeChild(ta);
        return ok;
    }

    /* ---------- Ctrl+C / Ctrl+A ---------- */
    document.addEventListener('keydown', function (e) {
        if (!e.ctrlKey && !e.metaKey) return;
        var key = (e.key || '').toLowerCase();

        if (key === 'c' && !e.shiftKey && !e.altKey) {
            var editable = isEditable(e.target);
            var text = '';
            if (editable && typeof editable.value === 'string') {
                var s = editable.selectionStart, t = editable.selectionEnd;
                if (s !== null && t !== null && t > s) {
                    text = editable.value.slice(s, t);
                }
            } else {
                text = getSelectionText();
            }
            if (text) {
                e.preventDefault();
                copyTextToClipboard(text);
            }
            return; // 无选区时不拦截，交给页面其它逻辑
        }

        if (key === 'a' && !e.shiftKey && !e.altKey) {
            var el = isEditable(e.target);
            if (el && typeof el.select === 'function') {
                e.preventDefault();
                el.select();
            } else {
                e.preventDefault();
                var sel = window.getSelection();
                if (sel && document.body) {
                    var range = document.createRange();
                    range.selectNodeContents(document.body);
                    sel.removeAllRanges();
                    sel.addRange(range);
                }
            }
        }
    }, true);

    /* ---------- 自绘右键菜单 ---------- */
    var menuEl = null;

    function destroyMenu() {
        if (menuEl && menuEl.parentNode) {
            menuEl.parentNode.removeChild(menuEl);
        }
        menuEl = null;
        document.removeEventListener('mousedown', onDocMouseDown, true);
        window.removeEventListener('wheel', destroyMenu, true);
    }

    function onDocMouseDown(e) {
        if (menuEl && !menuEl.contains(e.target)) destroyMenu();
    }

    function menuItem(label, enabled, onClick) {
        var item = document.createElement('div');
        item.textContent = label;
        item.style.cssText =
            'padding:6px 18px;font-size:13px;color:#1e293b;cursor:pointer;' +
            'white-space:nowrap;line-height:1.4;';
        if (!enabled) {
            item.style.color = '#cbd5e1';
            item.style.cursor = 'default';
            return item;
        }
        item.onmouseenter = function () { item.style.background = '#eff6ff'; };
        item.onmouseleave = function () { item.style.background = 'transparent'; };
        item.onclick = function (ev) {
            ev.stopPropagation();
            destroyMenu();
            onClick();
        };
        return item;
    }

    function showMenu(x, y) {
        destroyMenu();
        menuEl = document.createElement('div');
        menuEl.style.cssText =
            'position:fixed;z-index:2147483647;background:#ffffff;' +
            'border:1px solid #e2e8f0;border-radius:6px;box-shadow:0 4px 16px rgba(15,23,42,.18);' +
            'padding:4px 0;user-select:none;font-family:inherit;';

        var hasText = getSelectionText().length > 0;
        menuEl.appendChild(menuItem('复制', hasText, function () {
            var text = getSelectionText();
            if (text) copyTextToClipboard(text);
        }));
        menuEl.appendChild(menuItem('全选', true, function () {
            var sel = window.getSelection();
            if (sel && document.body) {
                var range = document.createRange();
                range.selectNodeContents(document.body);
                sel.removeAllRanges();
                sel.addRange(range);
            }
        }));

        document.body.appendChild(menuEl);

        var rect = menuEl.getBoundingClientRect();
        var vw = window.innerWidth, vh = window.innerHeight;
        var px = Math.min(x, vw - rect.width - 4);
        var py = Math.min(y, vh - rect.height - 4);
        menuEl.style.left = Math.max(4, px) + 'px';
        menuEl.style.top = Math.max(4, py) + 'px';

        document.addEventListener('mousedown', onDocMouseDown, true);
        window.addEventListener('wheel', destroyMenu, true);
    }

    document.addEventListener('contextmenu', function (e) {
        // 输入框内保留原生行为基础：同样交给自绘菜单（WebView2 原生菜单已禁用）
        e.preventDefault();
        showMenu(e.clientX, e.clientY);
    });

    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape') destroyMenu();
    });
})();
