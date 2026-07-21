/**
 * 文件上传即时加载遮罩 — 纯前端触发层。
 *
 * 目标：在用户「选定文件 / 拖放落下」的瞬间立即显示全屏遮罩动画，
 * 早于 dcc.Upload 的 base64 编码与服务端回调，提供严格同步的视觉反馈。
 *
 * 关闭由 callbacks.py 中的客户端回调在 store-data-loaded / cmp-state 变化时调用
 * window.hideUploadOverlay() 完成；此处另设安全超时强制隐藏。
 */
(function () {
  'use strict';

  var overlay = null;
  var hideTimer = null;
  var SAFETY_TIMEOUT_MS = 60000; // 异常卡死兜底

  function getOverlay() {
    if (!overlay) {
      overlay = document.getElementById('upload-overlay');
    }
    return overlay;
  }

  function showOverlay() {
    var el = getOverlay();
    if (!el) return;
    if (el.style.display === 'flex') return; // 已显示，防重复
    el.style.display = 'flex';
    if (hideTimer) clearTimeout(hideTimer);
    hideTimer = setTimeout(function () {
      hideOverlay();
      // eslint-disable-next-line no-console
      console.warn('[upload_overlay] 安全超时，强制隐藏遮罩');
    }, SAFETY_TIMEOUT_MS);
  }

  function hideOverlay() {
    var el = getOverlay();
    if (!el) return;
    el.style.display = 'none';
    if (hideTimer) {
      clearTimeout(hideTimer);
      hideTimer = null;
    }
  }

  // 暴露给 callbacks.py 的客户端回调
  window.showUploadOverlay = showOverlay;
  window.hideUploadOverlay = hideOverlay;

  // 绑定隐藏 file input 的 change 事件：OS 文件对话框返回即触发（早于编码）
  function bindInput(input) {
    if (!input || input.__uploadOverlayBound) return;
    input.__uploadOverlayBound = true;
    input.addEventListener('change', function () { 
      showOverlay();
    });
  }

  // 绑定拖放区 drop 事件（捕获阶段，确保早于 react-dropzone 处理）
  function bindDropZone(zone) {
    if (!zone || zone.__uploadOverlayDropBound) return;
    zone.__uploadOverlayDropBound = true;
    zone.addEventListener('drop', function () {
      showOverlay();
    }, true);
  }

  function scan() {
    // 所有上传组件内部的 file input
    var inputs = document.querySelectorAll('input[type="file"]');
    for (var i = 0; i < inputs.length; i++) bindInput(inputs[i]);

    // 上传拖放容器（dcc.Upload 渲染的 .upload-zone）
    var zones = document.querySelectorAll('.upload-zone');
    for (var j = 0; j < zones.length; j++) bindDropZone(zones[j]);
  }

  // 初始扫描
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', scan);
  } else {
    scan();
  }

  // 持续观察 DOM：dcc.Upload 的 input 可能在切换模式/雷达后重建
  if (typeof MutationObserver !== 'undefined') {
    var observer = new MutationObserver(function () {
      scan();
    });
    observer.observe(document.body, { childList: true, subtree: true });
  }
})();
