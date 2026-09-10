"""Resolve a persistent, user-writable directory for exported files."""
from __future__ import annotations

import os
from pathlib import Path

APP_EXPORT_SUBDIR = Path('RadarWaveAnalyzer') / 'exports'
EXPORT_DIR_ENV = 'RADAR_WAVE_EXPORT_DIR'


def _get_documents_dir() -> Path:
    """Return the OS Documents folder, including redirected Windows folders."""
    if os.name == 'nt':
        try:
            import ctypes

            # CSIDL_PERSONAL resolves localized and OneDrive-redirected Documents.
            buffer = ctypes.create_unicode_buffer(32768)
            result = ctypes.windll.shell32.SHGetFolderPathW(  # type: ignore[attr-defined]
                None, 5, None, 0, buffer,
            )
            if result == 0 and buffer.value:
                return Path(buffer.value)
        except (AttributeError, OSError, ValueError):
            pass

    return Path.home() / 'Documents'


def get_export_dir() -> str:
    """Create and return the persistent export directory.

    ``RADAR_WAVE_EXPORT_DIR`` can override the location for managed deployments
    and tests. By default files are stored under the current user's Documents
    folder, so a packaged executable never writes beside itself or into the
    temporary PyInstaller extraction directory.
    """
    configured = os.environ.get(EXPORT_DIR_ENV, '').strip()
    if configured:
        output_dir = Path(os.path.expandvars(configured)).expanduser()
    else:
        output_dir = _get_documents_dir() / APP_EXPORT_SUBDIR

    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir.resolve())
