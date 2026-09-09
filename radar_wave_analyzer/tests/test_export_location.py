"""Export-directory regression tests."""
from radar_wave_analyzer.core import export_location  # noqa: E402


def test_export_dir_is_created_under_documents(monkeypatch, tmp_path):
    documents = tmp_path / 'Documents'
    monkeypatch.delenv(export_location.EXPORT_DIR_ENV, raising=False)
    monkeypatch.setattr(export_location, '_get_documents_dir', lambda: documents)

    result = export_location.get_export_dir()

    expected = documents / 'RadarWaveAnalyzer' / 'exports'
    assert result == str(expected.resolve())
    assert expected.is_dir()


def test_export_dir_environment_override_is_created(monkeypatch, tmp_path):
    custom_dir = tmp_path / 'managed-exports'
    monkeypatch.setenv(export_location.EXPORT_DIR_ENV, str(custom_dir))

    result = export_location.get_export_dir()

    assert result == str(custom_dir.resolve())
    assert custom_dir.is_dir()
