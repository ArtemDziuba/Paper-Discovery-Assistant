"""First-use installation and subsequent offline loading of the pinned base encoder."""

import ctypes
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from pda import ranking


class FakeModel:
    def save_pretrained(self, path, **kwargs):
        folder = Path(path)
        folder.mkdir()
        (folder / "model.safetensors").write_bytes(b"synthetic model fixture")
        (folder / "config.json").write_text("{}", encoding="utf-8")


@pytest.fixture
def store(tmp_path, monkeypatch):
    folder = tmp_path / "models" / "all-MiniLM-L6-v2"
    monkeypatch.setattr(ranking, "MINILM_PATH", folder)
    return folder


def install_factory(monkeypatch, factory):
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=factory)
    )


def source_manifest(folder):
    folder.mkdir(parents=True)
    (folder / "pda_source.json").write_text(
        json.dumps({"model": ranking.MODEL, "revision": ranking.REVISION}), encoding="utf-8"
    )


def test_first_use_reuses_hub_cache_then_loads_project_folder_offline(store, monkeypatch, tmp_path):
    calls = []

    def factory(source, **kwargs):
        calls.append((source, kwargs))
        if len(calls) > 1:
            assert source == str(store), "Do not resolve a Hub model after local installation"
            assert kwargs["local_files_only"] is True
        return FakeModel()

    install_factory(monkeypatch, factory)
    ranking.SemanticEncoder("cpu")
    assert (store / "model.safetensors").is_file()
    assert json.loads((store / "pda_source.json").read_bytes())["revision"] == ranking.REVISION
    monkeypatch.chdir(tmp_path)
    ranking.SemanticEncoder("cpu")
    assert len(calls) == 2
    assert calls[0] == (
        ranking.MODEL,
        {"revision": ranking.REVISION, "device": "cpu", "local_files_only": True},
    )
    assert calls[1] == (str(store), {"device": "cpu", "local_files_only": True})


def test_missing_hub_cache_downloads_pinned_revision_once(store, monkeypatch):
    calls = []

    def factory(source, **kwargs):
        calls.append((source, kwargs))
        if source == ranking.MODEL and kwargs.get("local_files_only"):
            raise OSError("No cached weights")
        return FakeModel()

    install_factory(monkeypatch, factory)
    ranking.SemanticEncoder("cpu")
    ranking.SemanticEncoder("cpu")
    assert len(calls) == 3
    assert calls[1] == (ranking.MODEL, {"revision": ranking.REVISION, "device": "cpu"})
    assert calls[2][0] == str(store) and calls[2][1]["local_files_only"]


@pytest.mark.parametrize("failure", ["download", "save"])
def test_interrupted_install_leaves_no_ready_folder_and_can_be_retried(store, monkeypatch, failure):
    sibling = store.parent / "finetuned-minilm" / "model.safetensors"
    sibling.parent.mkdir(parents=True)
    sibling.write_bytes(b"existing fine-tuned weights")

    class BrokenSave(FakeModel):
        def save_pretrained(self, path, **kwargs):
            super().save_pretrained(path, **kwargs)
            raise OSError("Interrupted save")

    def broken_factory(source, **kwargs):
        if failure == "download":
            raise OSError("Download unavailable")
        return BrokenSave()

    install_factory(monkeypatch, broken_factory)
    with pytest.raises(OSError):
        ranking.SemanticEncoder("cpu")
    assert not store.exists()
    assert not list(store.parent.glob(".minilm-*"))
    assert sibling.read_bytes() == b"existing fine-tuned weights"
    install_factory(monkeypatch, lambda *args, **kwargs: FakeModel())
    ranking.SemanticEncoder("cpu")
    assert store.is_dir()


@pytest.mark.parametrize("manifest", [None, {"model": "other", "revision": "other"}])
def test_incomplete_or_wrong_revision_folder_does_not_trigger_network(store, monkeypatch, manifest):
    store.mkdir(parents=True)
    if manifest is not None:
        (store / "pda_source.json").write_text(json.dumps(manifest), encoding="utf-8")

    def forbidden(*args, **kwargs):
        pytest.fail("Do not download or load a different revision for an existing invalid folder")

    install_factory(monkeypatch, forbidden)
    with pytest.raises(RuntimeError, match="incomplete or has a different revision"):
        ranking.SemanticEncoder("cpu")
    assert store.exists()


def test_broken_local_weights_do_not_fall_back_to_download(store, monkeypatch):
    source_manifest(store)
    calls = []

    def factory(source, **kwargs):
        calls.append(source)
        assert source == str(store) and kwargs["local_files_only"]
        raise OSError("Broken local weights")

    install_factory(monkeypatch, factory)
    with pytest.raises(OSError, match="Broken local weights"):
        ranking.SemanticEncoder("cpu")
    assert calls == [str(store)]


def test_threads_install_model_once(store, monkeypatch):
    calls = []

    def factory(source, **kwargs):
        calls.append(source)
        return FakeModel()

    install_factory(monkeypatch, factory)
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(ranking.load_minilm, ["cpu", "cpu"]))
    assert calls.count(ranking.MODEL) == 1
    assert calls.count(str(store)) == 1


def test_finetuned_weights_do_not_initialize_the_base_model(store, tmp_path, monkeypatch):
    folder = tmp_path / "fine-tuned"
    folder.mkdir()
    (folder / "training_manifest.json").write_text(json.dumps({"run_id": "test-run"}))
    calls = []

    def factory(source, **kwargs):
        calls.append((source, kwargs))
        return FakeModel()

    install_factory(monkeypatch, factory)
    encoder = ranking.SemanticEncoder("cpu", model_path=folder)
    assert encoder.model_name == ranking.FINETUNED_MODEL
    assert calls == [(str(folder), {"device": "cpu", "local_files_only": True})]
    assert not store.exists()


def test_permission_denial_is_not_reported_as_a_revision_mismatch(store, monkeypatch):
    store.mkdir(parents=True)
    original_read_text = Path.read_text

    def denied(path, *args, **kwargs):
        if path == store / "pda_source.json":
            raise PermissionError("Windows account cannot read this file")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", denied)
    install_factory(
        monkeypatch, lambda *args, **kwargs: pytest.fail("Do not load or download weights")
    )
    with pytest.raises(RuntimeError, match="Access denied reading") as error:
        ranking.SemanticEncoder("cpu")
    assert isinstance(error.value.__cause__, PermissionError)
    assert "revision" not in str(error.value)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DACL inheritance regression")
def test_installed_model_keeps_project_permission_inheritance(store, monkeypatch):
    from ctypes import wintypes

    # The staging parent simulates a project directory whose users must retain access.
    store.parent.mkdir(parents=True)
    security = ctypes.WinDLL("advapi32", use_last_error=True)
    get_security = security.GetFileSecurityW
    get_security.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    get_security.restype = wintypes.BOOL
    get_control = security.GetSecurityDescriptorControl
    get_control.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.WORD),
        ctypes.POINTER(wintypes.DWORD),
    ]
    get_control.restype = wintypes.BOOL

    def inherits_permissions(path):
        size = wintypes.DWORD()
        get_security(str(path), 4, None, 0, ctypes.byref(size))
        descriptor = ctypes.create_string_buffer(size.value)
        assert get_security(str(path), 4, descriptor, size.value, ctypes.byref(size))
        control, revision = wintypes.WORD(), wintypes.DWORD()
        assert get_control(descriptor, ctypes.byref(control), ctypes.byref(revision))
        return not bool(control.value & 0x1000)  # SE_DACL_PROTECTED

    assert inherits_permissions(store.parent)
    install_factory(monkeypatch, lambda *args, **kwargs: FakeModel())
    ranking.SemanticEncoder("cpu")
    assert inherits_permissions(store)
    assert inherits_permissions(store / "pda_source.json")
    assert inherits_permissions(store / "model.safetensors")
