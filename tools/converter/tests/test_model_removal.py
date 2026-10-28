"""Model removal is reversible and never deletes imported or ambiguous files."""

import json

import pytest

from stark_convert.model_removal import (
    RemovalError,
    prepare_removal,
    removal_info,
    remove_model,
    restore_from_trash,
)

IDENTIFIER = "a" * 32


@pytest.fixture
def owned(tmp_path):
    root = tmp_path / "models"
    path = root / IDENTIFIER
    path.mkdir(parents=True)
    (path / "weights").write_bytes(b"synthetic weights")
    model = {"id": IDENTIFIER, "name": "Synthetic model", "path": str(path), "source": "huggingface:owner/model"}
    return root, path, model


def fake_trash(tmp_path):
    root = tmp_path / "synthetic-trash"
    root.mkdir(exist_ok=True)

    def move(path):
        target = root / path.name
        if target.exists():
            index = 2
            while (root / f"{path.name}-{index}").exists():
                index += 1
            target = root / f"{path.name}-{index}"
        path.rename(target)
        return target

    return root, move


def test_managed_model_moves_to_trash_before_registry_commit(owned, tmp_path):
    root, path, model = owned
    library = [model]
    trash, move = fake_trash(tmp_path)
    plan = prepare_removal(model, root, library)
    assert removal_info(model, root, library)["action"] == "trash"

    def commit():
        assert not path.exists() and (trash / IDENTIFIER / "weights").read_bytes() == b"synthetic weights"
        library.remove(model)

    result = remove_model(plan, commit, trash=move)
    assert result == {"model_id": IDENTIFIER, "action": "trash", "moved_to_trash": True}
    assert library == [] and (trash / IDENTIFIER).is_dir()


def test_local_raw_conversion_in_managed_folder_is_owned(owned, tmp_path):
    root, _path, model = owned
    model["source"] = "local:" + str(tmp_path / "original-raw-checkpoint")
    assert removal_info(model, root, [model])["action"] == "trash"


@pytest.mark.parametrize("location", ["external", "inside-managed", "missing"])
def test_registered_local_folder_only_unregisters_and_never_calls_trash(owned, tmp_path, location):
    root, path, model = owned
    if location == "external":
        path = tmp_path / "imported"
        path.mkdir()
        (path / "weights").write_bytes(b"original data")
    elif location == "missing":
        path = tmp_path / "missing-import"
    model.update(path=str(path), source="local:" + str(path))
    library = [model]
    before = list(path.iterdir()) if path.exists() else []
    plan = prepare_removal(model, root, library)
    assert plan.action == "unregister"

    def forbidden(_path):
        pytest.fail("attempted to move imported files")

    remove_model(plan, lambda: library.remove(model), trash=forbidden)
    assert library == []
    assert (list(path.iterdir()) if path.exists() else []) == before


@pytest.mark.parametrize(
    "case",
    [
        "root",
        "outside",
        "wrong-id",
        "alias",
        "dotdot-alias",
        "symlink",
        "shared",
        "shared-alias",
        "trash-root",
        "bad-source",
    ],
)
def test_ambiguous_managed_ownership_falls_back_to_library_only(owned, tmp_path, case):
    root, path, model = owned
    library = [model]
    if case == "root":
        model["path"] = str(root)
    elif case == "outside":
        model["path"] = str(tmp_path)
    elif case == "wrong-id":
        model["id"] = "b" * 32
    elif case == "alias":
        alias = tmp_path / "model-alias"
        alias.symlink_to(root, target_is_directory=True)
        model["path"] = str(alias / IDENTIFIER)
    elif case == "dotdot-alias":
        model["path"] = str(root / ".." / "models" / IDENTIFIER)
    elif case == "bad-source":
        model["source"] = "huggingface:"
    elif case == "symlink":
        target = tmp_path / "elsewhere"
        path.rename(target)
        path.symlink_to(target, target_is_directory=True)
    elif case in {"shared", "shared-alias"}:
        other = path
        if case == "shared-alias":
            other = tmp_path / "shared-alias"
            other.symlink_to(path, target_is_directory=True)
        library.append({**model, "id": "b" * 32, "path": str(other), "source": "local:" + str(other)})
    else:
        new_root = tmp_path / ".Trash" / "models"
        new_root.parent.mkdir()
        root.rename(new_root)
        root = new_root
        model["path"] = str(root / IDENTIFIER)
    assert removal_info(model, root, library)["action"] == "unregister"
    plan = prepare_removal(model, root, library)
    calls = []
    remove_model(
        plan, lambda: calls.append("committed"), trash=lambda _path: pytest.fail("ambiguous files were moved")
    )
    assert calls == ["committed"]


def test_missing_managed_folder_can_be_unregistered(owned):
    root, path, model = owned
    (path / "weights").unlink()
    path.rmdir()
    assert prepare_removal(model, root, [model]).action == "unregister"


@pytest.mark.parametrize("change", ["replace-folder", "root-alias", "new-shared-alias"])
def test_stale_plan_does_not_move_changed_or_newly_shared_folder(owned, tmp_path, change):
    root, path, model = owned
    extra = tmp_path / "competing-ref"
    library = [model, {**model, "id": "b" * 32, "path": str(extra), "source": "local:" + str(extra)}]
    plan = prepare_removal(model, root, library)
    original = tmp_path / "original"
    if change == "replace-folder":
        path.rename(original)
        path.mkdir()
    elif change == "root-alias":
        root.rename(original)
        root.symlink_to(original, target_is_directory=True)
    else:
        extra.symlink_to(path, target_is_directory=True)
    with pytest.raises(RemovalError, match="folder changed"):
        remove_model(
            plan, lambda: pytest.fail("stale plan committed"), trash=lambda _path: pytest.fail("stale files moved")
        )
    assert path.is_dir()


def test_persistence_failure_restores_trashed_folder_and_registry_callback_can_roll_back(owned, tmp_path):
    root, path, model = owned
    trash, move = fake_trash(tmp_path)
    library = [model]
    registry = tmp_path / "models.json"
    registry.write_text(json.dumps(library))
    before = registry.read_bytes()
    plan = prepare_removal(model, root, library)

    def fail_commit():
        library.remove(model)
        try:
            raise OSError("private synthetic filesystem failure")
        except OSError:
            library.append(model)
            raise

    with pytest.raises(RemovalError, match="files were kept"):
        remove_model(plan, fail_commit, trash=move)
    assert path.is_dir() and not (trash / IDENTIFIER).exists()
    assert (path / "weights").read_bytes() == b"synthetic weights"
    assert library == [model] and registry.read_bytes() == before


def test_rollback_never_overwrites_a_new_folder_at_original_path(owned, tmp_path):
    root, path, model = owned
    trash, move = fake_trash(tmp_path)
    plan = prepare_removal(model, root, [model])

    def fail_commit():
        path.mkdir()
        (path / "different").write_bytes(b"keep this new folder")
        raise OSError()

    with pytest.raises(RemovalError, match="model is in Trash"):
        remove_model(plan, fail_commit, trash=move)
    assert (path / "different").read_bytes() == b"keep this new folder"
    assert (trash / IDENTIFIER / "weights").read_bytes() == b"synthetic weights"


def test_trash_name_collision_preserves_both_folders(owned, tmp_path):
    root, path, model = owned
    trash, move = fake_trash(tmp_path)
    (trash / IDENTIFIER).mkdir()
    (trash / IDENTIFIER / "old").write_bytes(b"previous trash item")
    remove_model(prepare_removal(model, root, [model]), lambda: None, trash=move)
    assert (trash / IDENTIFIER / "old").read_bytes() == b"previous trash item"
    assert (trash / (IDENTIFIER + "-2") / "weights").read_bytes() == b"synthetic weights"


def test_trash_failure_keeps_files_and_never_commits(owned):
    root, path, model = owned

    def fail(_path):
        raise OSError("private synthetic failure")

    with pytest.raises(RemovalError, match="could not be removed"):
        remove_model(prepare_removal(model, root, [model]), lambda: pytest.fail("failed Trash committed"), trash=fail)
    assert (path / "weights").read_bytes() == b"synthetic weights"


def test_prepare_rejects_unknown_or_duplicate_selection(owned):
    root, _path, model = owned
    with pytest.raises(RemovalError, match="Choose a model"):
        prepare_removal(dict(model), root, [model])
    with pytest.raises(RemovalError, match="Choose a model"):
        prepare_removal(model, root, [model, dict(model)])


def test_restore_exclusive_preserves_existing_file(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    with pytest.raises(OSError):
        restore_from_trash(source, destination)
    assert source.is_dir() and destination.is_dir()


def test_fixed_error_never_contains_exception_details(owned):
    root, _path, model = owned

    def private_failure():
        raise RuntimeError("secret-like synthetic payload must not escape")

    model["source"] = "local:" + model["path"]
    with pytest.raises(RemovalError) as failure:
        remove_model(prepare_removal(model, root, [model]), private_failure)
    assert "secret-like" not in str(failure.value)


def test_invalid_record_metadata_is_library_only_without_exposing_details(tmp_path):
    model = {"id": "saved-model", "name": "Synthetic", "path": str(tmp_path / "missing"), "source": "custom:unknown"}
    assert removal_info(model, tmp_path / "models", [model])["action"] == "unregister"
    assert set(removal_info(model, tmp_path / "models", [model])) == {"schema_version", "action", "label", "message"}
