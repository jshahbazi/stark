"""Reversible removal of owned model folders, or library-only removal of imports."""

from __future__ import annotations

import ctypes
import os
import re
import stat
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


class RemovalError(Exception):
    """A fixed, content-free removal failure."""


@dataclass(frozen=True)
class RemovalPlan:
    model_id: str
    action: str
    path: Path
    model_dir: Path
    root_identity: tuple[int, int] | None
    identity: tuple[int, int] | None
    other_paths: tuple[Path, ...]


def _absolute(value) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError()
    text = str(value)
    if not text or len(os.fsencode(text)) > 4096 or any(ord(char) < 32 or ord(char) == 127 for char in text):
        raise ValueError()
    path = Path(text)
    if not path.is_absolute() or path != Path(os.path.abspath(path)):
        raise ValueError()
    return path


def _identity(path: Path) -> tuple[int, int]:
    value = path.lstat()
    if not stat.S_ISDIR(value.st_mode) or path.resolve(strict=True) != path:
        raise ValueError()
    return value.st_dev, value.st_ino


def _same_path(first: Path, second: Path) -> bool:
    try:
        if first == second or first.resolve(strict=False) == second.resolve(strict=False):
            return True
        try:
            left, right = first.stat(), second.stat()
        except FileNotFoundError:
            return False
        return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)
    except (OSError, ValueError, RuntimeError):
        # An unreadable competing reference cannot authorize moving shared files.
        return True


def _owned(model: dict, model_dir: Path, models: list[dict]):
    identifier = model.get("id")
    if not isinstance(identifier, str) or re.fullmatch(r"[0-9a-f]{32}", identifier) is None:
        raise ValueError()
    root, path = _absolute(model_dir), _absolute(model.get("path"))
    if path.parent != root or path.name != identifier or any(part in {".Trash", ".Trashes"} for part in root.parts):
        raise ValueError()
    source = model.get("source")
    if not isinstance(source, str) or not source.startswith(("huggingface:", "local:")):
        raise ValueError()
    if (
        source.startswith("huggingface:")
        and re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}",
            source.removeprefix("huggingface:"),
        )
        is None
    ):
        raise ValueError()
    if source.startswith("local:") and _absolute(source.removeprefix("local:")) == path:
        # Already-converted local folders are registered in place, never owned.
        raise ValueError()
    root_identity, identity = _identity(root), _identity(path)
    others = []
    for entry in models:
        if entry is model or entry.get("id") == identifier:
            continue
        other = _absolute(entry.get("path"))
        if _same_path(path, other):
            raise ValueError()
        others.append(other)
    return root, path, root_identity, identity, tuple(others)


def removal_info(model: dict, model_dir: Path, models: list[dict]) -> dict:
    """Unsafe or unowned paths always offer library-only removal, never deletion."""
    try:
        _owned(model, model_dir, models)
        return {
            "schema_version": 1,
            "action": "trash",
            "label": "Move to Trash",
            "message": "Move this model to Trash and remove it from Stark. You can recover its files from Trash.",
        }
    except (OSError, ValueError, RuntimeError, TypeError, UnicodeError):
        return {
            "schema_version": 1,
            "action": "unregister",
            "label": "Remove from Stark",
            "message": "Remove this model from Stark. Its files will be left untouched.",
        }


def prepare_removal(model: dict, model_dir: Path, models: list[dict]) -> RemovalPlan:
    """Prepare under the library lock; the move rechecks directory identities."""
    try:
        identifier = model["id"]
        if (
            not isinstance(identifier, str)
            or not 0 < len(identifier) <= 128
            or sum(entry.get("id") == identifier for entry in models) != 1
            or not any(entry is model for entry in models)
        ):
            raise ValueError()
        # Unregistering an unsafe legacy route does not access that route.
        path, root = Path(model["path"]), Path(model_dir)
        if removal_info(model, root, models)["action"] == "trash":
            root, path, root_identity, identity, others = _owned(model, root, models)
            return RemovalPlan(identifier, "trash", path, root, root_identity, identity, others)
        return RemovalPlan(identifier, "unregister", path, root, None, None, ())
    except (KeyError, OSError, ValueError, RuntimeError, TypeError, UnicodeError):
        raise RemovalError("Choose a model from your library to remove.") from None


def _validate_move(plan: RemovalPlan) -> None:
    if (
        plan.action != "trash"
        or plan.path.parent != plan.model_dir
        or plan.path.name != plan.model_id
        or _identity(plan.model_dir) != plan.root_identity
        or _identity(plan.path) != plan.identity
        or any(_same_path(plan.path, other) for other in plan.other_paths)
    ):
        raise ValueError()


def system_trash(path: Path) -> Path:
    """Use macOS Foundation's recoverable Trash API without a shell or UI script."""
    if sys.platform != "darwin":
        raise RemovalError("Moving model files to Trash requires macOS.")
    pool = None
    try:
        ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
        objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        address = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value

        def method(result, *arguments):
            return ctypes.CFUNCTYPE(result, ctypes.c_void_p, ctypes.c_void_p, *arguments)(address)

        selector = objc.sel_registerName
        send = method(ctypes.c_void_p)
        send_object = method(ctypes.c_void_p, ctypes.c_void_p)
        pool = send(objc.objc_getClass(b"NSAutoreleasePool"), selector(b"new"))
        text = method(ctypes.c_void_p, ctypes.c_char_p)(
            objc.objc_getClass(b"NSString"), selector(b"stringWithUTF8String:"), os.fsencode(path)
        )
        url = send_object(objc.objc_getClass(b"NSURL"), selector(b"fileURLWithPath:"), text)
        manager = send(objc.objc_getClass(b"NSFileManager"), selector(b"defaultManager"))
        destination, error = ctypes.c_void_p(), ctypes.c_void_p()
        success = method(
            ctypes.c_bool, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)
        )(
            manager,
            selector(b"trashItemAtURL:resultingItemURL:error:"),
            url,
            ctypes.byref(destination),
            ctypes.byref(error),
        )
        if not success or not destination.value:
            raise ValueError()
        destination_text = send(destination.value, selector(b"path"))
        encoded = method(ctypes.c_char_p)(destination_text, selector(b"UTF8String"))
        if not encoded:
            raise ValueError()
        return _absolute(Path(os.fsdecode(encoded)))
    except (OSError, ValueError, TypeError, UnicodeError, AttributeError):
        raise RemovalError("Could not move this model to Trash. Check folder permissions and try again.") from None
    finally:
        if pool:
            method(None)(pool, selector(b"drain"))


def _rename_exclusive(source: Path, destination: Path) -> None:
    """Restore atomically without overwriting anything created at the old path."""
    if sys.platform == "darwin":
        library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        operation = library.renamex_np
        operation.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        operation.restype = ctypes.c_int
        result = operation(os.fsencode(source), os.fsencode(destination), 0x00000004)
    elif sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        operation = getattr(library, "renameat2", None)
        if operation is None:
            raise OSError()
        operation.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        operation.restype = ctypes.c_int
        result = operation(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    else:
        raise OSError()
    if result != 0:
        raise OSError()


def restore_from_trash(source: Path, destination: Path) -> None:
    _rename_exclusive(source, destination)


def remove_model(
    plan: RemovalPlan,
    commit: Callable[[], None],
    *,
    trash: Callable[[Path], Path] = system_trash,
    restore: Callable[[Path, Path], None] = restore_from_trash,
) -> dict:
    """Move owned files, commit library mutation, and restore on commit failure.

    The caller holds its library operation/lock and stops only the removed
    model's worker first. Its commit callback must restore the original in-memory
    registry if persistence fails. Imported folders are never moved or deleted.
    """
    if not isinstance(plan, RemovalPlan) or plan.action not in {"trash", "unregister"}:
        raise RemovalError("Choose a model from your library to remove.")
    destination = None
    if plan.action == "trash":
        try:
            _validate_move(plan)
        except (OSError, ValueError, RuntimeError, TypeError):
            raise RemovalError("The model folder changed. Refresh your library before removing it.") from None
        try:
            destination = _absolute(trash(plan.path))
            if _identity(destination) != plan.identity or plan.path.exists() or plan.path.is_symlink():
                raise ValueError()
        except RemovalError:
            raise
        except (OSError, ValueError, RuntimeError, TypeError, UnicodeError):
            raise RemovalError("The model could not be removed. Check Trash before trying again.") from None
    try:
        commit()
    except BaseException:
        if destination is not None:
            try:
                if _identity(plan.model_dir) != plan.root_identity or _identity(destination) != plan.identity:
                    raise ValueError()
                restore(destination, plan.path)
                if _identity(plan.path) != plan.identity:
                    raise ValueError()
            except BaseException:
                raise RemovalError(
                    "The library could not be updated and the model is in Trash. "
                    "Restore it from Trash before retrying."
                ) from None
        raise RemovalError("The model library could not be updated. The model files were kept.") from None
    return {"model_id": plan.model_id, "action": plan.action, "moved_to_trash": destination is not None}
