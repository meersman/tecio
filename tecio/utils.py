"""General utilities for tecplot."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from pathlib import Path


class TecplotNotFoundError(RuntimeError):
    """Raised when Tecplot installation or components cannot be located."""


class TecplotConversionError(RuntimeError):
    """Raised when Tecplot fails a command line conversion operation."""


_TEC_EXECUTABLE_ALIASES = (
    "tec360",
    "tec360EX",
    "tecplot",
)

_VERSION_REGEX = re.compile(r"(20\d{2})\s*[Rr]\s*(\d+)")


def _run_which(cmd: str) -> str | None:
    """Run `which cmd` explicitly as a fallback."""
    try:
        proc = subprocess.run(
            ["which", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except Exception:
        return None

    path = proc.stdout.strip()
    return path if path else None


def _find_tec_executable() -> str:
    """Locate Tecplot executable using multiple strategies."""
    # 1. shutil.which
    for name in _TEC_EXECUTABLE_ALIASES:
        exe = shutil.which(name)
        if exe:
            return os.path.realpath(exe)

    # 2. explicit `which`
    for name in _TEC_EXECUTABLE_ALIASES:
        exe = _run_which(name)
        if exe:
            return os.path.realpath(exe)

    # 3. macOS: search /Applications for Tecplot .app
    if platform.system() == "Darwin":
        applications_dir = "/Applications"

        if os.path.isdir(applications_dir):
            for root, dirs, _ in os.walk(applications_dir):
                for d in dirs:
                    if not d.endswith(".app"):
                        continue
                    if "Tecplot" not in d:
                        continue

                    app_path = os.path.join(root, d)
                    macos_dir = os.path.join(app_path, "Contents", "MacOS")
                    if not os.path.isdir(macos_dir):
                        continue

                    # Expected executable name matches app bundle name
                    app_base_name = os.path.splitext(d)[0]
                    expected_exe = os.path.join(macos_dir, app_base_name)

                    if os.path.isfile(expected_exe) and os.access(
                        expected_exe, os.X_OK
                    ):
                        return os.path.realpath(expected_exe)

                    # Fallback: choose executable files that are NOT libraries
                    for fname in os.listdir(macos_dir):
                        if fname.endswith((".so", ".dylib")):
                            continue
                        exe_path = os.path.join(macos_dir, fname)
                        if os.path.isfile(exe_path) and os.access(exe_path, os.X_OK):
                            return os.path.realpath(exe_path)

    raise TecplotNotFoundError(
        "Unable to locate Tecplot executable.\n"
        "Tried PATH lookup, `which`, and macOS /Applications scan."
    )


def _extract_version_from_path(path: str) -> str | None:
    """Extract Tecplot version YYYYR# from a filesystem path."""
    match = _VERSION_REGEX.search(path)
    if match:
        year, release = match.groups()
        return f"{year}R{release}"
    return None


def _extract_version_from_executable(exe_path: str) -> str | None:
    """Ask Tecplot executable for its version."""
    try:
        proc = subprocess.run(
            [exe_path, "-v"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except Exception:
        return None

    match = _VERSION_REGEX.search(proc.stdout or "")
    if match:
        year, release = match.groups()
        return f"{year}R{release}"

    return None


def get_tec_exe() -> str:
    """Return absolute path to Tecplot executable."""
    return _find_tec_executable()


def get_tec_bin() -> str:
    """Return Tecplot bin directory.

    - macOS: Contents/Frameworks
    - Linux: directory containing the executable
    """
    exe = get_tec_exe()

    if platform.system() == "Darwin":
        path = exe
        while path and not path.endswith(".app"):
            parent = os.path.dirname(path)
            if parent == path:
                break
            path = parent

        if not path.endswith(".app"):
            raise TecplotNotFoundError(
                f"Unable to locate Tecplot .app from executable:\n  {exe}"
            )

        frameworks = os.path.join(path, "Contents", "Frameworks")
        if not os.path.isdir(frameworks):
            raise TecplotNotFoundError(
                f"Tecplot Frameworks directory not found:\n  {frameworks}"
            )

        return frameworks

    # Linux
    bin_dir = os.path.dirname(exe)
    if not os.path.isdir(bin_dir):
        raise TecplotNotFoundError(f"Tecplot bin directory not found:\n  {bin_dir}")

    return bin_dir


def get_tecio_lib() -> str:
    """Return full path to the tecio shared library.

    - Linux: libtecio.so
    - macOS: libtecio.dylib.
    """
    bin_dir = get_tec_bin()

    libname = "libtecio.dylib" if platform.system() == "Darwin" else "libtecio.so"
    libpath = os.path.join(bin_dir, libname)

    if not os.path.isfile(libpath):
        raise TecplotNotFoundError(f"TecIO library not found:\n  {libpath}")

    return libpath


def get_tec_version() -> str:
    """Return Tecplot version formatted as YYYYR#."""
    exe = get_tec_exe()

    # 1. From executable path
    version = _extract_version_from_path(exe)
    if version:
        return version

    # 2. From bin directory
    bin_dir = get_tec_bin()
    version = _extract_version_from_path(bin_dir)
    if version:
        return version

    # 3. From executable output
    version = _extract_version_from_executable(exe)
    if version:
        return version

    raise TecplotNotFoundError(
        "Unable to determine Tecplot version.\n"
        f"Executable: {exe}\n"
        f"Bin dir:    {bin_dir}"
    )


def convert_to_szl(
    input_path: str | os.PathLike,
    output_path: str | os.PathLike | None = None,
) -> Path:
    """Convert a Tecplot data file to SZL (``.szplt``) format from the command line."""
    input_path = Path(input_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Resolve output path
    if output_path is None:
        szplt_path = input_path.with_suffix(".szplt")
    else:
        szplt_path = Path(output_path).resolve()

    # Return immediately if output already exists (no re-conversion)
    if szplt_path.is_file():
        return szplt_path

    # Locate the executable (raises TecplotNotFoundError if missing)
    exe = get_tec_exe()

    # Build the command
    # tec360ex -convert -quiet -nobatch <input> -o <output>
    cmd = [
        str(exe),
        "-convert",
        "-quiet",
        str(input_path),
        "-o",
        str(szplt_path),
    ]

    # Run the conversion
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            check=False,  # we check manually for a clearer error message
        )
    except OSError as exc:
        raise TecplotConversionError(
            f"Failed to launch Tecplot executable: {exe}\n{exc}"
        ) from exc

    # Check process exit status
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        stdout = result.stdout.decode(errors="replace").strip()
        detail = stderr or stdout or "(no output)"
        raise TecplotConversionError(
            f"Tecplot conversion failed (exit code {result.returncode}).\n"
            f"Command: {' '.join(cmd)}\n"
            f"Output:  {detail}"
        )

    # Verify the output file was actually created
    if not szplt_path.is_file():
        raise TecplotConversionError(
            f"Tecplot exited successfully but the expected output file was "
            f"not created: {szplt_path}\n"
            f"Command: {' '.join(cmd)}"
        )

    return szplt_path
