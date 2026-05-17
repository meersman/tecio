# Configuration file for the Sphinx documentation builder.
# ruff: noqa D100 D103

import shutil
import tomllib
from pathlib import Path

# -- Project information -----------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"

with PYPROJECT.open("rb") as f:
    data = tomllib.load(f)

project_info = data["project"]
project = project_info["name"]
release = project_info["version"]
copyright = "2026, John Meesman"
author = "John Meesman"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Napoleon settings (Google-style docstrings) -----------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True

# -- Autodoc settings --------------------------------------------------------

autodoc_default_options = {
    "show-inheritance": True,
    "member-order": "bysource",
}

autodoc_typehints = "none"
autodoc_class_signature = "separated"

# -- Autosummary settings ----------------------------------------------------

autosummary_generate = True
autosummary_generate_overwrite = True

# -- Intersphinx mapping -----------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- MyST parser -------------------------------------------------------------

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    "colon_fence",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = "tecio"
html_static_path = ["_static"]

html_theme_options = {
    "repository_url": "https://github.com/meersman/tecio",
    "use_repository_button": True,
    "use_download_button": False,
    "show_toc_level": 2,
    "navigation_with_keys": True,
}


# -- Demo auto-copy ----------------------------------------------------------

def _copy_demos(app) -> None:
    """Copy demo markdown and images into the docs source tree."""
    docs_dir = Path(__file__).parent
    demos_src = docs_dir.parent / "demos"

    if not demos_src.is_dir():
        return

    _image_exts = {".gif", ".png", ".jpg", ".jpeg", ".svg"}
    _text_exts = {".md"}

    for demo_dir in sorted(demos_src.iterdir()):
        if not demo_dir.is_dir():
            continue
        dst = docs_dir / "_demos" / demo_dir.name
        dst.mkdir(parents=True, exist_ok=True)
        for src_file in demo_dir.iterdir():
            if src_file.suffix in _text_exts | _image_exts:
                shutil.copy2(src_file, dst / src_file.name)


def setup(app):
    app.connect("builder-inited", _copy_demos)
