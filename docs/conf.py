# Configuration file for the Sphinx documentation builder.
# ruff: noqa D100 D103

import shutil
from pathlib import Path

import tecio

# -- Project information -----------------------------------------------------

project = "tecio"
copyright = "2026, John Meesman"
author = "John Meesman"
release = tecio.__version__

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
    # "bysource" requires accurate __module__ for source lookup; our __module__
    # overrides (e.g. Write.__module__ = "tecio.szl") point to __init__.py,
    # not the actual _write.py, so bysource silently fails.  Use alphabetical.
    "member-order": "alphabetical",
}

# "description" calls typing.get_type_hints() on every member.  With our
# __module__ overrides, get_type_hints() evaluates annotations against
# __init__.py's namespace, which does not import ValueLocation, Sequence,
# npt, Any, etc.  This raises NameError and Sphinx silently skips the member.
# "none" bypasses get_type_hints() entirely.  Types are still documented via
# Napoleon's Args/Returns sections from the Google-style docstrings.
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
