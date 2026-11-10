from importlib.metadata import version as package_version


project = "XARP"
author = "caetanolab"
copyright = "2026, caetanolab"
release = package_version("xarp")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = []
html_context = {
    "display_github": True,
    "github_user": "HAL-UCSB",
    "github_repo": "xarp",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_preserve_defaults = True
python_use_unqualified_type_names = True

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True
