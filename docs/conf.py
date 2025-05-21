
import datetime
import os
import sys
# from pathlib import Path
# import sphinx_autosummary_accessors  # noqa

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyPRMS'
copyright = datetime.datetime.now().strftime("%Y")
author = 'USGS Developers and Community'

# sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(0, os.path.abspath("../pyPRMS"))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.todo',
              'sphinx_autodoc_typehints']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

todo_include_todos = True

autosummary_generate = True

autodoc_default_options = {
    "members": True,
    # "imported-members": True,
    "inherited-members": True,
    "undoc-members": True,
    "private-members": False,  #
    # "special-members": "",
    "exclude-members": "__init__",
}

autodoc_typehints = "description"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_title = 'pyPRMS'

html_static_path = ['_static']

html_context = {
    "github_user": "DOI-USGS",
    "github_repo": "pyPRMS",
    "github_version": "development",
    "doc_path": "docs",
}

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    # 'analytics_id': 'G-XXXXXXXXXX',  #  Provided by Google in your dashboard
    # 'analytics_anonymize_ip': False,
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': 'white',
    'flyout_display': 'hidden',
    'version_selector': True,
    'language_selector': True,
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}
# html_theme_options = dict(
#     # analytics_id=''  this is configured in rtfd.io
#     # canonical_url="",
#     repository_url="https://github.com/DOI-USGS/pyPRMS",
#     repository_branch="main",
#     path_to_docs="docs",
#     use_edit_page_button=True,
#     use_repository_button=True,
#     use_issues_button=True,
#     # home_page_in_toc=True,
#     # show_navbar_depth=1,
#     # show_toc_level=1,
# )

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "numba": ("https://numba.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "cftime": ("https://unidata.github.io/cftime", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

