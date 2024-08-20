# Configuration file for the Sphinx documentation builder.

from datetime import datetime
import cellacdc

# -- Project information

project = 'Cell-ACDC'
author = cellacdc.__author__
copyright = f'{datetime.now():%Y}, {author}'

version = cellacdc.__version__
release = version


# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.email',
    'sphinx_tabs.tabs',
    'sphinx_copybutton'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

html_favicon = 'https://raw.githubusercontent.com/SchmollerLab/Cell_ACDC/main/cellacdc/resources/icon.ico'

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

# -- My css
html_static_path = ['static']
html_css_files = [
    'css/custom.css',
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = [
    '_build', 
    'Thumbs.db', 
    '.DS_Store', 
    '_gui_packages.rst',
    '_models_list.rst'
]