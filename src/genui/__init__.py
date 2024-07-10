
# inject the celery app into the genui package
from genui.settings.celery import celery_app
__all__ = ('celery_app',)

from genui.about import __version__ as genui_version
__version__ = genui_version
