"""Main GUI mixin composition root."""

from __future__ import annotations


class MainGuiMixin:
    """Composition adapter properties to support legacy view-model routing."""

    @property
    def view_model(self):
        return self

    @property
    def cca_edits(self):
        return self

    @property
    def cca_workflows(self):
        return self

    @property
    def edit_id(self):
        return self

    @property
    def frame_metadata(self):
        return self

    @property
    def formatting(self):
        return self

    @property
    def geometry(self):
        return self

    @property
    def label_edits(self):
        return self

    @property
    def lineage(self):
        return self

    @property
    def model_registry(self):
        return self

    @property
    def object_counts(self):
        return self

    @property
    def points(self):
        return self

    @property
    def tables(self):
        return self

    @property
    def workspace(self):
        return self
