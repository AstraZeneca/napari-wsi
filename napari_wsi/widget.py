from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from magicgui.types import FileDialogMode
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FileEdit,
    Image,
    LineEdit,
    PushButton,
    RadioButtons,
)

from .backends.common import WSIReaderBackend
from .color_transform import ColorSpace
from .common import WSIStore, open_store

if TYPE_CHECKING:
    import napari


class PathChoice(StrEnum):
    PATH = "Path"
    URL = "URL"


class WSIReaderWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer") -> None:
        super().__init__()
        self._viewer = viewer
        self._store: WSIStore | None = None

        # Define the interactive widget elements.
        self._backend_edit = ComboBox(label="Backend", choices=WSIReaderBackend)
        self._color_space_edit = ComboBox(label="Color Space", choices=ColorSpace)
        self._choice_edit = RadioButtons(choices=PathChoice, orientation="horizontal")
        self._path_edit = FileEdit(label="Path")
        self._url_edit = LineEdit(label="URL")
        self._annotations_edit = CheckBox(label="Load Annotations", value=False)
        self._load_button = PushButton(name="Load")

        # Define the non-editable widget elements.
        self._slide_field = LineEdit(label="Slide ID", enabled=False, visible=False)
        self._patient_field = LineEdit(label="Patient ID", enabled=False, visible=False)
        self._label_image_field = Image(visible=False)

        # Set up custom event handlers.
        self._backend_edit.changed.connect(self._on_backend_changed)
        self._choice_edit.changed.connect(self._on_choice_changed)
        self._load_button.clicked.connect(self._on_load_button_clicked)

        # Set the initial values, triggering the changed events.
        self._choice_edit.value = PathChoice.PATH
        self._color_space_edit.value = ColorSpace.RAW
        self._backend_edit.value = WSIReaderBackend.WSIDICOM

        self.extend(
            [
                self._backend_edit,
                self._choice_edit,
                self._path_edit,
                self._url_edit,
                self._color_space_edit,
                self._annotations_edit,
                self._load_button,
                self._slide_field,
                self._patient_field,
                # We don't add the label image to the container, so that it will
                # be opened in a separate window, for better visibility.
            ]
        )

    @property
    def path(self) -> str | Path:
        if self._choice_edit.value == PathChoice.PATH:
            assert isinstance(self._path_edit.value, Path)
            return self._path_edit.value
        if self._choice_edit.value == PathChoice.URL:
            return self._url_edit.value
        raise ValueError(f"Invalid choice: {self._choice_edit.value}")

    @property
    def backend(self) -> WSIReaderBackend:
        return self._backend_edit.value

    @property
    def color_space(self) -> ColorSpace:
        return self._color_space_edit.value

    @property
    def layer_type(self) -> tuple[str, ...]:
        layer_type: tuple[str, ...] = ("image",)
        if self._annotations_edit.value:
            layer_type += ("shapes", "points")
        return layer_type

    def _on_backend_changed(self, value: WSIReaderBackend) -> None:
        if value == WSIReaderBackend.WSIDICOM:
            self._path_edit.mode = FileDialogMode.EXISTING_DIRECTORY
            self._choice_edit.visible = True
            self._annotations_edit.visible = True
        else:
            self._path_edit.mode = FileDialogMode.EXISTING_FILE
            self._choice_edit.visible = False
            self._choice_edit.value = PathChoice.PATH
            self._annotations_edit.visible = False

    def _on_choice_changed(self, value: str) -> None:
        self._path_edit.visible = value == PathChoice.PATH
        self._url_edit.visible = value == PathChoice.URL

    def _on_load_button_clicked(self) -> None:
        self._store = open_store(
            path=self.path, backend=self.backend, color_space=self.color_space
        )

        if self.backend == WSIReaderBackend.WSIDICOM:
            self._store.to_viewer(self._viewer, layer_type=self.layer_type)
        else:
            self._store.to_viewer(self._viewer)

        # Display the label image, if available.
        self._label_image_field.visible = False
        if self._store.label_image is not None:
            # Let's set the image width to the widget width, which should be reasonable.
            self._label_image_field.set_data(self._store.label_image, width=self.width)
            self._label_image_field.visible = True

        # Display additional metadata, if available.
        self._slide_field.visible = False
        self._patient_field.visible = False
        if self.backend == WSIReaderBackend.WSIDICOM:
            # The backend must be installed, or we could not have opened the store.
            from .backends.wsidicom import WSIDicomStore

            assert isinstance(self._store, WSIDicomStore)
            metadata = self._store.typed_metadata
            self._slide_field.value = metadata.slide.identifier or ""
            self._patient_field.value = metadata.patient.identifier or ""
            self._slide_field.visible = True
            self._patient_field.visible = True

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
        super().close()
