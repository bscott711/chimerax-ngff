"""GUI checks: File > Open Folder... lists the format, and dropping a store opens it.

Run with a display (a private Xvfb is enough):
  NGFF_GUI_STORE=<store> chimerax --exit --script tests/chimerax/gui_open.py
Builds the real Open Folder dialog and reads its format list, runs the command the
dialog issues on accept, and sends the main window a QDropEvent for the store folder,
which is what dragging the folder from a file manager does. Exits 1 on failure.
"""

import os
import sys

from chimerax.core.commands import run

store = os.environ["NGFF_GUI_STORE"]
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    msg = f"{'PASS' if ok else 'FAIL'}: {name}" + (f" ({detail})" if detail else "")
    print(msg)
    sys.__stdout__.write(msg + "\n")
    sys.__stdout__.flush()


def opened():
    return [m for m in session.models.list() if m.parent is session.models.scene_root_model
            and getattr(getattr(m, "opened_data_format", None), "name", "") == "OME-Zarr (NGFF)"]


fmt = session.data_formats["ngff"]
from chimerax.open_command import dialog  # noqa: E402

d = dialog.OpenFolderDialog(session.ui.main_window, session)
items = [d._format_selector.itemText(i) for i in range(d._format_selector.count())]
check("Open Folder lists the format", fmt.synopsis in items, f"{len(items)} formats")
d.deleteLater()

# What OpenFolderDialog.display() runs after the user picks the folder and format.
from chimerax.core.commands import FileNameArg  # noqa: E402

run(session, f"open {FileNameArg.unparse(store)} format {fmt.nicknames[0]}")
check("Open Folder command opens the store", len(opened()) == 1)
run(session, "close session")

from Qt.QtCore import QMimeData, QPointF, Qt, QUrl  # noqa: E402
from Qt.QtGui import QDropEvent  # noqa: E402

mime = QMimeData()
mime.setUrls([QUrl.fromLocalFile(store)])
event = QDropEvent(QPointF(10, 10), Qt.DropAction.CopyAction, mime,
                   Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
session.ui.main_window.dropEvent(event)
check("dropping the folder opens it", len(opened()) == 1)

if not all(results):
    os._exit(1)
