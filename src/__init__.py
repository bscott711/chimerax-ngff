"""ChimeraX-NGFF: open OME-Zarr (OME-NGFF) images with ChimeraX's own numpy and imagecodecs."""

from chimerax.core.toolshed import BundleAPI


class _NGFFBundleAPI(BundleAPI):
    api_version = 1

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        from chimerax.open_command import OpenerInfo

        class NGFFOpenerInfo(OpenerInfo):
            def open(self, session, data, file_name, **kw):
                from .open import open_ngff
                # ChimeraX does not convert the classifier's "batch:false" to a bool, so
                # pyproject-built openers run in batch mode: data is then a list of paths
                # and file_name is the open command's name option (or None).
                paths = data if isinstance(data, (list, tuple)) else [data]
                models, messages = [], []
                for path in paths:
                    name = file_name if len(paths) == 1 else None
                    m, msg = open_ngff(session, path, name, **kw)
                    models.extend(m)
                    messages.append(msg)
                return models, "\n".join(messages)

            @property
            def open_args(self):
                from chimerax.core.commands import StringArg
                return {"series": StringArg}

        return NGFFOpenerInfo()


bundle_api = _NGFFBundleAPI()
