import os
from pathlib import Path

import jpype
import orekit_jpype as orekit

_orekit_data_configured = False


def initialize_orekit() -> None:
    global _orekit_data_configured

    if not os.environ.get("JAVA_HOME"):
        try:
            from jdk4py import JAVA_HOME

            os.environ["JAVA_HOME"] = str(JAVA_HOME)
        except ImportError:
            pass

    if not jpype.isJVMStarted():
        orekit.initVM()

    if not _orekit_data_configured:
        from orekit_jpype.pyhelpers import setup_orekit_curdir

        data_path = (
            Path(__file__).resolve().parent
            / "dynamics"
            / "orekit"
            / "orekit-data.zip"
        )
        setup_orekit_curdir(str(data_path))
        _orekit_data_configured = True
