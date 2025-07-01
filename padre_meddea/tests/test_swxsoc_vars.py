import pytest

from padre_meddea.util.util import parse_science_filename, create_science_filename

TIME = "2024-04-06T12:06:21"
TIME_FORMATTED = "20240406T120621"

import os


def test_os_envs():
    assert os.environ["SWXSOC_MISSION"] == "padre"


# fmt: off
@pytest.mark.parametrize("instrument,time,level,version,filename", [
    ("meddea", TIME, "l1", "1.2.3", f"padre_meddea_l1_{TIME_FORMATTED}_v1.2.3.fits"),
    ("meddea", TIME, "l2", "2.4.5", f"padre_meddea_l2_{TIME_FORMATTED}_v2.4.5.fits"),
    ("sharp", TIME, "l2", "1.3.5", f"padre_sharp_l2_{TIME_FORMATTED}_v1.3.5.fits"),
    ("sharp", TIME, "l3", "2.4.5", f"padre_sharp_l3_{TIME_FORMATTED}_v2.4.5.fits"),
]
)
def test_parse_science_filename(instrument, time, level, version, filename):
    tokens = parse_science_filename(filename)
    assert tokens['instrument'] == instrument
    assert tokens['test'] == False
    assert tokens['level'] == level
    assert tokens['version'] == version
# fmt: on
