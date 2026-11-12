import pytest
from spikespy.MultiTraceView import MultiTraceView
from spikespy.ViewerState import ViewerState
from time import time
import logging


@pytest.fixture()
def state():
    import os
    fname = r"/Users/xs19785/Documents/Open Ephys/06-dec.h5"
    if not os.path.exists(fname):
        pytest.skip(f"Local test file {fname} not found")
    state = ViewerState()
    state.loadFile(fname)
    return state


def test_MultiTraceView_nextItem(qtbot, state: ViewerState):
    view = MultiTraceView(state=state)
    view.show()
    qtbot.addWidget(view)

    o = []
    for _ in range(5):
        t = time()
        state.setStimNo(state.stimno + 1)
        o.append(time() - t)
        qtbot.screenshot(view)
    av_time = sum(o) / len(o)
    logging.warning(f"average time: {av_time}")
    if av_time > 0.125:
        raise Exception(f"render time too long: {av_time}")
