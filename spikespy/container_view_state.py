"""Per-channel-group viewing state.

A ``ChannelGroupContainerView`` hosts several views that share the data in the
application's ``ViewerState`` but should be able to browse stimuli
independently. ``ContainerViewerState`` forwards everything to the real state
except the currently displayed stimulus index and the "follow live events"
flag, which are local to the container.
"""

from typing import Any, Dict, Optional

from PySide6.QtCore import QObject, Signal


class ContainerViewerState(QObject):
    """A thin, per-container overlay on a shared :class:`ViewerState`."""

    onStimNoChange = Signal(int)
    onFollowLiveChanged = Signal(bool)
    onFollowLiveEventsChanged = Signal(bool)
    onContainerVisualSettingsChanged = Signal(dict)

    def __init__(self, base, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._base = base
        self._base_connections = []
        self._stimno = int(getattr(base, "stimno", 0) or 0)
        self._follow_live = bool(getattr(base, "follow_live", True))
        self.visual_settings: Dict[str, Any] = {
            "x_range": None,
            "y_range": None,
            "scale_mode": "std",
            "color_scale": (0.5, 2.0),
            "trace_scale": 1.0,
            "show_track_window": True,
            "show_template": True,
        }
        self._connect_base()

    # ------------------------------------------------------------------
    # Base-state plumbing
    # ------------------------------------------------------------------
    def _connect_base(self):
        if self._base is None:
            return
        try:
            self._base.onNewSweep.connect(self._on_new_sweep)
            self._base_connections.append((self._base.onNewSweep, self._on_new_sweep))
        except (AttributeError, TypeError, RuntimeError):
            pass
        try:
            self._base.onLoadNewFile.connect(self._on_data_reloaded)
            self._base_connections.append((self._base.onLoadNewFile, self._on_data_reloaded))
        except (AttributeError, TypeError, RuntimeError):
            pass

    def _disconnect_base(self):
        for sig, slot in self._base_connections:
            try:
                sig.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self._base_connections.clear()

    def rebase(self, base):
        """Point this overlay at a new backing state, keeping local follow mode."""
        if base is self._base:
            return
        self._disconnect_base()
        self._base = base
        self._stimno = int(getattr(base, "stimno", 0) or 0)
        self._connect_base()
        self.onStimNoChange.emit(self._stimno)

    def _on_new_sweep(self, stimno: int):
        if self._follow_live:
            self._set_stimno(stimno)

    def _on_data_reloaded(self):
        latest = self.get_latest_event_idx()
        if latest is None:
            return
        self._set_stimno(latest if self._follow_live else min(self._stimno, latest))

    def __getattr__(self, name: str) -> Any:
        try:
            base = object.__getattribute__(self, "_base")
        except AttributeError as exc:
            raise AttributeError(name) from exc
        return getattr(base, name)

    # ------------------------------------------------------------------
    # Local visual settings
    # ------------------------------------------------------------------
    def get_visual_settings(self) -> Dict[str, Any]:
        return dict(self.visual_settings)

    def update_visual_settings(self, **kwargs) -> Dict[str, Any]:
        self.visual_settings.update(kwargs)
        updated = dict(self.visual_settings)
        self.onContainerVisualSettingsChanged.emit(updated)
        return updated

    # ------------------------------------------------------------------
    # Local stimulus index / follow mode
    # ------------------------------------------------------------------
    @property
    def stimno(self) -> int:
        return self._stimno

    @stimno.setter
    def stimno(self, value: int):
        self._set_stimno(value)

    @property
    def follow_live(self) -> bool:
        return self._follow_live

    @follow_live.setter
    def follow_live(self, value: bool):
        if value:
            self.set_follow_live(True)
        elif self._follow_live:
            self._follow_live = False
            self.onFollowLiveChanged.emit(False)
            self.onFollowLiveEventsChanged.emit(False)

    def _clamp_stimno(self, stimno: int) -> int:
        event_signal = getattr(self._base, "event_signal", None)
        if event_signal is not None and len(event_signal) > 0:
            return max(0, min(int(stimno), len(event_signal) - 1))
        return max(0, int(stimno))

    def _set_stimno(self, stimno: int, emit: bool = True):
        new_stim = self._clamp_stimno(stimno)
        if new_stim == self._stimno:
            return
        self._stimno = new_stim
        if emit:
            self.onStimNoChange.emit(new_stim)

    def setStimNo(self, stimno: int, user_set: bool = True):
        # Manual navigation pauses following for this container only.
        if user_set and self._follow_live:
            self._follow_live = False
            self.onFollowLiveChanged.emit(False)
            self.onFollowLiveEventsChanged.emit(False)
        self._set_stimno(stimno)

    def set_follow_live(self, follow: bool):
        follow = bool(follow)
        changed = follow != self._follow_live
        self._follow_live = follow
        if follow:
            latest = self.get_latest_event_idx()
            if latest is not None:
                self._set_stimno(latest)
        if changed:
            self.onFollowLiveChanged.emit(follow)
            self.onFollowLiveEventsChanged.emit(follow)

    def set_follow_live_events(self, enable: bool):
        self.set_follow_live(enable)

    def is_following_live_events(self) -> bool:
        if not self._follow_live:
            return False
        latest = self.get_latest_event_idx()
        if latest is None:
            return bool(self._follow_live)
        return self._stimno == latest

    def get_latest_event_idx(self) -> Optional[int]:
        return self._base.get_latest_event_idx()

    # ------------------------------------------------------------------
    # Stimulus-index dependent operations
    # ------------------------------------------------------------------
    def setUnit(self, latency, stimno=None, unit_idx=None, unitid=None, save_undo=True):
        if stimno is None:
            stimno = self._stimno
        return self._base.setUnit(
            latency,
            stimno=stimno,
            unit_idx=unit_idx,
            unitid=unitid,
            save_undo=save_undo,
        )
