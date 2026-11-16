from typing import Callable, List, Optional, Tuple
from PySide6.QtCore import Qt, QObject
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QToolBar
import pyqtgraph as pg


class PGNavigationMixin:
    """
    Mixin for PyQtGraph visualizer views providing Matplotlib-like
    navigation modes (Normal, Pan, Zoom to Rectangle) and coordinate range access.
    """

    nav_mode: str = ""  # "", "pan", "zoom"
    _nav_mode_callbacks: Optional[List[Callable[[str], None]]] = None
    _action_pan: Optional[QAction] = None
    _action_zoom: Optional[QAction] = None
    _action_reset: Optional[QAction] = None

    def init_navigation(self, toolbar: Optional[QToolBar] = None):
        """Initialize navigation mode and optionally attach navigation actions to a toolbar."""
        self.nav_mode = ""
        self._nav_mode_callbacks = []
        if toolbar is not None:
            self.add_navigation_actions(toolbar)
        self.set_nav_mode("")

    def get_nav_mode(self) -> str:
        """Returns the current navigation mode: '', 'pan', or 'zoom'."""
        return getattr(self, "nav_mode", "")

    def is_nav_active(self) -> bool:
        """True if either Pan or Zoom mode is currently active."""
        return getattr(self, "nav_mode", "") != ""

    def register_nav_mode_callback(self, cb: Callable[[str], None]):
        if self._nav_mode_callbacks is None:
            self._nav_mode_callbacks = []
        if cb not in self._nav_mode_callbacks:
            self._nav_mode_callbacks.append(cb)

    def unregister_nav_mode_callback(self, cb: Callable[[str], None]):
        if self._nav_mode_callbacks and cb in self._nav_mode_callbacks:
            self._nav_mode_callbacks.remove(cb)

    def get_nav_plots(self) -> List[pg.PlotItem]:
        """Returns all PlotItem instances associated with this view."""
        plots = []
        if hasattr(self, "main_plot") and self.main_plot is not None:
            plots.append(self.main_plot)
        if hasattr(self, "_aux_plot_by_channel") and isinstance(
            self._aux_plot_by_channel, dict
        ):
            for p in self._aux_plot_by_channel.values():
                if p not in plots:
                    plots.append(p)
        if hasattr(self, "heat_plot") and self.heat_plot is not None:
            if self.heat_plot not in plots:
                plots.append(self.heat_plot)
        if hasattr(self, "bar_plot") and self.bar_plot is not None:
            if self.bar_plot not in plots:
                plots.append(self.bar_plot)
        if hasattr(self, "_grid_items") and isinstance(self._grid_items, list):
            for item in self._grid_items:
                if isinstance(item, tuple) and len(item) > 0:
                    p = item[0]
                    if p not in plots:
                        plots.append(p)
        return plots

    def get_primary_nav_plot(self) -> Optional[pg.PlotItem]:
        """Returns the primary PlotItem for range queries and coordinate manipulation."""
        if hasattr(self, "main_plot") and self.main_plot is not None:
            return self.main_plot
        if hasattr(self, "heat_plot") and self.heat_plot is not None:
            return self.heat_plot
        plots = self.get_nav_plots()
        return plots[0] if plots else None

    def set_nav_mode(self, mode: str):
        """Sets navigation mode: '' / 'normal', 'pan', or 'zoom'."""
        normalized = (mode or "").lower().strip()
        if normalized in ("pan", "panmode"):
            self.nav_mode = "pan"
        elif normalized in ("zoom", "zoommode", "rect", "rectmode"):
            self.nav_mode = "zoom"
        else:
            self.nav_mode = ""

        # Update all plots managed by this view
        for p in self.get_nav_plots():
            if hasattr(p, "vb") and p.vb is not None:
                vb = p.vb
                if self.nav_mode == "pan":
                    vb.setMouseEnabled(x=True, y=True)
                    vb.setMouseMode(pg.ViewBox.PanMode)
                    vb.setCursor(Qt.OpenHandCursor)
                elif self.nav_mode == "zoom":
                    vb.setMouseEnabled(x=True, y=True)
                    vb.setMouseMode(pg.ViewBox.RectMode)
                    vb.setCursor(Qt.CrossCursor)
                else:
                    # Normal mode: disable left-drag pan so clicking does not accidentally move axes
                    vb.setMouseEnabled(x=False, y=False)
                    vb.setCursor(Qt.ArrowCursor)

        # Sync toolbar actions if present
        if getattr(self, "_action_pan", None) is not None:
            self._action_pan.blockSignals(True)
            self._action_pan.setChecked(self.nav_mode == "pan")
            self._action_pan.blockSignals(False)

        if getattr(self, "_action_zoom", None) is not None:
            self._action_zoom.blockSignals(True)
            self._action_zoom.setChecked(self.nav_mode == "zoom")
            self._action_zoom.blockSignals(False)

        # Notify callbacks (e.g. ViewSettingsPanel)
        if getattr(self, "_nav_mode_callbacks", None):
            for cb in list(self._nav_mode_callbacks):
                try:
                    cb(self.nav_mode)
                except Exception:
                    pass

    def get_view_range(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Returns ((xmin, xmax), (ymin, ymax)) of the primary plot."""
        plot = self.get_primary_nav_plot()
        if plot is not None and hasattr(plot, "viewRange"):
            try:
                (x0, x1), (y0, y1) = plot.viewRange()
                xmin, xmax = float(min(x0, x1)), float(max(x0, x1))
                ymin, ymax = float(min(y0, y1)), float(max(y0, y1))
                return ((xmin, xmax), (ymin, ymax))
            except Exception:
                pass
        return ((0.0, 1.0), (0.0, 1.0))

    def set_view_range(
        self,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
    ):
        """Sets x_range and/or y_range on the primary plot."""
        plot = self.get_primary_nav_plot()
        if plot is not None:
            if x_range is not None:
                x_min, x_max = float(min(x_range)), float(max(x_range))
                if x_min < x_max:
                    plot.setXRange(x_min, x_max, padding=0)
            if y_range is not None:
                y_min, y_max = float(min(y_range)), float(max(y_range))
                if y_min < y_max:
                    plot.setYRange(y_min, y_max, padding=0)

    def reset_view(self):
        """Reset plot view range to default / auto range."""
        plot = self.get_primary_nav_plot()
        if plot is not None and hasattr(plot, "autoRange"):
            plot.enableAutoRange()

    def add_navigation_actions(self, toolbar: QToolBar):
        """Adds checkable Pan, Zoom, and Reset actions to the given toolbar."""
        toolbar.addSeparator()

        self._action_pan = toolbar.addAction("Pan")
        self._action_pan.setCheckable(True)
        self._action_pan.setToolTip(
            "Pan (drag with left mouse, zoom with right mouse / wheel)"
        )
        self._action_pan.toggled.connect(self._on_action_pan_toggled)

        self._action_zoom = toolbar.addAction("Zoom")
        self._action_zoom.setCheckable(True)
        self._action_zoom.setToolTip("Zoom to box (drag rectangle with left mouse)")
        self._action_zoom.toggled.connect(self._on_action_zoom_toggled)

        self._action_reset = toolbar.addAction("Reset")
        self._action_reset.setToolTip("Reset view range to default")
        self._action_reset.triggered.connect(self.reset_view)

    def _on_action_pan_toggled(self, checked: bool):
        if checked:
            self.set_nav_mode("pan")
        else:
            if self.nav_mode == "pan":
                self.set_nav_mode("")

    def _on_action_zoom_toggled(self, checked: bool):
        if checked:
            self.set_nav_mode("zoom")
        else:
            if self.nav_mode == "zoom":
                self.set_nav_mode("")
