import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QToolBar
from PySide6.QtGui import QAction
from PySide6.QtCore import Signal, Slot
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT

from ..ViewerState import ViewerState
from ..live_orchestrator import LiveFrameConsumerMixin


class QMatplotlib(LiveFrameConsumerMixin, QMainWindow):
    # Define a custom signal for updating the plot
    plot_update_signal = Signal(bool)

    def __init__(self, state=None, parent=None, include_matplotlib_toolbar=True):
        super().__init__(parent)
        self.settings = {}

        # Create a central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create a vertical layout
        layout = QVBoxLayout(central_widget)

        # Create a Matplotlib figure and canvas
        self.figure = Figure(layout="tight")
        self.canvas = FigureCanvas(self.figure)

        # Add the canvas to the layout
        layout.addWidget(self.canvas)

        # Connect the click event handler
        self.canvas.mpl_connect("button_press_event", self.on_click)

        # Connect the plot update signal to the update_plot slot
        self.plot_update_signal.connect(self.update_figure)

        # Plot some example data

        # self.setup_figure()
        if state is not None:
            self.setState(state)

        if include_matplotlib_toolbar:
            self.matplotlib_toolbar = NavigationToolbar2QT(self.canvas, self)
            self.addToolBar(self.matplotlib_toolbar)

        # Create a toolbar
        toolbar = self.create_toolbar()
        if toolbar is not None:
            if hasattr(self, "matplotlib_toolbar"):
                self.addToolBarBreak()
            self.addToolBar(toolbar)

    def setState(self, state: ViewerState):
        self.state: ViewerState = state
        self.state.onUnitChange.connect(self.onUnitChange)
        self.state.onUnitGroupChange.connect(self.onUnitGroupChange)
        # Static file loads are processed through the same orchestrator
        # pipeline as live sweeps (single point of view processing).
        self._register_data_reload(state, self.onLoadNewFile, name="reload_redraw")
        self.plot_update_signal.emit(True)
        if self.figure is not None:

            self.figure.clear()
            self.setup_figure()
            self.update_figure()

    def onUnitChange(self):
        # Emit the plot update signal
        self.plot_update_signal.emit(False)

    def onUnitGroupChange(self):
        # Emit the plot update signal
        self.plot_update_signal.emit(False)

    def onLoadNewFile(self):
        # Emit the plot update signal
        self.plot_update_signal.emit(True)

    def create_toolbar(self):
        toolbar = QToolBar("Settings")
        # self.addToolBar(toolbar)

        # plot_action = QAction("Plot Example Data", self)
        # plot_action.triggered.connect(self.update_figure)
        # toolbar.addAction(plot_action)

    def setup_figure(self):
        self.ax = self.figure.add_subplot(111)

    def draw_figure(self):
        """
        this is the main plotting function.
        Override this to create your plot.
        """
        self.ax.plot([1, 2, 3, 4, 5], [1, 4, 9, 16, 25])

    def on_limits_change(self, event_ax):
        print(f"xlim changed to: {event_ax.get_xlim()}")
        print(f"ylim changed to: {event_ax.get_ylim()}")

    def on_click(self, event):
        if event.inaxes:
            print(f"Clicked at x={event.xdata}, y={event.ydata}")

    def update_figure(self, should_clear=False):
        # Implement the logic to update the plot
        if should_clear:
            self.figure.clear()
            self.setup_figure()
        self.draw_figure()
        self.canvas.draw()

    def get_settings(self):
        return self.settings

    def set_settings(self, values, clear=False):
        if clear:
            self.settings = values
        else:
            for k, v in values.items():
                self.settings[k] = v

        # self.setup_figure()
        self.update_figure()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QMatplotlib()
    window.show()
    sys.exit(app.exec())
