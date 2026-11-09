"""Run dashboard: one self-contained HTML file built from a run's metrics log.

    from dashboard import render_dashboard, display_dashboard
    path = render_dashboard(system["run_dir"])      # writes <run_dir>/dashboard.html
    display_dashboard(system["run_dir"])            # same, and shows it in Jupyter/Colab
"""

from dashboard.render import display_dashboard, render_dashboard, render_fragment

__all__ = ["render_dashboard", "display_dashboard", "render_fragment"]
