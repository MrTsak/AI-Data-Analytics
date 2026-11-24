"""Headless checks for the desktop app (needs a display, e.g. `xvfb-run -a pytest`)."""
import os

import pytest

if not os.environ.get("DISPLAY"):
    pytest.skip("no display available", allow_module_level=True)

ctk = pytest.importorskip("customtkinter")


@pytest.fixture(scope="module")
def app():
    import main

    root = ctk.CTk()
    application = main.DiabetesPredictorApp(root)
    yield application
    root.destroy()


def _show(app, tab):
    app.tabview.set(tab)
    app._update_tab_content(tab)
    app.root.update()


def test_prediction_uses_selected_gender(app):
    _show(app, "Live Prediction")
    app.gender_var.set("Female")
    female = app.build_patient()
    app.gender_var.set("Male")
    male = app.build_patient()
    assert female["gender"] == "Female" and male["gender"] == "Male"


def test_slider_label_updates_without_drag(app):
    _show(app, "Live Prediction")
    _, slider, label = app.feature_widgets["age"]
    slider.set(70)
    slider._command(70)  # what a click invokes
    assert label.cget("text") == "70"


def test_all_tabs_render(app):
    for tab in app.tabs:
        _show(app, tab)
        assert app.content_frame is not None and app.content_frame.winfo_exists()
