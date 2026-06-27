"""Runtime validation for v80 using Streamlit AppTest."""
from pathlib import Path
from streamlit.testing.v1 import AppTest

root = Path(__file__).resolve().parent
zip_path = Path("/mnt/data/WhatsApp Chat - Sitra8-58 clean & test(2).zip")

at = AppTest.from_file(str(root / "app.py"), default_timeout=180).run(timeout=180)
assert not at.exception, list(at.exception)
assert len(at.radio) >= 1 and at.radio[0].label == "Theme"

at.radio[0].set_value("Dark").run(timeout=180)
assert not at.exception, list(at.exception)

if zip_path.exists():
    at.file_uploader[0].upload(zip_path.name, zip_path.read_bytes()).run(timeout=180)
    assert not at.exception, list(at.exception)
    choose_wells = [w for w in at.multiselect if w.label == "Choose wells"]
    assert choose_wells, "Choose wells multiselect was not created"
    assert "S8-58" in choose_wells[0].options

print("Production Test Dashboard v80 runtime/AppTest: PASS")
