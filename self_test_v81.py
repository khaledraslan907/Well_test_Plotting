"""Runtime validation for v81 using Streamlit AppTest.

The field-file tests run when the files are available in /mnt/data. They are
skipped on a clean deployment where private validation files are absent.
"""
from pathlib import Path
from streamlit.testing.v1 import AppTest

root = Path(__file__).resolve().parent


def find_widget(items, label):
    return next(item for item in items if item.label == label)


at = AppTest.from_file(str(root / "app.py"), default_timeout=300).run(timeout=300)
assert not at.exception, list(at.exception)
assert find_widget(at.number_input, "New test after inactive gap (hours)").value == 12.0
assert not any(item.label == "How should tests be separated?" for item in at.selectbox)

cases = [
    Path("/mnt/data/WhatsApp Chat - Sitra8-58 clean & test(2).zip"),
    Path("/mnt/data/12-S8-58 (12-06-2026)(4).xlsx"),
    Path("/mnt/data/SPR(2).xlsx"),
]

for file_path in cases:
    if not file_path.exists():
        continue
    test_app = AppTest.from_file(str(root / "app.py"), default_timeout=300).run(timeout=300)
    test_app.file_uploader[0].upload(file_path.name, file_path.read_bytes()).run(timeout=300)
    assert not test_app.exception, (file_path.name, list(test_app.exception))

    test_app.radio[0].set_value("Dark").run(timeout=300)
    assert not test_app.exception, (file_path.name, "dark", list(test_app.exception))

    gap = find_widget(test_app.number_input, "New test after inactive gap (hours)")
    gap.set_value(8.0).run(timeout=300)
    assert not test_app.exception, (file_path.name, "gap", list(test_app.exception))

    aggregation = find_widget(test_app.selectbox, "Average readings by time interval")
    aggregation.set_value("30 minutes").run(timeout=300)
    assert not test_app.exception, (file_path.name, "aggregation", list(test_app.exception))

    plot_style = find_widget(test_app.selectbox, "Plot style")
    plot_style.set_value("Overlay actual values").run(timeout=300)
    assert not test_app.exception, (file_path.name, "plot style", list(test_app.exception))

print("Production Test Dashboard v81 runtime/AppTest: PASS")
