TMU Dashboard v33 cloud export-safe fix

Main fix:
- Prevents Streamlit Cloud "Oh no" crashes caused by generating large PNG/PDF export files on every app rerun.
- Export files are now created only after the user clicks Prepare.
- Export figure sizes and DPI are limited to safe values for Streamlit Cloud memory.

Files to upload together:
- app.py
- tmu_parser.py
- requirements.txt
- runtime.txt

After upload:
- Commit changes.
- On Streamlit Cloud, open Manage app and click Reboot app.

If the app still shows the generic crash page:
- Open Manage app > Logs.
- Copy the first red traceback line and send it.
