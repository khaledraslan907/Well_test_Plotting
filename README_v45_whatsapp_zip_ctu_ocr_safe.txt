TMU Production Test Dashboard v45

Changes in v45:
- Fully disabled nearest-time CTU/OCR suggestions. CTU rows are never assigned or suggested to a well/test by time.
- Fixed pandas dtype crash when WhatsApp bold names like *S8-58* or * are found.
- CTU/OCR rows keep exact WhatsApp/photo timestamp only; user manually chooses Well/Test in review table.
- OCR limit default increased to 300 images; set 0 for no limit. Use checkbox to skip OCR completely.
- WhatsApp ZIP importer parses _chat.txt plus Excel/CSV/PDF/DOCX/image attachments.
- Test IDs are assigned by well name and same-well time gap only.

Deploy files:
- app.py
- tmu_parser.py
- requirements.txt
- packages.txt
- runtime.txt

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
