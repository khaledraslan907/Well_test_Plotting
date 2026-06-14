TMU Dashboard v43 - WhatsApp ZIP + CTU OCR + Safe Test Linking

Main purpose
------------
This version lets the user upload:
- Normal Excel / CSV / Word / PDF test files
- WhatsApp exported ZIP files, including _chat.txt and attachments
- Direct CTU/HMI screen images: JPG / JPEG / PNG
- Pasted WhatsApp TMU messages

What changed
------------
1) WhatsApp exported ZIP support
   - The uploader now accepts .zip.
   - The parser opens WhatsApp export ZIP files.
   - It reads _chat.txt.
   - It parses Excel / CSV / PDF / Word attachments inside the ZIP.
   - It can OCR CTU/HMI images inside the ZIP if enabled.
   - It ignores common WhatsApp system/noise messages.

2) Test timing logic by well name
   - Same well continues as the same test until the selected gap is exceeded.
   - Different well name = different test stream.
   - Default gap is 12 hours, editable in the Streamlit sidebar.
   - A test_id column is created automatically.

3) CTU/HMI image OCR
   - Added OCR for CTU/PICO-style ALL DATA screen photos.
   - Output fields include CTU Weight, Lt Weight, Wellhead Pressure, Circulation Pressure, Reel Depth, Reel Speed, Fluid Rate, N2 Flow, Fluid Total, and N2 Total.
   - OCR is best-effort and can be imperfect when the photo is blurry or has glare.

4) Safe OCR linking
   - CTU image rows are NOT automatically assigned to a well/test.
   - The app only suggests a nearest test when a confirmed text/Excel/PDF row is close in time.
   - Suggestions are not applied unless the user approves them in the sidebar.
   - The OCR review table allows manual Well/Test ID assignment.
   - This prevents misleading data from being silently plotted under the wrong well.

5) Streamlit UI additions
   - Upload help for WhatsApp ZIPs.
   - OCR enable/disable toggle.
   - Maximum OCR images per ZIP setting to keep free Streamlit runs fast.
   - CTU/OCR review table.
   - Test/period selection in addition to well selection.
   - Metrics now include detected Tests.

Important note about OCR accuracy
---------------------------------
The OCR module is intentionally conservative. For low-quality CTU photos, some numbers may be partially or incorrectly recognized. The app marks OCR rows as review_required and requires user review before linking/plotting. Excel, PDF, and typed WhatsApp test reports remain the preferred high-confidence data sources.

Deployment on Streamlit Cloud
-----------------------------
Upload these files:
- app.py
- tmu_parser.py
- requirements.txt
- runtime.txt
- packages.txt

packages.txt contains:
- tesseract-ocr

This is required for OCR on Streamlit Cloud.

Run locally
-----------
pip install -r requirements.txt
streamlit run app.py
