v53 changes
===========
1. Prevents Streamlit Cloud crashes from accidentally inflated Excel used ranges
   such as A1:FU1048518 or A1:XFC56.
2. Adds lightweight XLSX preflight and a bounded XML reader that reads only real,
   non-empty TMU cells instead of allocating the whole Excel used range.
3. Keeps hidden/far-right Pumping Pressure extraction through the XML rescue.
4. Merges repeated uploads by normalized Well + Date/Time (minute):
   - keeps the most complete row;
   - fills missing fields from the incomplete copy;
   - ties prefer the later uploaded file;
   - avoids merging Unknown wells or missing timestamps.
5. Adds per-file progress/error isolation and explicit garbage collection in app.py.
