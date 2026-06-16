v52 changes:
- Added XLSX XML-based Pumping Pressure extraction. This directly scans the workbook XML for headers such as pumping.p / Pump P even when the column is hidden or far to the right (e.g., S8-58 column EG).
- Preserved well names such as B3C18-7 during Pumping Pressure merging so the merge does not fail by shortening the name to C18-7.
- Cache bust build id updated to force Streamlit to reparse uploaded files.
- Added post-mapping safety to keep Pumping Pressure numeric and visible in plotting columns.
- Improved Plotly event/note visibility: more top margin, labels placed above the full chart area, bold labels, stronger white label boxes, and full-height event lines spanning all subplots so dragging the line moves the visual event reference across the whole chart.
- Tested with S8-58 and B3 C18-7 Excel examples.
