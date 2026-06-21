v54 changes
- Fixed Pandas 3 / Python 3.14 Arrow string dtype crash in the safe XLSX XML parser.
- Raw Excel XML cells are now always stored in object-dtype DataFrames before Excel date/time conversion.
- Date/time conversion assignments use object arrays, preventing Invalid value for dtype str errors.
- Keeps v53 safe bounded XML parsing and duplicate well+minute merge logic.
- Parser cache build ID bumped to v54.
