TMU Dashboard v60 - Cleaner Sidebar and Choke UI

Changes
1. Removed technical choke conflict/ambiguity messages from the normal user interface.
2. Removed the displayed choke-conversion formula caption.
3. Removed the "Also show original choke columns" checkbox.
   - Unified choke modes show only the unified choke feature.
   - "Keep source units separate" shows the original choke fields automatically.
4. Removed the unit-change explanatory caption below feature selection.
5. Reorganized the sidebar into seven collapsible sections:
   1. Data input
   2. Processing options
   3. Units and choke
   4. Time and timeline
   5. Wells and features
   6. Chart appearance
   7. Events and notes
6. Kept data input and well/feature selection open by default; advanced controls are collapsed.

Deployment
Replace the repository files with this package and reboot the Streamlit app.
No manual cache clearing is needed because this update changes only the interface and display controls.
