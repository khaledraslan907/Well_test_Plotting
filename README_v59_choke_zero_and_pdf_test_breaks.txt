TMU Production Test Dashboard - v59

Main fixes
==========
1. False choke zeros
   - Uses explicit float64 assignment instead of pandas combine_first.
   - Adds "Treat zero choke as blank/template value" (default ON).
   - Zero values are removed before choke-unit conversion/filling.
   - Choke is filled only inside the same well + source file + sheet.
   - The same protection applies to unified and optional raw choke curves.

2. Separate tests in PDF/PNG
   - Multi-chart Matplotlib export now plots every detected segment separately.
   - The last point of one test is never joined to the first point of another.
   - Dark dashed vertical boundaries are drawn in interactive and exported charts.
   - Test 1/Test 2/Data gap text labels were removed as requested.

3. Cache invalidation
   - Parser build ID changed to v59 so old v58 parsed/session bundles are not reused.

Deployment
==========
Replace all repository files with this package, commit, and reboot the Streamlit app.
No manual cache-clear button is required.
