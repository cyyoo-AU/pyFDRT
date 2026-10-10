import os
import glob
import argparse

from drt_pipeline import DRTAnalyzer
from pyfdrt.io.drt_exporter import export_results_to_excel
from pyfdrt.io.drt_plotter import save_circuit_diagram, save_drt_plots


def main():
    parser = argparse.ArgumentParser(description="Batch analysis of EIS data files using pyFDRT pipeline.")
    parser.add_argument("--input_dir", type=str, default="Data", help="Directory containing EIS data files.")
    parser.add_argument("--output_dir", type=str, default="Results", help="Directory to save output files.")
    parser.add_argument("--file_pattern", type=str, default="*", help="File name pattern (e.g. *.txt, *.z, *.csv)")
    # Deconvolution method (task #76). zarc is the validated default
    # (drt-deconvolution.md 4.0: the only method recovering all known-truth
    # spectra); lasso/cwt are the legacy cross-check bases.
    parser.add_argument("--deconv-method", dest="deconv_method",
                        choices=["zarc", "lasso", "cwt"], default="zarc",
                        help="Peak deconvolution method (default: zarc). "
                             "lasso/cwt are cross-check bases - see "
                             "docs/specs/drt-deconvolution.md section 4.0.")

    args = parser.parse_args()

    # Check that the input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect data files
    patterns = [args.file_pattern] if args.file_pattern != "*" else ["*.txt", "*.z", "*.csv", "*.dat"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(args.input_dir, pat)))
    
    # De-duplicate and sort
    files = sorted(list(set(files)))

    if not files:
        print(f"No files found in '{args.input_dir}' matching pattern '{args.file_pattern}'.")
        return

    print(f"Found {len(files)} files to analyze in '{args.input_dir}'.")

    # Create the pipeline instance
    analyzer = DRTAnalyzer(logger=print)

    # Use the best-performing GUI defaults
    deconv_label = {"zarc": "ZARC + BIC",
                    "lasso": "LASSO + Curve-Fit",
                    "cwt": "CWT + BIC"}[args.deconv_method]
    config = {
        'l_corr_mode': 'Auto (Smart Detection)',
        'ext_decades': 2,
        'ext_method': 'Auto (Linear → RC fallback)',
        'lambda_mode': 'Auto (L-Curve)',
        'max_peaks': 10,
        'criterion': 'Conservative (BIC)',
        'deconvolution_method': deconv_label,
    }

    # A per-file failure must not abort the whole batch, but it must surface in
    # the final summary rather than being lost in the scroll-back. The Excel
    # exporter now raises ExcelExportError on a write failure instead of
    # returning False, so a failed export lands here like any other failure.
    succeeded = []
    failed = []
    for fpath in files:
        fname = os.path.basename(fpath)
        base_name, _ = os.path.splitext(fname)
        print("\n" + "="*60)
        print(f"Analyzing: {fname}")
        print("="*60)

        try:
            # 1. Load data and run the analysis
            raw_data = analyzer.load_data(fpath)
            res = analyzer.run(config)

            optimizer = res['optimizer']
            results = res['results']

            # 2. Export to Excel
            excel_path = os.path.join(args.output_dir, f"{base_name}_results.xlsx")
            export_results_to_excel(excel_path, optimizer, results, logger=print,
                                    inductance=res.get('L_val', 0.0))

            # 3. Save the plot image
            image_path = os.path.join(args.output_dir, f"{base_name}_plot.png")
            save_drt_plots(image_path, optimizer, results, original_raw_data=raw_data)
            print(f"-> [OK] Plot saved: {image_path}")

            # 4. Save the circuit schematic. Its own file, not a fourth panel:
            # the schematic is one wide, short row and a quarter-width column
            # would compress it to illegibility (drt-io.md 6.1).
            circuit_path = os.path.join(args.output_dir, f"{base_name}_circuit.png")
            save_circuit_diagram(circuit_path, results,
                                 inductance=res.get('L_val', 0.0))
            print(f"-> [OK] Circuit saved: {circuit_path}")

            succeeded.append(fname)
        except Exception as e:
            print(f"-> [Failed] Error analyzing {fname}: {e}")
            import traceback
            traceback.print_exc()
            failed.append((fname, str(e)))

    print("\n" + "="*60)
    print(f"Batch analysis complete: {len(succeeded)} succeeded, "
          f"{len(failed)} failed (of {len(files)}).")
    if failed:
        print("Failed files:")
        for fname, err in failed:
            print(f"  - {fname}: {err}")

if __name__ == "__main__":
    main()
