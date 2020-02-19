# parse arguments
import argparse

# use PlotCand to plot candidates and save to disk
from predict import extract_candidates

if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('filterbank_name', type=str, help='Path to filterbank file with candidates to be predicted.')
    parser.add_argument('frb_cand_file', type=str, help='Path to .txt file containing data about pulses.')
    parser.add_argument('--NCHAN', type=int, default=64, help='Number of frequency channels to resize psrchive files to.')
    parser.add_argument('--save_pdf', type=str, default="raw_candidates", help='Filename to save plot of top 5 candidates.')

    args = parser.parse_args()

    frb_info = extract_candidates(args.filterbank_name, args.frb_cand_file, save_png=True)