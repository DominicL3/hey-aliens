import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse

"""Compiles a PDF of the RFI files that have been predicted to be FRBs.
Also compiles a separate PDF of a chosen number of RFI files that have 
been predicted to be false."""

if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('rfi_array', type=str)
    parser.add_argument('predictions_bool_file', type=str)
    parser.add_argument('pdf_frb', type=str)
    parser.add_argument('pdf_rfi', type=str)
    parser.add_argument('--num_false', default=8, type=int)
    parser.add_argument('--seed', default=2400, type=int)

    args = parser.parse_args()

    # input path to RFI array and array of predictions [True, False...False]
    rfi_array_name = args.rfi_array
    predictions_bool_file = args.predictions_bool_file

    rfi_array = np.load(rfi_array_name)
    predictions = np.load(predictions_bool_file)

    # separate predicted FRBs from predicted RFI
    frb_predicted = rfi_array[predictions]
    rfi_predicted = rfi_array[~np.array(predictions)]

    # save the predicted FRBs to a PDF
    with PdfPages(args.pdf_frb) as pdf:
        for FRB in frb_predicted:
            fig, ax = plt.subplots()
            ax.imshow(FRB)
            pdf.savefig(fig, dpi=300)
            plt.close(fig)

    # save a certain number of purportedly false values
    np.random.seed(args.seed)
    with PdfPages(args.pdf_rfi) as pdf:
        for random_RFI_idx in np.random.randint(0, high=len(rfi_predicted), size=args.num_false):
            fig, ax = plt.subplots()
            ax.imshow(rfi_predicted[random_RFI_idx])
            pdf.savefig(fig, dpi=300)
            plt.close(fig)
