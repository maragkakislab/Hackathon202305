"""
Converts the 5' and 3' coordinates of reads into coordinates on meta-features.
In essence, each feature is divided into bins; the 5' and 3' coordinates of
each read contained in each feature are replaced by the bin index in which
they are included. For example for a feature of length 100 containing a read
with coordinates [5', 3']: [15, 95] the corresponding meta-coordinates for
10 bins are [1, 9]. THe script works only for +ve strands and BED coordinates
grater than certain length of nucleotide set in 'min-len' parameter.
"""


import sys
import pysam
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def parse_args(args_list):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--ifile", required=True, nargs='+',
                        help="Input SAM/BAM file")
    parser.add_argument("-b", "--bed", required=True,
                        help="BED file with features")
    parser.add_argument("-c", "--bins", default=20, type=int,
                        help="Number of bins per feature. default: %(default)s")
    parser.add_argument("-l", "--min-len", default=200, type=int,
                        help="Skip features shorter than this. default:%(default)s")
    parser.add_argument("-s", "--col_delimiter", default="\t",
                        help="Delimiter for output file, default:<TAB>")
    parser.add_argument("-n", "--iname", nargs='+',
                        help="Name to be used for SAM/BAM file in output table columns")
    parser.add_argument("-o", "--pdf", required=True,
                        help="Output pdf file with plots")
    parser.add_argument("--ohist_5p", required=True,
                        help="Output tab-separated file with histogram for 5'")
    parser.add_argument("--ohist_3p", required=True,
                        help="Output tab-separated file with histogram for 3'")
    parser.add_argument("--skip-stdout", action='store_true',
                        help="Do not print new coordinates to STDOUT")

    # DEPRECATED OPTIONS. These are silently ignored for now.
    parser.add_argument("-f", "--sam", action='store_true',
                        help="[Deprecated] Input is in the SAM format")
    parser.add_argument("-d", "--col-qname-name", default="qname",
                        help="[Deprecated] Header for column query name, default: qname")
    parser.add_argument("-e", "--col-feat-name", default="feat",
                        help="[Deprecated] Header of column wih features, default: feat")
    parser.add_argument("-g", "--col-bin5p-name", default="bin5p",
                        help="[Deprecated] Header for 5 prime bins of read, default: bin5p")
    parser.add_argument("-j", "--col-bin3p-name", default="bin3p",
                        help="[Deprecated] Header for 3 prime bins of read, default: bin3p")

    # Parse the argument list.
    args = parser.parse_args(args_list)

    # If provided, ensure the number of names is equal to the number of files.
    if args.iname is not None and len(args.iname) != len(args.ifile):
        raise ValueError(
            'Error: provided --iname values should be same number as files')

    return args


def read_bed_to_dict(bedfile, min_len=None):
    """
    Parse BED file and create a dictionary associating the reference name to
    the corresponding start and end positions.
    """
    region_coords = {}
    with open(bedfile) as bed:
        for line in bed:
            line = line.strip().split("\t")
            if line[5] == "-":
                raise ValueError(
                    'Error: BED entry on negative strand found.\n'+line)
            refid = line[0]
            ref_start = int(line[1])
            ref_end = int(line[2])
            if min_len is not None and (ref_end - ref_start) < min_len:
                continue
            if refid in region_coords:
                raise ValueError(
                    'Error: multiple BED entries on same reference found.\n'+line)
            region_coords[refid] = [ref_start, ref_end]
    return region_coords


def main():
    args = parse_args(sys.argv[1:])
    delim = args.col_delimiter

    # Parse BED file and create a hash associating the reference name to the
    # corresponding start and end positions.
    bed_coords = read_bed_to_dict(args.bed, args.min_len)

    # Print header for stdout.
    writer = None
    if not args.skip_stdout:
        writer = sys.stdout
        print(delim.join(['file', 'qname', 'feat', 'bin5p', 'bin3p']),
              file=writer)

    # Loop on all input BAM/SAM files and print the meta-coords to stdout.
    # Also return data frames with the corresponding histograms per file.
    # Printing and histogram creation is done simultaneously for
    # optimization.
    dfs_5p, dfs_3p = [], []
    for i, ifile in enumerate(args.ifile):
        name = str(i) if args.iname is None else args.iname[i]
        df5p, df3p = calc_meta_coords_for_file(ifile, name, bed_coords,
                                               bins=args.bins, writer=writer,
                                               delim=delim)
        df5p['density'] = df5p['count'] / \
            df5p.groupby('file')['count'].transform('sum')
        df3p['density'] = df3p['count'] / \
            df3p.groupby('file')['count'].transform('sum')
        dfs_5p.append(df5p)
        dfs_3p.append(df3p)

    # Concatenate all data frames. The data frames contain column 'file' that
    # can help distinguish between histograms of different files.
    df5p_all = pd.concat(dfs_5p)
    df3p_all = pd.concat(dfs_3p)

    # Create figures.
    figs = []
    figs.append(create_fig_for_df(df5p_all, '5p'))
    figs.append(create_fig_for_df(df3p_all, '3p'))
    save_figs_in_pdf(figs, args.pdf)

    # Print the data frame with histograms to a file.
    df5p_all.to_csv(args.ohist_5p, sep="\t", index=False)
    df3p_all.to_csv(args.ohist_3p, sep="\t", index=False)


def save_figs_in_pdf(figs, pdf):
    """
    Plot the figures in a pdf.
    """
    with PdfPages(pdf) as pages:
        for f in figs:
            pages.savefig(f)
            plt.close()


def create_fig_for_df(df, name):
    """
    Create the figure.
    """
    fig, ax = plt.subplots(layout='constrained')
    fig.suptitle(name)
    sns.lineplot(ax=ax, data=df, x='bin', y='density', hue='file')
    plt.xlabel('Binned position')
    plt.ylabel('Read density')
    return fig


def get_binned_coords(bed_start, bed_end, read_pos, bins):
    """
    Calculate the new binned coords for the read.
    """
    bin_len = (bed_end - bed_start)/bins
    binned_pos = int((read_pos - bed_start)/bin_len)

    if binned_pos < 0 or binned_pos >= bins:
        return None

    return binned_pos


def calc_meta_coords_for_file(ifile, name, bed_coords, bins=20, writer=None,
                              delim='\t'):

    # Create dataframes that will hold the read counts per bin, separately for
    # each end.
    def init_data_frame(name, bins):
        return pd.DataFrame({'file': name, 'bin': range(bins), 'count': 0})

    df5p = init_data_frame(name, bins)
    df3p = init_data_frame(name, bins)

    # Identify whether we have a BAM or SAM file.
    filemode = 'rb' if ifile[-3:] == 'bam' else 'r'

    # Loop on the BAM/SAM file and calculate the bins
    infile = pysam.AlignmentFile(ifile, filemode)
    for read in infile:
        if (read.is_unmapped or read.is_reverse or
                read.reference_name not in bed_coords):
            continue

        bstart, bend = bed_coords[read.reference_name]
        bin5p = get_binned_coords(bstart, bend, read.reference_start, bins)
        bin3p = get_binned_coords(bstart, bend, read.reference_end-1, bins)

        if bin5p is not None:
            df5p.loc[bin5p, 'count'] += 1
        if bin3p is not None:
            df3p.loc[bin3p, 'count'] += 1

        if writer is not None:  # Print the binned coordinates
            print(delim.join([name, read.query_name, read.reference_name,
                              str(bin5p), str(bin3p)]), file=writer)

    return df5p, df3p


if __name__ == "__main__":
    main()
