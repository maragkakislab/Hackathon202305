import sys
import argparse
import csv

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-i", "--input")
parser.add_argument("-b", "--bins", default = 20)
args = parser.parse_args()

def normalize(x):
    s = sum(x)
    return [i/s for i in x]

aggregated_data = {}
with open(args.input, 'r') as file:
    reader = csv.reader(file, delimiter="\t")
    next(reader)  # Skip the header row if present
    for row in reader:
        transcript = row[2]
        if transcript not in aggregated_data:
            aggregated_data[transcript] = {
                '5p': [0 for i in range(0,int(args.bins))],
                '3p': [0 for i in range(0,int(args.bins))]}
        aggregated_data[transcript]['5p'][int(row[3])] += 1
        aggregated_data[transcript]['3p'][int(row[4])] += 1


# Print the aggregated data
print("transcript_id\thist5p\thist3p")
for transcript, v in aggregated_data.items():
    hist5p = ",".join(str(i) for i in normalize(v['5p']))
    hist3p = ",".join(str(i) for i in normalize(v['3p']))

    print(f"{transcript}\t{hist5p}\t{hist3p}")
