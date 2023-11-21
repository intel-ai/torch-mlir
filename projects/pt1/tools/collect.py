import os
import argparse

def process(filename):
    result = {}
    with open(filename, 'r') as f:
        data = f.readlines()
        for line in data:
            if 'elapsed' in line:
                tmp = line.strip().split()
                print(tmp[0].strip(':'), tmp[-2])
                result[tmp[0].strip(':')] = tmp[-2]
        result['size'] = filename.split('_')[0].strip('./')
    return result

def collect(path, result):
    collection = []
    for file in os.listdir(path):
        if file.endswith('.log'):
            print(os.path.join(path, file))
            res = process(os.path.join(path, file))
            collection.append(res)

    with open(result, 'w') as f:
        header = sorted(collection[0].keys())
        f.write(",".join(header))
        for item in collection:
            values = []
            for h in header:
                values.append(item[h])
            f.write("\n")
            f.write(",".join(values))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for parsing debug timer dumps to convert it to a csv format.")
    parser.add_argument('path', type=str, help="Provide a path to log files location.")
    parser.add_argument('-o', '--output', default="result.csv", help="Resulting output filename.")
    args = parser.parse_args()
    collect(args.path, args.output)
