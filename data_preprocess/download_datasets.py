from __future__ import unicode_literals, print_function, division
import os, argparse
from datasets import load_dataset


def build_dataset(droot, dsource, dsplit):
    if dsource == "cnndm":
        dataset = load_dataset('cnn_dailymail', '3.0.0', split=f'{dsplit}')
    elif dsource == "xsum":
        dataset = load_dataset('xsum', split=f'{dsplit}')
    filepath = os.path.join(droot, f"{dsplit}.json")
    dataset.to_json(filepath)

def pre_parse():
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument('--dataset_root', type=str, default="./dataset",
                        help='The root directory in which original datasets are downloaded.')
    parser.add_argument('--datasource', type=str, default="cnndm,xsum",
                        help='Data source.')
    parser.add_argument('--data_splits', type=str, default="train,validation,test",
                        help='Data splits.')
    args = parser.parse_args()
    return args

def main():
    args = pre_parse()
    datasource = args.datasource.split(",")
    data_splits = args.data_splits.split(",")
    for dsource in datasource:
        droot = os.path.join(args.dataset_root, dsource, "downloaded")
        os.makedirs(droot, exist_ok=True)
        for dsplit in data_splits:
            build_dataset(droot, dsource, dsplit)

if __name__ == "__main__":
    main()
    print("Done")
