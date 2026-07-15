import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--log_name", help="log filename to count tokens in")
args=parser.parse_args()

with open(f"logs/{args.log_name}.log") as f:
    lines = f.readlines()
    lines = [l for l in lines if "token count" in l]
    tokens = sum([int(l.split(" ")[-1]) for l in lines])
    print(tokens)