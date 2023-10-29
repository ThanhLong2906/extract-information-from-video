import argparse 

parser = argparse.ArgumentParser()

parser.add_argument("--video", type=str)

args = parser.parse_args()

if __name__=='__main__':
    print(args.video)

