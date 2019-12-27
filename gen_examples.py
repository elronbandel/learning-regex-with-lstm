from gen_data import gen_data
import re

def main():
    true = re.compile('[1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+')
    false = re.compile('[1-9]+a+[1-9]+c+[1-9]+b+[1-9]+d+[1-9]+')
    gen_data(true, false, 'data/train', 100000, 'data/test', 25000)

if __name__ == "__main__":
    main()