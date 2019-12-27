from xeger import Xeger
import re

def gen_data(true_regex, false_regex, train_file, train_size, test_file, test_size):
    gen = Xeger(limit=30)
    t = lambda: ' '.join((gen.xeger(true_regex), '1\n'))
    f = lambda: ' '.join((gen.xeger(false_regex), '0\n'))
    with open(train_file, 'w+') as train:
        train.writelines([t() for _ in range(int(train_size/2))] + [f() for _ in range(int(train_size/2))])
    with open(test_file, 'w+') as test:
        test.writelines([t() for _ in range(int(test_size/2))] + [f() for _ in range(int(test_size/2))])


def main():
    true = re.compile('[1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+')
    false = re.compile('[1-9]+a+[1-9]+c+[1-9]+b+[1-9]+d+[1-9]+')
    gen_data(true, false, 'data/train', 40000, 'data/test', 10000)


if __name__ == "__main__":
    main()