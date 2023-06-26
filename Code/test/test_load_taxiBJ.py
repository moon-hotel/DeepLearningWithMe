import sys

sys.path.append('../../')

from utils import TaxiBJ

if __name__ == '__main__':
    days_test = 7 * 4
    T = 48
    len_closeness = 3  # length of closeness dependent sequence
    len_period = 1  # length of  peroid dependent sequence
    len_trend = 1  # length of trend dependent sequence
    len_test = T * days_test
    taxibj = TaxiBJ(len_test=len_test, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend)
    taxibj.show_example()
    test_iter, mmn = taxibj.load_train_test_data(is_train=False)
    for XC_test, XP_test, XT_test, Y_test, meta_feature_test, timestamp_test in test_iter:
        print(XC_test.shape)
        print(XP_test.shape)
        print(XT_test.shape)
        print(Y_test.shape)
        print(meta_feature_test.shape)
        print(len(timestamp_test))
        break
