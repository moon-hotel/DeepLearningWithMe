from fine_tune import SupervisedDataset
from Baichuan2_7B_Base.tokenization_baichuan import BaichuanTokenizer

if __name__ == '__main__':
    tokenizer = BaichuanTokenizer.from_pretrained("./Baichuan2_7B_Base")
    data_path = 'data/test_data.json'
    model_max_length = 96
    dataset = SupervisedDataset(data_path, tokenizer, model_max_length)
    for x in dataset:
        print(x)  # torch.Size([96])
        # {'input_ids': tensor([  195, 92676, 19278, 48278, 26702, 93319, 92364, 73791, 10430, 82831,
        #             5,   196,  2015,    65,  2835, 11024,  1853,  8736,    70, 23387,
        #         92855, 23656,    68, 89446, 92614, 79107,    66,     5,  5380, 24616,
        #         93660, 96261,    65, 92731, 94404, 84465, 92381,    66,   195, 17759,
        #         93319, 92347, 26702, 11090, 15473,    68,     5,   196, 17759, 93319,
        #         92400, 92441, 93849, 92786, 93676, 94859, 93151, 31506, 97923, 92336,
        #         92335, 92383, 92373, 97905, 92381,    65, 92813, 94893, 94459,  2537,
        #            65, 10430, 26231,    66,     2,     0,     0,     0,     0,     0,
        #             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #             0,     0,     0,     0,     0,     0]),
        # 'labels': tensor([    2,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
        #          -100,  -100,  2015,    65,  2835, 11024,  1853,  8736,    70, 23387,
        #         92855, 23656,    68, 89446, 92614, 79107,    66,     5,  5380, 24616,
        #         93660, 96261,    65, 92731, 94404, 84465, 92381,    66,     2,  -100,
        #          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100, 17759, 93319,
        #         92400, 92441, 93849, 92786, 93676, 94859, 93151, 31506, 97923, 92336,
        #         92335, 92383, 92373, 97905, 92381,    65, 92813, 94893, 94459,  2537,
        #            65, 10430, 26231,    66,     2,  -100,  -100,  -100,  -100,  -100,
        #          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
        #          -100,  -100,  -100,  -100,  -100,  -100]),
        #  'attention_mask': tensor([ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #          True,  True,  True,  True,  True, False, False, False, False, False,
        #         False, False, False, False, False, False, False, False, False, False,
        #         False, False, False, False, False, False])}

        break

