from fine_tune import SupervisedDataset
from Baichuan2_7B_Base.tokenization_baichuan import BaichuanTokenizer

if __name__ == '__main__':
    tokenizer = BaichuanTokenizer.from_pretrained("./Baichuan2_7B_Base")
    data_path = 'data/test_data.json'
    model_max_length = 128
    # dataset = SupervisedDataset(data_path, tokenizer, model_max_length)
    a =[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2015, 65, 2835, 11024, 1853, 8736, 70, 23387,
     92855, 23656, 68, 89446, 92614, 79107, 66, 5, 5380, 24616, 93660, 96261, 65, 92731, 94404, 84465, 92381, 66, 2,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 17759, 93319, 92400, 92441, 93849, 92786, 93676, 94859,
     93151, 31506, 97923, 92336, 92335, 92383, 92373, 97905, 92381, 65, 92813, 94893, 94459, 2537, 65, 10430, 26231, 66]
    print(tokenizer.tokenize(tokenizer.decode(a)))
    print(tokenizer.decode(tokenizer.eos_token_id))