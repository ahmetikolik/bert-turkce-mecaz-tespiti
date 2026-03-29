from transformers import AutoTokenizer, BertForMaskedLM
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("ytu-ce-cosmos/turkish-tiny-bert-uncased")

print(tokenizer.tokenize("bugün dünyaları yedim ve canım sıkıldı o yüzden altıma sıçtım"))
print("bugün dünyaları yedim".split())