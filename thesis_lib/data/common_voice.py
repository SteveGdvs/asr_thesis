import csv


def read_tsv(dataset_path, tsv_name):
	data = []
	with open(dataset_path + tsv_name, encoding="utf-8") as train_tsc:
		reader = csv.reader(train_tsc, delimiter="\t", strict=True)
		next(reader)  # skip header
		for row in reader:
			file_name = row[1].replace(".mp3", ".wav")
			sentence = row[2]
			data.append((file_name, sentence))
	return data
