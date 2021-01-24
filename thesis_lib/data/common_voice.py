import csv


def read_tsv(dataset_path, tsv_name):
	data = []
	with open(dataset_path + tsv_name, encoding="utf-8") as train_tsc:
		reader = csv.reader(train_tsc, delimiter="\t", strict=True)
		next(reader)  # skip header
		for row in reader:
			file_name = row[1].replace(".mp3", ".wav").strip()
			sentence = row[2].strip()
			data.append((file_name, sentence))
	return data


def read_filtered_tsv(dataset_path, tsv_name):
	data = []
	with open(dataset_path + tsv_name, encoding="utf-8") as train_tsc:
		reader = csv.reader(train_tsc, delimiter="\t", strict=True)
		for row in reader:
			file_name = row[0].strip()
			sentence = row[1].strip()
			data.append((file_name, sentence))
	return data
