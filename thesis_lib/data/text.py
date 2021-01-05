def load_parallel_texts(path, lines_limit=None, max_length=None, remove_duplicates=True, remove_in_list=None):
	with open(path, encoding="utf_8") as f:
		lines = f.readlines()

	if lines_limit is not None:
		lines = lines[: min(lines_limit, len(lines) - 1)]

	raw_texts = []
	for line in lines:
		parts = line.split("\t")
		input_text = parts[0].strip()
		target_text = parts[1].strip()
		raw_texts.append((input_text, target_text))

	print("Read {0} lines from \"{1}\"".format(len(lines), path))

	if max_length is not None:
		new_raw_texts = []

		for in_txt, tar_txt in raw_texts:
			if len(in_txt) < max_length and len(tar_txt) < max_length:
				new_raw_texts.append((in_txt, tar_txt))

		print("- Removed {0} sentences exceeding {1} characters".format(len(raw_texts) - len(new_raw_texts), max_length))
		raw_texts = new_raw_texts

	if remove_duplicates:
		texts_dup_set = set()
		new_raw_texts = []

		for in_txt, tar_txt in raw_texts:
			if in_txt not in texts_dup_set:
				texts_dup_set.add(in_txt)
				new_raw_texts.append((in_txt, tar_txt))

		print("- Removed {0} duplicates".format(len(raw_texts) - len(new_raw_texts)))
		raw_texts = new_raw_texts

	if remove_in_list:
		for word in remove_in_list:
			new_raw_texts = []
			for in_txt, tar_txt in raw_texts:
				if word.lower() not in in_txt.lower():
					new_raw_texts.append((in_txt, tar_txt))

			print("- Removed {0} lines containing the word {1}".format((len(raw_texts) - len(new_raw_texts)), word))
			raw_texts = new_raw_texts

	print("Loaded {0} sentences".format(len(raw_texts)))
	return raw_texts
