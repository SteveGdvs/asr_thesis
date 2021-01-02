import multiprocessing
import subprocess
from pathlib import Path

from tqdm import tqdm


class TC:
	"""
	terminal_colors
	escape sequences for coloring terminal text
	"""
	GREEN = '\033[92m'
	RED = '\033[31m'
	YELLOW = '\033[33m'
	ENDC = '\033[0m'


# function used in the multiprocessing pool to convert a single mp3 to wav
def mp_converter(filepath):
	command = f"ffmpeg -n -i \"{filepath}\" -ac 1 -acodec pcm_s16le -ar 16000 \"{filepath.parent / (filepath.stem + '.wav')}\" -loglevel error"

	ret = subprocess.run(command, capture_output=True)
	return ret.returncode, b"already exists" in ret.stderr, filepath.stem


# function to filter what files are not converted to wav
def filter_audio_files(mp3_files, wav_files):
	mp3_filenames = [mp3_file.stem for mp3_file in mp3_files]
	wav_filenames = [wav_file.stem for wav_file in wav_files]

	# create a set containing mp3 files that have no wav equivalent
	wav_not_existing = set(mp3_filenames) ^ set(wav_filenames)

	final_mp3_file_to_convert = []
	for mp3_file in mp3_files:
		if mp3_file.stem in wav_not_existing:
			final_mp3_file_to_convert.append(mp3_file)
	return final_mp3_file_to_convert


def convert2wav(path, delete_mp3=True, delete_wav=False, debug=False):
	if delete_wav:
		wav_files = Path(path).glob("*.wav")
		for wav_f in wav_files:
			wav_f.unlink()
		print(f"{TC.RED}Deleted previous wav files!{TC.ENDC}")

	mp3_files = list(Path(path).glob("*.mp3"))

	if len(mp3_files) == 0:
		print("No mp3 files exist")
		return

	wav_files = list(Path(path).glob("*.wav"))
	final_files = filter_audio_files(mp3_files, wav_files)

	print("Total mp3 files: {0}".format(len(mp3_files)))
	print("Total wav files: {0}".format(len(wav_files)))
	print("Converting {0} mp3 files to wav files".format(len(final_files)))

	if len(final_files) > 0:
		workers = multiprocessing.cpu_count()
		with multiprocessing.Pool(workers) as p:
			for return_code, existing, filename in tqdm(p.imap_unordered(mp_converter, final_files), total=len(final_files)):
				if debug:
					if return_code == 0:
						print(f"{filename}: {TC.GREEN}converted!{TC.ENDC}")
					else:
						if existing:
							print(f"{filename}: {TC.YELLOW}Already exists skipping{TC.ENDC}")
						else:
							print(f"{filename}: {TC.RED}ERROR:ffmpeg return code: {return_code}{TC.ENDC}")

			print(f"{TC.GREEN}Done!{TC.ENDC}")

	if delete_mp3:
		for mp3_f in mp3_files:
			mp3_f.unlink()
		print(f"{TC.RED}Deleted all mp3 files!{TC.ENDC}")
