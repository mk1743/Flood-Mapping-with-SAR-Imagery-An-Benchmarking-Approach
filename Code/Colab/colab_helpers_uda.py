import subprocess
import os

unzip = True

from tqdm import tqdm

req_path = "Code/Requirements/req.txt"

subprocess.check_call(['pip', 'install', '-r', req_path])
subprocess.check_call(['apt', "install", "rar"])

if unzip:

    import patoolib
    from tqdm import tqdm

    data_list = ["/content/drive/MyDrive/Masterarbeit/Code/Data/UrbanSARFloods/02_FO_original.rar", "/content/drive/MyDrive/Masterarbeit/Code/Data/UrbanSARFloods/03_FU_original.rar"]

    for i in tqdm(range(len(data_list))):
        unzip_path = data_list[i]

        name_base = os.path.basename(unzip_path)  # Gibt "02_FO_original.rar" zurück
        name = name_base[:-12] 

        print(f" \n Unzip Data of Domain {name}")

        output_folder = "/content"

        patoolib.extract_archive(archive = unzip_path, outdir=output_folder)


