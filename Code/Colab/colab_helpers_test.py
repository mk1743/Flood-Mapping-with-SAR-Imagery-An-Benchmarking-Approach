import subprocess

unzip = True

req_path = "Code/Requirements/req.txt"

subprocess.check_call(['pip', 'install','-r', req_path])
subprocess.check_call(["apt", "install", "rar"]) # for rar archiv 

if unzip:

    import patoolib

    source_folder = "/content/drive/MyDrive/Masterarbeit/Code/Data/UrbanSARFloods/testing_case_orig.rar"

    output_folder = "/content"

    patoolib.extract_archive(archive = source_folder, outdir=output_folder, verbosity = 1)


