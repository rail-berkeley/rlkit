import sys
import subprocess

cmd = F"aws s3 sync --exact-timestamp s3://richard-doodad-buck/doodad/01-16-FetchStack1-v1 ../s3_files/"
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
print(cmd)
for line in iter(process.stdout.readline, b''):
    sys.stdout.buffer.write(line)