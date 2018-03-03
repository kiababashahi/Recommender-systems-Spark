import subprocess, os

def test_basic_als():
    command="python ./answers/basic_als_recommender.py 123"
    process = subprocess.Popen(command, shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    code=process.wait()
    print(process.stderr.read())
    assert(not code), "Command failed"
    assert(abs(float(process.stdout.read().decode("utf-8"))-1.60)<0.03)

