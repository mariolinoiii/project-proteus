"""Project Proteus v3 dependency setup. Never installs Python or drivers."""
from pathlib import Path
import subprocess, sys
def main():
    print("Project Proteus v3: installing application dependencies",flush=True)
    subprocess.check_call([sys.executable,"-m","pip","install","-r",str(Path(__file__).resolve().parent/"requirements.txt")])
    subprocess.check_call([sys.executable,"-c","import numpy,torch,rasterio,scipy,pyproj,matplotlib,PIL"])
    (Path(sys.prefix)/".proteus_ready").write_text("v3",encoding="utf-8")
if __name__=="__main__": main()
