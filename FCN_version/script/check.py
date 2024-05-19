import importlib

# List of required packages
required_packages = [
    "python==3.8.18",
    "torch==1.21.1",
    "torchvision==0.13.1",
    "imageio==2.22.4",
    "numpy==1.14.0",
    "tqdm==4.64.1"
]

def check_packages():
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package.split("==")[0])
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("The following packages are missing:")
        for package in missing_packages:
            print(package)
    else:
        print("All required packages are installed.")

if __name__ == "__main__":
    check_packages()