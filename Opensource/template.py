"""
Python script to create the required project structure

"""
import os

dirs = [
    os.path.join("data", "raw"),
    os.path.join("data", "processed"),
    "notebooks",
    "saved_models",
    "src",
    "report"
]

for dir in dirs:
    os.makedirs(dir, exist_ok=True)
    with open(os.path.join(dir, ".gitkeep"), 'w') as f:
        pass


file_ = [
    "dvc.yaml",
    "params.yaml",
    ".gitignore",
    os.path.join('src', '__init__.py')
]

for f in file_:
    with open(f, 'w') as file:
        pass
