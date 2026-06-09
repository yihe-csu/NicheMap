# 📘 How to use NicheMap (Prerequisites)

## How to use NicheMap

Before using `NicheMap`, please make sure the following steps are completed:

- A Python environment (recommended: Python ≥ 3.10) is properly activated
- All required dependencies are installed
- The `NicheMap` package is accessible in your Python path

------

## 1. Activate environment

Using `conda`:

```
conda activate nichemap
```

Or using `venv`:

```
# Linux / Mac
source nichemap/bin/activate

# Windows
nichemap\Scripts\activate
```

------

## 2. Install dependencies

```
pip install -r requirements.txt
```

------

## 3. Add NicheMap to Python path (for local development)

If you are running NicheMap locally (without installation), add the project path before importing:

```
import sys
import os

sys.path.append(os.path.abspath("your/path/to/NicheMap"))
import nichemap
```

------

## 4. Verify installation

```
import nichemap

print("NicheMap is ready to use.")
```
