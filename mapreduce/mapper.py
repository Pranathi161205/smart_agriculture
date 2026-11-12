#!/usr/bin/env python3

import sys
from pathlib import PurePosixPath

for line in sys.stdin:
    path = line.strip()

    if not path:
        continue

    parts = PurePosixPath(path).parts

    if len(parts) < 2:
        continue

    category = parts[-2]

    # Emit category-wise count
    print(f"CATEGORY|{category}\t1")

    # Identify healthy and diseased images
    if "healthy" in category.lower():
        print("STATUS|Healthy\t1")
    else:
        print("STATUS|Diseased\t1")
