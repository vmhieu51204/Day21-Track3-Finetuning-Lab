#!/usr/bin/env python3
"""Dong goi bai nop Lab 21 theo Option A trong rubric.md

Cach dung:
    python scripts/package_zip.py <MSSV>
    Vi du: python scripts/package_zip.py 20210001
"""
import os
import sys
import zipfile

def main():
    if len(sys.argv) < 2:
        print("Huong dan: python scripts/package_zip.py <MSSV>")
        print("Vi du:     python scripts/package_zip.py 20210001")
        sys.exit(1)

    mssv = sys.argv[1].strip()
    zip_filename = f"lab21_{mssv}.zip"
    prefix = f"lab21_{mssv}"

    is_option_b = "--option-b" in sys.argv or "-b" in sys.argv

    print(f"Dang dong goi {zip_filename} ({'Option B: GitHub + HF Hub' if is_option_b else 'Option A: ZIP gon'})...")
    with zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. submission/REPORT.md
        zf.write("submission/REPORT.md", f"{prefix}/submission/REPORT.md")

        # 2. results/
        if os.path.exists("results"):
            for fname in sorted(os.listdir("results")):
                if not fname.startswith("."):
                    zf.write(f"results/{fname}", f"{prefix}/results/{fname}")

        # 3. LINKS.md (neu co hoac neu dung Option B)
        if os.path.exists("LINKS.md"):
            zf.write("LINKS.md", f"{prefix}/LINKS.md")

        # 4. adapters/correct/ & notebooks/ (chi danh cho Option A)
        if not is_option_b:
            for fname in ["adapter_model.safetensors", "adapter_config.json"]:
                fpath = os.path.join("adapters", "correct", fname)
                if os.path.exists(fpath):
                    zf.write(fpath, f"{prefix}/adapters/correct/{fname}")

            if os.path.exists("notebooks"):
                for fname in sorted(os.listdir("notebooks")):
                    if fname.endswith((".py", ".ipynb")) and not fname.startswith("."):
                        zf.write(f"notebooks/{fname}", f"{prefix}/notebooks/{fname}")

    size_kb = os.path.getsize(zip_filename) / 1024
    if size_kb >= 1024:
        print(f"Thanh cong! Da tao file: {zip_filename} ({size_kb / 1024:.1f} MB)")
    else:
        print(f"Thanh cong! Da tao file: {zip_filename} ({size_kb:.1f} KB)")

if __name__ == "__main__":
    main()
