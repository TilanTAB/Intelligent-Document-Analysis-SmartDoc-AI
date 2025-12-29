import os
import shutil
import glob

def clean_pycache():
    """
    Recursively finds and removes all __pycache__ directories and .pyc files
    within the current working directory.
    """
    print("Starting Python cache cleanup...")
    
    # Find and remove __pycache__ directories
    pycache_dirs = glob.glob('**/__pycache__', recursive=True)
    if not pycache_dirs:
        print("No __pycache__ directories found.")
    else:
        for path in pycache_dirs:
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    print(f"Removed directory: {path}")
            except OSError as e:
                print(f"Error removing directory {path}: {e}")

    # Find and remove remaining .pyc files (less common)
    pyc_files = glob.glob('**/*.pyc', recursive=True)
    if not pyc_files:
        print("No .pyc files found.")
    else:
        for path in pyc_files:
            try:
                if os.path.isfile(path):
                    os.remove(path)
                    print(f"Removed file: {path}")
            except OSError as e:
                print(f"Error removing file {path}: {e}")

    print("Cleanup complete.")

if __name__ == "__main__":
    clean_pycache()
