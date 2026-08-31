import os
import zipfile

def package_project(output_filename: str = "university_admissions_rag_assistant.zip"):
    """
    Packages all project source files, documents, tests, configs, and README into a clean .zip archive.
    Excludes temporary virtualenv files, pycache, and git histories.
    """
    root_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(root_dir, output_filename)

    ignore_dirs = {"venv", ".venv", "__pycache__", ".git", ".pytest_cache"}
    ignore_extensions = {".pyc", ".pyo", ".zip"}

    file_count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(root_dir):
            # Exclude ignored directories
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            for file in files:
                ext = os.path.splitext(file)[1]
                if ext in ignore_extensions:
                    continue

                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, root_dir)

                # Skip the output zip file itself if located in root
                if os.path.abspath(full_path) == os.path.abspath(zip_path):
                    continue

                zipf.write(full_path, rel_path)
                file_count += 1

    print(f"Successfully created packaging deliverable '{output_filename}' containing {file_count} files.")
    return zip_path

if __name__ == "__main__":
    package_project()
