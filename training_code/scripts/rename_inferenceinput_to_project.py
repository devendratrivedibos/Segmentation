import os

# List all folders to process
folders = [
r"X:\BURHANPUR_MAY 2026\MHMPBORDER-SHINDHIBASTI_2026-05-17_09-35-46\SECTION-2\MASKS_NEW",
]

for folder_path in folders:
    # Extract section name from folder path
    section = os.path.basename(os.path.dirname(folder_path))  # e.g., "SECTION-1"
    project_name = os.path.basename(os.path.dirname(os.path.dirname(folder_path)))  # e.g., "SIDDHATEK-KORTI_2025-06-21_13-13-05"

    for name in os.listdir(folder_path):
        if name.startswith("inference_input-"):
            try:
                old_path = os.path.join(folder_path, name)
                new_name = name.replace("inference_input-", f"{project_name}_{section}_", 1)
                new_path = os.path.join(folder_path, new_name)

                # Print before renaming
                print(f"Renaming: {name} -> {new_name}")

                # Perform rename safely
                os.rename(old_path, new_path)
            except Exception as e:
                print(f"Error renaming {name}: {e}")