import os
import pandas as pd

def create_dataset_from_folders():
    # 1. Define where your extracted folders are
    # IMPORTANT: Change this if your folders are named differently!
    root_path = 'data/raw' 
    
    # List of expected categories (folder names)
    categories = ['business', 'entertainment', 'politics', 'sport', 'tech']
    
    data = []
    
    print(f"Scanning folders in '{root_path}'...")
    
    # 2. Loop through each category folder
    for category in categories:
        folder_path = os.path.join(root_path, category)
        
        if not os.path.exists(folder_path):
            print(f"⚠️ Warning: Folder '{folder_path}' not found. Skipping...")
            continue
            
        print(f"Processing category: {category}...")
        
        # 3. Read every .txt file in that folder
        file_names = os.listdir(folder_path)
        
        for file_name in file_names:
            file_path = os.path.join(folder_path, file_name)
            
            try:
                # 'latin-1' encoding is safer for these old text files
                with open(file_path, 'r', encoding='latin-1') as f:
                    text = f.read()
                    data.append({'text': text, 'category': category})
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

    # 4. Save to CSV
    if len(data) > 0:
        df = pd.DataFrame(data)
        output_path = 'data/raw/bbc-text.csv'
        df.to_csv(output_path, index=False)
        print(f"\n✅ Success! Created '{output_path}' with {len(df)} articles.")
        print(df.head())
    else:
        print("\n❌ Error: No data found! Check your folder structure.")

if __name__ == "__main__":
    create_dataset_from_folders()