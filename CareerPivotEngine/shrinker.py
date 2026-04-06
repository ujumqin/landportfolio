import pandas as pd
import json
import os
# Get the folder where THIS script is saved
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create the full path to your JSON
json_path = os.path.join(script_dir, 'archetype_mapping.json')
parquet_path = os.path.join(script_dir, 'archetype_mapping.parquet')

print(f"Opening: {json_path}")

# Load your huge JSON
with open(json_path, 'r') as f:
    mapping = json.load(f)

# Flatten it into a list for a DataFrame
rows = []
for arch_id, courses in mapping.items():
    for c in courses:
        rows.append({
            'archetype_id': arch_id,
            'title': c['title'],
            'url': c['url'],
            'embedding': c['embedding']
        })

df = pd.DataFrame(rows)

# Save as Parquet (This will likely shrink it from 235MB to ~40MB)
df.to_parquet(parquet_path, compression='snappy')
print(f"Compressed mapping saved to: {parquet_path}")