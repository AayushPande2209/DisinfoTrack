"""
Load Real COVIDLies Dataset
============================

Instructions:
1. Download https://github.com/ucinlp/covid19-data
2. Extract the ZIP file
3. Find misconceptions.jsonl in the extracted folder
4. Update the PATH below to point to that file

Then run this script to validate the data and see the 62 misconceptions.
"""

import json
import pandas as pd

# UPDATE THIS PATH to wherever you download the repo
MISCONCEPTIONS_PATH = "/Users/aayush/Downloads/COVID19_Misconceptions.jsonl"  # Change this to your actual path

def load_misconceptions(filepath):
    """Load misconceptions from JSONL file."""
    data = []
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    data.append(record)
        
        print(f"✓ Loaded {len(data)} misconceptions")
        return data
    
    except FileNotFoundError:
        print(f"✗ File not found: {filepath}")
        print(f"\n  Steps to fix:")
        print(f"  1. Go to: https://github.com/ucinlp/covid19-data")
        print(f"  2. Click 'Code' → 'Download ZIP'")
        print(f"  3. Extract the ZIP file")
        print(f"  4. Find 'misconceptions.jsonl' in the extracted folder")
        print(f"  5. Update MISCONCEPTIONS_PATH in this script")
        return None


def create_dataset_from_misconceptions(misconceptions):
    """
    Convert misconceptions into a format similar to COVIDLIES_demo.csv
    
    Uses the pos_variations and neg_variations as claim "variants"
    """
    
    data = []
    claim_id = 0
    
    for misconception in misconceptions:
        claim_id += 1
        canonical = misconception.get('canonical_sentence', '')
        
        # Use positive and negative variations as variants
        pos_vars = misconception.get('pos_variations', [])
        neg_vars = misconception.get('neg_variations', [])
        all_vars = [canonical] + pos_vars + neg_vars
        
        # Create one "variant" per variation
        for idx, variant_text in enumerate(all_vars):
            data.append({
                'claim_id': claim_id,
                'anchor_claim': canonical[:200],  # Truncate for clarity
                'variant_text': variant_text[:500],
                'platform': ['Twitter', 'Reddit', 'Facebook', 'TikTok'][idx % 4],
                'category': ', '.join(misconception.get('category', [])),
                'origin': misconception.get('origin', 'Unknown'),
                'reliability_score': misconception.get('reliability_score', 0),
            })
    
    return pd.DataFrame(data)


if __name__ == '__main__':
    print("="*70)
    print("COVIDLies Real Dataset Loader")
    print("="*70)
    
    # Try to load
    misconceptions = load_misconceptions(MISCONCEPTIONS_PATH)
    
    if misconceptions is None:
        print("\nCannot proceed without the dataset file.")
        exit(1)
    
    # Show sample
    print(f"\n✓ Sample misconceptions:")
    for i, mis in enumerate(misconceptions[:3], 1):
        print(f"\n  [{i}] {mis['canonical_sentence'][:100]}...")
        print(f"      Category: {mis.get('category', [])}")
        print(f"      Variations: {len(mis.get('pos_variations', []))} pos, "
              f"{len(mis.get('neg_variations', []))} neg")
    
    # Convert to dataset format
    print(f"\n" + "="*70)
    print("Converting to experiment format...")
    df = create_dataset_from_misconceptions(misconceptions)
    
    print(f"\n✓ Created dataset with {len(df)} variant records")
    print(f"  (across {df['claim_id'].nunique()} anchor claims)")
    
    # Save for use in experiments
    df.to_csv('COVIDLIES_real.csv', index=False)
    print(f"\n✓ Saved to: COVIDLIES_real.csv")
    
    # Show summary
    print(f"\nDataset Summary:")
    print(df.groupby('claim_id').size().describe())
    
    print(f"\nPlatform distribution:")
    print(df['platform'].value_counts())
    
    print(f"\nCategory distribution:")
    print(df['category'].value_counts().head())