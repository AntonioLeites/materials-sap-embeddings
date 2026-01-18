"""
Complete duplicate detection example
"""

import sys
sys.path.append('..')

from src.embeddings import RPT1Embeddings
from src.similarity import DuplicateDetector
from src.sap_connector import create_sample_sap_data
import pandas as pd


def main():
    print("=" * 60)
    print("Duplicate Detection Example")
    print("=" * 60)
    
    # 1. Generate sample data
    print("\n1. Generating sample material data...")
    df = create_sample_sap_data(n_materials=50)
    print(f"   Created {len(df)} materials")
    print(f"\n   Sample materials:")
    print(df[['MATNR', 'MAKTX']].head())
    
    # 2. Initialize RPT-1
    print("\n2. Initializing RPT-1...")
    rpt1 = RPT1Embeddings()
    
    # 3. Generate embeddings
    print("\n3. Generating embeddings...")
    materials = df['MAKTX'].tolist()
    embeddings = rpt1.encode_batch(materials, batch_size=16)
    print(f"   Generated embeddings: {embeddings.shape}")
    
    # 4. Detect duplicates
    print("\n4. Detecting duplicates...")
    detector = DuplicateDetector(threshold=0.85)
    duplicates = detector.find_duplicates(materials, embeddings=embeddings)
    
    print(f"   Found {len(duplicates)} potential duplicate pairs")
    
    # 5. Create report
    if duplicates:
        print("\n5. Creating duplicate report...")
        report = detector.create_duplicate_report(materials, duplicates[:10])
        print("\n   Top 10 duplicate pairs:")
        print(report.to_string(index=False))
        
        # Save report
        report_path = "duplicate_report.csv"
        report.to_csv(report_path, index=False)
        print(f"\n   ✓ Full report saved to: {report_path}")
    else:
        print("\n   No duplicates found above threshold")
    
    print("\n" + "=" * 60)
    print("✓ Example completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()