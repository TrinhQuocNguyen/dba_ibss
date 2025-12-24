"""
Quick script to verify the column order in survey_data_complete.csv
Run this after generating data to confirm structure
"""

import pandas as pd

def verify_column_order(csv_file='research_data/survey_data_complete.csv'):
    """
    Verify that CSV columns are in the correct order
    """
    
    print("="*80)
    print("VERIFYING CSV COLUMN ORDER")
    print("="*80)
    
    # Load CSV
    try:
        df = pd.read_csv(csv_file)
        print(f"\n✓ Loaded: {csv_file}")
        print(f"  Shape: {df.shape}")
    except FileNotFoundError:
        print(f"\n✗ File not found: {csv_file}")
        print("  Please run: python 1_generate_enhanced.py")
        return False
    
    # Expected column order
    expected_order = [
        # 1. ID and Demographics (10)
        'Participant_ID', 'Country', 'Age', 'Gender', 'Position_Level', 
        'Tenure_Years', 'Education', 'Industry', 'Org_Size_Category', 'Org_Size_Numeric',
        
        # 2. LRAIT Dimension Scores (4)
        'TC_Score', 'CMC_Score', 'EA_Score', 'ALO_Score',
        
        # 3. LRAIT Items (32)
        'TC1', 'TC2', 'TC3', 'TC4', 'TC5', 'TC6', 'TC7', 'TC8',
        'CMC1', 'CMC2', 'CMC3', 'CMC4', 'CMC5', 'CMC6', 'CMC7', 'CMC8',
        'EA1', 'EA2', 'EA3', 'EA4', 'EA5', 'EA6', 'EA7', 'EA8',
        'ALO1', 'ALO2', 'ALO3', 'ALO4', 'ALO5', 'ALO6', 'ALO7', 'ALO8',
        
        # 4. Outcome Scores (4)
        'OI_Score', 'SA_Score', 'OL_Score', 'Overall_Success',
        
        # 5. Outcome Items (12)
        'OI1', 'OI2', 'OI3', 'OI4',
        'SA1', 'SA2', 'SA3', 'SA4',
        'OL1', 'OL2', 'OL3', 'OL4',
        
        # 6. Cultural Value Scores (4)
        'PD_Score', 'UA_Score', 'Collectivism_Score', 'LTO_Score',
        
        # 7. Cultural Value Items (12)
        'PD1', 'PD2', 'PD3',
        'UA1', 'UA2', 'UA3',
        'IC1', 'IC2', 'IC3',
        'LTO1', 'LTO2', 'LTO3',
        
        # 8. Survey Date (1)
        'Survey_Date'
    ]
    
    actual_columns = list(df.columns)
    
    # Check if all expected columns exist
    missing_cols = [col for col in expected_order if col not in actual_columns]
    extra_cols = [col for col in actual_columns if col not in expected_order]
    
    print("\n" + "="*80)
    print("COLUMN CHECK")
    print("="*80)
    
    if missing_cols:
        print(f"\n✗ Missing columns ({len(missing_cols)}):")
        for col in missing_cols:
            print(f"   - {col}")
    
    if extra_cols:
        print(f"\n✗ Extra columns ({len(extra_cols)}):")
        for col in extra_cols:
            print(f"   + {col}")
    
    if not missing_cols and not extra_cols:
        print("\n✓ All expected columns present")
        print(f"✓ Total columns: {len(actual_columns)}")
    
    # Check order
    print("\n" + "="*80)
    print("ORDER VERIFICATION")
    print("="*80)
    
    mismatches = []
    for i, expected_col in enumerate(expected_order, 1):
        if i <= len(actual_columns):
            actual_col = actual_columns[i-1]
            if expected_col != actual_col:
                mismatches.append((i, expected_col, actual_col))
    
    if not mismatches:
        print("\n✓✓✓ COLUMN ORDER IS PERFECT!")
        print("\nColumn structure:")
        print("  Columns   1-10: ID & Demographics")
        print("  Columns  11-14: LRAIT Scores (TC, CMC, EA, ALO)")
        print("  Columns  15-46: LRAIT Items (32 items)")
        print("  Columns  47-50: Outcome Scores")
        print("  Columns  51-62: Outcome Items (12 items)")
        print("  Columns  63-66: Cultural Scores")
        print("  Columns  67-78: Cultural Items (12 items)")
        print("  Column     79: Survey Date")
        print(f"\n  Total: {len(actual_columns)} columns")
        
        # Show sample of first few columns
        print("\n" + "="*80)
        print("FIRST 20 COLUMNS:")
        print("="*80)
        for i in range(min(20, len(actual_columns))):
            print(f"  {i+1:2d}. {actual_columns[i]}")
        
        return True
        
    else:
        print(f"\n✗ COLUMN ORDER MISMATCH!")
        print(f"\nFound {len(mismatches)} mismatches:")
        print("\n{:>4s}  {:25s}  {:25s}".format("Pos", "Expected", "Actual"))
        print("-"*60)
        
        for pos, expected, actual in mismatches[:20]:  # Show first 20
            print(f"{pos:4d}. {expected:25s}  {actual:25s}")
        
        if len(mismatches) > 20:
            print(f"\n... and {len(mismatches) - 20} more mismatches")
        
        print("\nTO FIX:")
        print("  1. Make sure you're using the UPDATED generator")
        print("  2. The generator should have column reordering code")
        print("  3. Look for: quant_data = quant_data[column_order]")
        
        return False
    
    print("="*80)


def show_column_groups(csv_file='research_data/survey_data_complete.csv'):
    """
    Show detailed column grouping
    """
    
    df = pd.read_csv(csv_file)
    cols = list(df.columns)
    
    print("\n" + "="*80)
    print("DETAILED COLUMN STRUCTURE")
    print("="*80)
    
    groups = [
        ("ID & Demographics", 0, 10),
        ("LRAIT Dimension Scores", 10, 14),
        ("TC Items", 14, 22),
        ("CMC Items", 22, 30),
        ("EA Items", 30, 38),
        ("ALO Items", 38, 46),
        ("Outcome Scores", 46, 50),
        ("OI Items", 50, 54),
        ("SA Items", 54, 58),
        ("OL Items", 58, 62),
        ("Cultural Scores", 62, 66),
        ("PD Items", 66, 69),
        ("UA Items", 69, 72),
        ("IC Items", 72, 75),
        ("LTO Items", 75, 78),
        ("Survey Date", 78, 79)
    ]
    
    for group_name, start, end in groups:
        print(f"\n{group_name} (columns {start+1}-{end}):")
        for i in range(start, min(end, len(cols))):
            print(f"  {i+1:2d}. {cols[i]}")
    
    print("="*80)


if __name__ == "__main__":
    # Verify order
    is_correct = verify_column_order()
    
    # If correct, show detailed structure
    if is_correct:
        show_column_groups()
    
    print("\n" + "="*80)
    if is_correct:
        print("✓ CSV structure is perfect! Ready for analysis.")
    else:
        print("✗ Please regenerate data with updated generator.")
    print("="*80)