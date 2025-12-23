"""
Demographic Verification Script
Verifies that participants in both qualitative and quantitative phases
have matching demographic characteristics
"""

import pandas as pd
import json

def verify_demographic_matching(data_dir='research_data'):
    """
    Verify that all linked participants have matching demographics
    """
    
    print("="*80)
    print("DEMOGRAPHIC MATCHING VERIFICATION")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    survey_data = pd.read_csv(f'{data_dir}/survey_data_complete.csv')
    interview_data = pd.read_csv(f'{data_dir}/interview_metadata.csv')
    
    with open(f'{data_dir}/participant_linkage_masked.json', 'r') as f:
        linkage = json.load(f)
    
    print(f"✓ Loaded {len(survey_data)} survey responses")
    print(f"✓ Loaded {len(interview_data)} interviews")
    print(f"✓ Found {len(linkage)} linked participants")
    
    # Verify each linked participant
    print("\n" + "="*80)
    print("DETAILED VERIFICATION")
    print("="*80)
    
    all_match = True
    mismatches = []
    
    for qual_id, link_info in linkage.items():
        quant_id = link_info['quant_id']
        link_key = link_info['link_key']
        
        # Get records
        qual_record = interview_data[interview_data['Interview_ID'] == qual_id]
        quant_record = survey_data[survey_data['Participant_ID'] == quant_id]
        
        if len(qual_record) == 0:
            print(f"✗ ERROR: Qual ID {qual_id} not found in interview data")
            all_match = False
            continue
            
        if len(quant_record) == 0:
            print(f"✗ ERROR: Quant ID {quant_id} not found in survey data")
            all_match = False
            continue
        
        qual_record = qual_record.iloc[0]
        quant_record = quant_record.iloc[0]
        
        # Check demographics
        age_match = qual_record['Age'] == quant_record['Age']
        gender_match = qual_record['Gender'] == quant_record['Gender']
        industry_match = qual_record['Industry'] == quant_record['Industry']
        country_match = qual_record['Country'] == quant_record['Country']
        link_key_match = (qual_record['Survey_Link_Key'] == link_key and 
                         quant_record['Survey_Link_Key'] == link_key)
        
        if not all([age_match, gender_match, industry_match, country_match, link_key_match]):
            all_match = False
            mismatches.append({
                'qual_id': qual_id,
                'quant_id': quant_id,
                'age_match': age_match,
                'gender_match': gender_match,
                'industry_match': industry_match,
                'country_match': country_match,
                'link_key_match': link_key_match,
                'qual_age': qual_record['Age'],
                'quant_age': quant_record['Age'],
                'qual_gender': qual_record['Gender'],
                'quant_gender': quant_record['Gender'],
                'qual_industry': qual_record['Industry'],
                'quant_industry': quant_record['Industry']
            })
    
    # Report results
    print("\n" + "="*80)
    print("VERIFICATION RESULTS")
    print("="*80)
    
    if all_match:
        print("\n✓✓✓ SUCCESS! ✓✓✓")
        print(f"All {len(linkage)} linked participants have perfectly matching demographics!")
        print("\nVerified attributes:")
        print("  ✓ Age")
        print("  ✓ Gender")
        print("  ✓ Industry")
        print("  ✓ Country")
        print("  ✓ Link Keys")
    else:
        print(f"\n✗✗✗ FAILURE! ✗✗✗")
        print(f"Found {len(mismatches)} mismatches out of {len(linkage)} linked participants")
        print("\nMismatch details:")
        for mismatch in mismatches[:10]:  # Show first 10
            print(f"\n  Qual ID: {mismatch['qual_id']}")
            print(f"  Quant ID: {mismatch['quant_id']}")
            if not mismatch['age_match']:
                print(f"    ✗ Age mismatch: Interview={mismatch['qual_age']}, Survey={mismatch['quant_age']}")
            if not mismatch['gender_match']:
                print(f"    ✗ Gender mismatch: Interview={mismatch['qual_gender']}, Survey={mismatch['quant_gender']}")
            if not mismatch['industry_match']:
                print(f"    ✗ Industry mismatch: Interview={mismatch['qual_industry']}, Survey={mismatch['quant_industry']}")
    
    # Display examples of matching participants
    print("\n" + "="*80)
    print("EXAMPLE MATCHED PARTICIPANTS")
    print("="*80)
    
    for i, (qual_id, link_info) in enumerate(list(linkage.items())[:5]):
        quant_id = link_info['quant_id']
        
        qual_record = interview_data[interview_data['Interview_ID'] == qual_id].iloc[0]
        quant_record = survey_data[survey_data['Participant_ID'] == quant_id].iloc[0]
        
        print(f"\n{i+1}. Link Key: {link_info['link_key']}")
        print(f"   Interview ({qual_id}):")
        print(f"     Age: {qual_record['Age']}, Gender: {qual_record['Gender']}, Industry: {qual_record['Industry']}, Country: {qual_record['Country']}")
        print(f"   Survey ({quant_id}):")
        print(f"     Age: {quant_record['Age']}, Gender: {quant_record['Gender']}, Industry: {quant_record['Industry']}, Country: {quant_record['Country']}")
        
        if (qual_record['Age'] == quant_record['Age'] and 
            qual_record['Gender'] == quant_record['Gender'] and 
            qual_record['Industry'] == quant_record['Industry']):
            print(f"   Status: ✓ MATCH")
        else:
            print(f"   Status: ✗ MISMATCH")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Count by country
    japan_links = sum(1 for k in linkage.keys() if k.startswith('JP'))
    vietnam_links = sum(1 for k in linkage.keys() if k.startswith('VN'))
    
    print(f"\nLinked participants by country:")
    print(f"  Japan: {japan_links}")
    print(f"  Vietnam: {vietnam_links}")
    print(f"  Total: {len(linkage)}")
    
    # Overlap percentage
    total_interviews = len(interview_data)
    overlap_pct = (len(linkage) / total_interviews) * 100
    print(f"\nOverlap rate: {overlap_pct:.1f}% of interview participants also completed survey")
    
    print("\n" + "="*80)
    
    return all_match, mismatches


def generate_detailed_report(data_dir='research_data', output_file='demographic_match_report.txt'):
    """
    Generate a detailed text report of demographic matching
    """
    
    survey_data = pd.read_csv(f'{data_dir}/survey_data_complete.csv')
    interview_data = pd.read_csv(f'{data_dir}/interview_metadata.csv')
    
    with open(f'{data_dir}/participant_linkage_masked.json', 'r') as f:
        linkage = json.load(f)
    
    lines = []
    lines.append("="*80)
    lines.append("DEMOGRAPHIC MATCHING VERIFICATION REPORT")
    lines.append("="*80)
    lines.append("")
    
    lines.append(f"Total survey responses: {len(survey_data)}")
    lines.append(f"Total interviews: {len(interview_data)}")
    lines.append(f"Linked participants: {len(linkage)}")
    lines.append("")
    
    lines.append("="*80)
    lines.append("COMPLETE VERIFICATION TABLE")
    lines.append("="*80)
    lines.append("")
    lines.append(f"{'Qual ID':<15} {'Quant ID':<17} {'Age':<6} {'Gender':<8} {'Industry':<20} {'Country':<10} {'Match':<8}")
    lines.append("-"*80)
    
    for qual_id, link_info in sorted(linkage.items()):
        quant_id = link_info['quant_id']
        
        qual_record = interview_data[interview_data['Interview_ID'] == qual_id].iloc[0]
        quant_record = survey_data[survey_data['Participant_ID'] == quant_id].iloc[0]
        
        age_match = "✓" if qual_record['Age'] == quant_record['Age'] else "✗"
        gender_match = "✓" if qual_record['Gender'] == quant_record['Gender'] else "✗"
        industry_match = "✓" if qual_record['Industry'] == quant_record['Industry'] else "✗"
        country_match = "✓" if qual_record['Country'] == quant_record['Country'] else "✗"
        
        all_match = "✓" if (age_match == "✓" and gender_match == "✓" and 
                            industry_match == "✓" and country_match == "✓") else "✗"
        
        lines.append(f"{qual_id:<15} {quant_id:<17} {age_match:<6} {gender_match:<8} {industry_match:<20} {country_match:<10} {all_match:<8}")
    
    lines.append("")
    lines.append("="*80)
    
    # Write report
    output_path = f'{data_dir}/{output_file}'
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"\n✓ Detailed report saved to: {output_path}")


if __name__ == "__main__":
    # Run verification
    all_match, mismatches = verify_demographic_matching()
    
    # Generate detailed report
    generate_detailed_report()
    
    # Exit with appropriate code
    if all_match:
        print("\n✓ Verification passed! All demographics match.")
        exit(0)
    else:
        print(f"\n✗ Verification failed! Found {len(mismatches)} mismatches.")
        exit(1)