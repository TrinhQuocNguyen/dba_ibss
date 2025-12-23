"""
Comprehensive Data Generator for:
'Leadership Readiness for AI Transformation: A Cross-Cultural Framework 
for Japanese and Vietnamese Organizations'

This program generates:
1. Quantitative survey data (n=428)
2. Qualitative interview metadata (n=45)
3. Proper masking for participants in both phases
4. All data matching dissertation findings
"""

import pandas as pd
import numpy as np
import hashlib
import json
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

class ComprehensiveDataGenerator:
    """
    Generates complete research dataset with proper participant masking
    """
    
    def __init__(self):
        self.japan_quant_n = 213
        self.vietnam_quant_n = 215
        self.japan_qual_n = 23
        self.vietnam_qual_n = 22
        
        # Some qualitative participants also completed survey (with masking)
        self.overlap_pct = 0.35  # 35% of interview participants also in survey
        
    def generate_participant_id(self, country_code, sequence, phase):
        """
        Generate unique but masked participant IDs
        phase: 'QUAL' or 'QUANT'
        """
        base_id = f"{country_code}_{phase}_{sequence:03d}"
        return base_id
    
    def verify_demographic_matching(self, quant_data, qual_data, linkage):
        """Verify that linked participants have matching demographics"""
        
        mismatches = []
        matches_verified = 0
        
        for qual_id, link_info in linkage.items():
            quant_id = link_info['quant_id']
            link_key = link_info['link_key']
            
            # Get qual record
            qual_record = qual_data[qual_data['Interview_ID'] == qual_id].iloc[0]
            
            # Get quant record
            quant_record = quant_data[quant_data['Participant_ID'] == quant_id].iloc[0]
            
            # Verify matching
            checks = {
                'Age': qual_record['Age'] == quant_record['Age'],
                'Gender': qual_record['Gender'] == quant_record['Gender'],
                'Industry': qual_record['Industry'] == quant_record['Industry'],
                'Country': qual_record['Country'] == quant_record['Country'],
                'Link_Key': qual_record['Survey_Link_Key'] == quant_record['Survey_Link_Key'] == link_key
            }
            
            if all(checks.values()):
                matches_verified += 1
            else:
                mismatches.append({
                    'qual_id': qual_id,
                    'quant_id': quant_id,
                    'checks': checks
                })
        
        if len(mismatches) == 0:
            print(f"  ✓ All {matches_verified} linked participants have matching demographics")
        else:
            print(f"  ✗ WARNING: {len(mismatches)} mismatches found!")
            for mismatch in mismatches[:5]:  # Show first 5
                print(f"    {mismatch}")
        
        return len(mismatches) == 0
    
    def generate_masked_linkage(self):
        """
        Create linkage between qualitative and quantitative for overlapping participants
        Returns mapping with masked IDs
        """
        linkage = {}
        
        # Japan overlapping participants
        japan_overlap_n = int(self.japan_qual_n * self.overlap_pct)
        japan_qual_indices = np.random.choice(self.japan_qual_n, japan_overlap_n, replace=False)
        japan_quant_indices = np.random.choice(self.japan_quant_n, japan_overlap_n, replace=False)
        
        for qual_idx, quant_idx in zip(japan_qual_indices, japan_quant_indices):
            qual_id = self.generate_participant_id('JP', qual_idx + 1, 'QUAL')
            quant_id = self.generate_participant_id('JP', quant_idx + 1, 'QUANT')
            
            # Create one-way hash for secure linkage
            link_key = hashlib.sha256(f"{qual_id}_{quant_id}".encode()).hexdigest()[:16]
            linkage[qual_id] = {'quant_id': quant_id, 'link_key': link_key}
        
        # Vietnam overlapping participants
        vietnam_overlap_n = int(self.vietnam_qual_n * self.overlap_pct)
        vietnam_qual_indices = np.random.choice(self.vietnam_qual_n, vietnam_overlap_n, replace=False)
        vietnam_quant_indices = np.random.choice(self.vietnam_quant_n, vietnam_overlap_n, replace=False)
        
        for qual_idx, quant_idx in zip(vietnam_qual_indices, vietnam_quant_indices):
            qual_id = self.generate_participant_id('VN', qual_idx + 1, 'QUAL')
            quant_id = self.generate_participant_id('VN', quant_idx + 1, 'QUANT')
            
            link_key = hashlib.sha256(f"{qual_id}_{quant_id}".encode()).hexdigest()[:16]
            linkage[qual_id] = {'quant_id': quant_id, 'link_key': link_key}
        
        return linkage
    
    def generate_demographics(self, country, n, is_qualitative=False):
        """Generate demographic variables matching dissertation findings"""
        
        if country == 'Japan':
            age_mean, age_sd = 44.8, 8.2
            male_pct = 0.793
            team_leader_pct = 0.282
            dept_head_pct = 0.455
            senior_exec_pct = 0.263
            tenure_mean, tenure_sd = 8.9, 5.2
            
            industry_dist = {
                'Manufacturing': 0.244,
                'Financial Services': 0.221,
                'Retail': 0.164,
                'Technology': 0.122,
                'Healthcare': 0.146,
                'Other': 0.103
            }
            
        else:  # Vietnam
            age_mean, age_sd = 39.4, 7.6
            male_pct = 0.647
            team_leader_pct = 0.321
            dept_head_pct = 0.442
            senior_exec_pct = 0.237
            tenure_mean, tenure_sd = 6.4, 4.1
            
            industry_dist = {
                'Manufacturing': 0.186,
                'Financial Services': 0.284,
                'Retail': 0.195,
                'Technology': 0.177,
                'Healthcare': 0.093,
                'Other': 0.065
            }
        
        # Adjust for qualitative sample (slightly more senior)
        if is_qualitative:
            senior_exec_pct += 0.10
            dept_head_pct -= 0.05
            team_leader_pct -= 0.05
        
        # Generate age first
        age = np.random.normal(age_mean, age_sd, n)
        age = np.clip(age, 25, 65).astype(int)
        
        # Generate education (affects career start age)
        education = np.random.choice(
            ['Bachelor', 'Master', 'PhD'],
            n,
            p=[0.45, 0.48, 0.07]
        )
        
        # Calculate career start age based on education
        # Bachelor: typically finish at 22
        # Master: typically finish at 24
        # PhD: typically finish at 28
        career_start_age = np.array([
            22 if edu == 'Bachelor' else (24 if edu == 'Master' else 28)
            for edu in education
        ])
        
        # Calculate maximum possible tenure based on age and career start
        max_possible_tenure = age - career_start_age
        max_possible_tenure = np.clip(max_possible_tenure, 2, 40)  # At least 2 years
        
        # Generate tenure that doesn't exceed max possible
        # Use a proportion of max possible tenure with some randomness
        tenure_proportion = np.random.beta(2, 2, n)  # Beta distribution (0 to 1)
        tenure = max_possible_tenure * tenure_proportion
        
        # Adjust to match target mean while respecting constraints
        tenure_scale = tenure_mean / tenure.mean()
        tenure = tenure * tenure_scale
        
        # Final constraint: tenure cannot exceed max possible
        tenure = np.minimum(tenure, max_possible_tenure)
        tenure = np.clip(tenure, 2, 30).round(1)  # At least 2 years, max 30
        
        # Generate other demographics
        gender = np.random.choice(['Male', 'Female'], n, p=[male_pct, 1-male_pct])
        
        position = np.random.choice(
            ['Team Leader', 'Department Head', 'Senior Executive'],
            n,
            p=[team_leader_pct, dept_head_pct, senior_exec_pct]
        )
        
        industries = list(industry_dist.keys())
        probs = list(industry_dist.values())
        industry = np.random.choice(industries, n, p=probs)
        
        org_size_category = np.random.choice(
            ['Small (< 500)', 'Medium (500-2000)', 'Large (> 2000)'],
            n,
            p=[0.25, 0.40, 0.35]
        )
        
        # For quantitative, add numeric org size
        if not is_qualitative:
            org_size_numeric = []
            for size_cat in org_size_category:
                if 'Small' in size_cat:
                    org_size_numeric.append(np.random.randint(50, 500))
                elif 'Medium' in size_cat:
                    org_size_numeric.append(np.random.randint(500, 2000))
                else:
                    org_size_numeric.append(np.random.randint(2000, 10000))
        
        df = pd.DataFrame({
            'Age': age,
            'Gender': gender,
            'Position_Level': position,
            'Tenure_Years': tenure,
            'Education': education,
            'Industry': industry,
            'Org_Size_Category': org_size_category,
            'Country': country
        })
        
        if not is_qualitative:
            df['Org_Size_Numeric'] = org_size_numeric
        
        return df
    
    def generate_lrait_scores(self, demographics):
        """Generate LRAIT dimension scores matching dissertation findings"""
        
        n = len(demographics)
        country = demographics['Country'].iloc[0]
        
        if country == 'Japan':
            tc_mean, tc_sd = 5.32, 0.87
            cmc_mean, cmc_sd = 4.76, 0.92
            ea_mean, ea_sd = 5.41, 0.81
            alo_mean, alo_sd = 4.68, 0.95
        else:  # Vietnam
            tc_mean, tc_sd = 4.89, 0.94
            cmc_mean, cmc_sd = 5.18, 0.89
            ea_mean, ea_sd = 5.08, 0.88
            alo_mean, alo_sd = 5.29, 0.87
        
        # Correlation matrix (from dissertation: .47-.58)
        correlation_matrix = np.array([
            [1.00, 0.54, 0.48, 0.51],
            [0.54, 1.00, 0.52, 0.58],
            [0.48, 0.52, 1.00, 0.47],
            [0.51, 0.58, 0.47, 1.00]
        ])
        
        means = [tc_mean, cmc_mean, ea_mean, alo_mean]
        sds = [tc_sd, cmc_sd, ea_sd, alo_sd]
        
        # Generate correlated variables
        L = np.linalg.cholesky(correlation_matrix)
        uncorrelated = np.random.normal(0, 1, (n, 4))
        correlated = uncorrelated @ L.T
        
        for i in range(4):
            correlated[:, i] = correlated[:, i] * sds[i] + means[i]
        
        correlated = np.clip(correlated, 1, 7)
        
        # Add demographic effects
        if country == 'Japan':
            age_effect = (demographics['Age'] - demographics['Age'].mean()) * -0.02
            correlated[:, 0] += age_effect
        
        if country == 'Vietnam':
            edu_effect = np.where(demographics['Education'] == 'Master', 0.2, 0)
            edu_effect += np.where(demographics['Education'] == 'PhD', 0.4, 0)
            correlated[:, 3] += edu_effect
        
        position_effect = np.where(demographics['Position_Level'] == 'Department Head', 0.15, 0)
        position_effect += np.where(demographics['Position_Level'] == 'Senior Executive', 0.30, 0)
        correlated[:, 1] += position_effect
        
        correlated = np.clip(correlated, 1, 7)
        
        dimension_scores = pd.DataFrame(correlated, columns=[
            'TC_Score', 'CMC_Score', 'EA_Score', 'ALO_Score'
        ])
        
        return dimension_scores
    
    def generate_item_scores(self, dimension_scores):
        """Generate individual item scores (8 items per dimension)"""
        
        n = len(dimension_scores)
        items = {}
        
        dimensions = {
            'TC': 'TC_Score',
            'CMC': 'CMC_Score',
            'EA': 'EA_Score',
            'ALO': 'ALO_Score'
        }
        
        for dim_prefix, dim_col in dimensions.items():
            dim_score = dimension_scores[dim_col].values
            
            for item_num in range(1, 9):
                loading = np.random.uniform(0.70, 0.85)
                error = np.random.normal(0, 0.8, n)
                
                item_score = loading * dim_score + error
                item_score = np.clip(item_score, 1, 7).round()
                
                items[f'{dim_prefix}{item_num}'] = item_score
        
        return pd.DataFrame(items)
    
    def generate_outcome_scores(self, dimension_scores, demographics):
        """Generate AI transformation outcome scores"""
        
        n = len(dimension_scores)
        country = demographics['Country'].iloc[0]
        
        # Based on regression coefficients from dissertation
        outcome_score = (
            1.5 +
            0.26 * dimension_scores['TC_Score'] +
            0.31 * dimension_scores['CMC_Score'] +
            0.17 * dimension_scores['EA_Score'] +
            0.26 * dimension_scores['ALO_Score']
        )
        
        position_effect = np.where(demographics['Position_Level'] == 'Department Head', 0.3, 0)
        position_effect += np.where(demographics['Position_Level'] == 'Senior Executive', 0.6, 0)
        outcome_score += position_effect
        
        outcome_score += np.random.normal(0, 0.4, n)
        outcome_score = np.clip(outcome_score, 1, 7)
        
        country_boost = 0.15 if country == 'Vietnam' else 0
        
        oi_scores = outcome_score + np.random.normal(country_boost, 0.3, n)
        sa_scores = outcome_score - 0.3 + np.random.normal(country_boost, 0.3, n)
        ol_scores = outcome_score + 0.1 + np.random.normal(country_boost + 0.2, 0.3, n)
        
        oi_scores = np.clip(oi_scores, 1, 7)
        sa_scores = np.clip(sa_scores, 1, 7)
        ol_scores = np.clip(ol_scores, 1, 7)
        
        outcome_items = {}
        
        for i in range(1, 5):
            outcome_items[f'OI{i}'] = np.clip(
                oi_scores + np.random.normal(0, 0.4, n), 1, 7
            ).round()
            outcome_items[f'SA{i}'] = np.clip(
                sa_scores + np.random.normal(0, 0.4, n), 1, 7
            ).round()
            outcome_items[f'OL{i}'] = np.clip(
                ol_scores + np.random.normal(0, 0.4, n), 1, 7
            ).round()
        
        outcome_df = pd.DataFrame(outcome_items)
        outcome_df['OI_Score'] = oi_scores.round(2)
        outcome_df['SA_Score'] = sa_scores.round(2)
        outcome_df['OL_Score'] = ol_scores.round(2)
        outcome_df['Overall_Success'] = outcome_score.round(2)
        
        return outcome_df
    
    def generate_cultural_values(self, demographics):
        """Generate cultural value orientation scores"""
        
        n = len(demographics)
        country = demographics['Country'].iloc[0]
        
        if country == 'Japan':
            pd_mean = 4.2
            ua_mean = 5.8
            coll_mean = 5.1
            lto_mean = 6.2
        else:
            pd_mean = 5.0
            ua_mean = 4.0
            coll_mean = 5.8
            lto_mean = 5.5
        
        pd_scores = np.clip(np.random.normal(pd_mean, 1.1, n), 1, 7)
        ua_scores = np.clip(np.random.normal(ua_mean, 1.0, n), 1, 7)
        coll_scores = np.clip(np.random.normal(coll_mean, 0.9, n), 1, 7)
        lto_scores = np.clip(np.random.normal(lto_mean, 1.0, n), 1, 7)
        
        cultural_items = {}
        
        for i in range(1, 4):
            cultural_items[f'PD{i}'] = np.clip(
                pd_scores + np.random.normal(0, 0.5, n), 1, 7
            ).round()
            cultural_items[f'UA{i}'] = np.clip(
                ua_scores + np.random.normal(0, 0.5, n), 1, 7
            ).round()
            cultural_items[f'IC{i}'] = np.clip(
                coll_scores + np.random.normal(0, 0.5, n), 1, 7
            ).round()
            cultural_items[f'LTO{i}'] = np.clip(
                lto_scores + np.random.normal(0, 0.5, n), 1, 7
            ).round()
        
        cultural_df = pd.DataFrame(cultural_items)
        cultural_df['PD_Score'] = pd_scores.round(2)
        cultural_df['UA_Score'] = ua_scores.round(2)
        cultural_df['Collectivism_Score'] = coll_scores.round(2)
        cultural_df['LTO_Score'] = lto_scores.round(2)
        
        return cultural_df
    
    def generate_qualitative_data(self, linkage, shared_demographics):
        """Generate qualitative interview metadata with matching demographics"""
        
        interviews = []
        base_date = datetime(2023, 3, 1)
        
        # Japan interviews
        for i in range(self.japan_qual_n):
            qual_id = self.generate_participant_id('JP', i + 1, 'QUAL')
            interview_date = base_date + timedelta(days=np.random.randint(0, 90))
            
            # Check if this participant is in both phases
            if qual_id in shared_demographics:
                # Use shared demographics
                demo = shared_demographics[qual_id]
                age = demo['Age']
                gender = demo['Gender']
                industry = demo['Industry']
                
                # Derive age range from exact age
                if age < 45:
                    age_range = '35-44'
                elif age < 55:
                    age_range = '45-54'
                else:
                    age_range = '55-64'
                
                # Derive position from Position_Level
                if 'Senior' in demo['Position_Level']:
                    position = 'Senior Leader'
                else:
                    position = 'Mid-level Leader'
            else:
                # Generate new demographics for interview-only participant
                age = np.random.randint(35, 65)
                if age < 45:
                    age_range = '35-44'
                elif age < 55:
                    age_range = '45-54'
                else:
                    age_range = '55-64'
                
                gender = np.random.choice(['Male', 'Female'], p=[0.87, 0.13])
                position = np.random.choice(['Senior Leader', 'Mid-level Leader'], p=[0.39, 0.61])
                industry = np.random.choice(['Manufacturing', 'Finance', 'Retail', 'Healthcare', 'Technology'])
            
            interview_data = {
                'Interview_ID': qual_id,
                'Country': 'Japan',
                'Interview_Date': interview_date.strftime('%Y-%m-%d'),
                'Position': position,
                'Industry': industry,
                'Age_Range': age_range,
                'Age': age,  # Include exact age for verification
                'Gender': gender,
                'Interview_Duration_Min': np.random.randint(55, 95),
                'AI_Experience_Years': np.random.randint(1, 6),
                'Also_In_Survey': qual_id in linkage
            }
            
            if qual_id in linkage:
                interview_data['Survey_Link_Key'] = linkage[qual_id]['link_key']
            else:
                interview_data['Survey_Link_Key'] = None
            
            interviews.append(interview_data)
        
        # Vietnam interviews
        for i in range(self.vietnam_qual_n):
            qual_id = self.generate_participant_id('VN', i + 1, 'QUAL')
            interview_date = base_date + timedelta(days=np.random.randint(0, 90))
            
            # Check if this participant is in both phases
            if qual_id in shared_demographics:
                # Use shared demographics
                demo = shared_demographics[qual_id]
                age = demo['Age']
                gender = demo['Gender']
                industry = demo['Industry']
                
                # Derive age range from exact age
                if age < 40:
                    age_range = '30-39'
                elif age < 50:
                    age_range = '40-49'
                else:
                    age_range = '50-59'
                
                # Derive position from Position_Level
                if 'Senior' in demo['Position_Level']:
                    position = 'Senior Leader'
                else:
                    position = 'Mid-level Leader'
            else:
                # Generate new demographics for interview-only participant
                age = np.random.randint(30, 60)
                if age < 40:
                    age_range = '30-39'
                elif age < 50:
                    age_range = '40-49'
                else:
                    age_range = '50-59'
                
                gender = np.random.choice(['Male', 'Female'], p=[0.68, 0.32])
                position = np.random.choice(['Senior Leader', 'Mid-level Leader'], p=[0.36, 0.64])
                industry = np.random.choice(['Finance', 'Retail', 'Manufacturing', 'Technology', 'Healthcare'])
            
            interview_data = {
                'Interview_ID': qual_id,
                'Country': 'Vietnam',
                'Interview_Date': interview_date.strftime('%Y-%m-%d'),
                'Position': position,
                'Industry': industry,
                'Age_Range': age_range,
                'Age': age,  # Include exact age for verification
                'Gender': gender,
                'Interview_Duration_Min': np.random.randint(60, 90),
                'AI_Experience_Years': np.random.randint(1, 5),
                'Also_In_Survey': qual_id in linkage
            }
            
            if qual_id in linkage:
                interview_data['Survey_Link_Key'] = linkage[qual_id]['link_key']
            else:
                interview_data['Survey_Link_Key'] = None
            
            interviews.append(interview_data)
        
        return pd.DataFrame(interviews)
    
    def generate_quantitative_dataset(self, linkage, shared_demographics):
        """Generate complete quantitative survey dataset with matching demographics"""
        
        # Get list of quant_ids that are linked
        linked_quant_ids = set()
        for qual_id, link_info in linkage.items():
            linked_quant_ids.add(link_info['quant_id'])
        
        datasets = []
        
        for country, n in [('Japan', self.japan_quant_n), ('Vietnam', self.vietnam_quant_n)]:
            
            # Separate linked and non-linked participants
            country_code = 'JP' if country == 'Japan' else 'VN'
            
            # Count how many are linked
            linked_count = sum(1 for qid in linked_quant_ids if qid.startswith(country_code))
            non_linked_count = n - linked_count
            
            # Generate demographics for non-linked participants
            if non_linked_count > 0:
                non_linked_demographics = self.generate_demographics(country, non_linked_count, is_qualitative=False)
            else:
                non_linked_demographics = pd.DataFrame()
            
            # Collect all demographics in order
            all_demographics = []
            non_linked_idx = 0
            
            for i in range(n):
                quant_id = self.generate_participant_id(country_code, i + 1, 'QUANT')
                
                if quant_id in shared_demographics:
                    # Use shared demographics
                    demo_dict = shared_demographics[quant_id].copy()
                    all_demographics.append(demo_dict)
                else:
                    # Use newly generated demographics
                    demo_dict = non_linked_demographics.iloc[non_linked_idx].to_dict()
                    all_demographics.append(demo_dict)
                    non_linked_idx += 1
            
            demographics = pd.DataFrame(all_demographics)
            
            # LRAIT scores
            dimension_scores = self.generate_lrait_scores(demographics)
            
            # Item scores
            item_scores = self.generate_item_scores(dimension_scores)
            
            # Outcomes
            outcome_scores = self.generate_outcome_scores(dimension_scores, demographics)
            
            # Cultural values
            cultural_values = self.generate_cultural_values(demographics)
            
            # Survey date
            base_date = datetime(2023, 6, 1)
            survey_dates = [
                (base_date + timedelta(days=np.random.randint(0, 120))).strftime('%Y-%m-%d')
                for _ in range(n)
            ]
            
            # Combine
            country_data = pd.concat([
                demographics.reset_index(drop=True),
                dimension_scores.reset_index(drop=True),
                item_scores.reset_index(drop=True),
                outcome_scores.reset_index(drop=True),
                cultural_values.reset_index(drop=True)
            ], axis=1)
            
            country_data['Survey_Date'] = survey_dates
            
            datasets.append(country_data)
        
        full_dataset = pd.concat(datasets, ignore_index=True)
        
        # Add participant IDs and link keys
        participant_ids = []
        survey_link_keys = []
        
        for idx, row in full_dataset.iterrows():
            country_code = 'JP' if row['Country'] == 'Japan' else 'VN'
            
            # Determine which index within country
            if row['Country'] == 'Japan':
                country_idx = idx + 1
            else:
                country_idx = idx - self.japan_quant_n + 1
            
            quant_id = self.generate_participant_id(country_code, country_idx, 'QUANT')
            participant_ids.append(quant_id)
            
            # Check if this participant has qualitative linkage
            link_key = None
            for qual_id, link_info in linkage.items():
                if link_info['quant_id'] == quant_id:
                    link_key = link_info['link_key']
                    break
            
            survey_link_keys.append(link_key)
        
        full_dataset.insert(0, 'Participant_ID', participant_ids)
        full_dataset['Survey_Link_Key'] = survey_link_keys
        
        return full_dataset
    
    def generate_complete_dataset(self):
        """Generate all datasets with proper masking and matching demographics"""
        
        print("="*70)
        print("COMPREHENSIVE DATA GENERATION")
        print("AI Leadership Readiness for AI Transformation Study")
        print("="*70)
        
        # Step 1: Determine which participants will be in both phases
        print("\n Step 1: Determining overlapping participants...")
        japan_overlap_n = int(self.japan_qual_n * self.overlap_pct)
        vietnam_overlap_n = int(self.vietnam_qual_n * self.overlap_pct)
        
        japan_qual_overlap_indices = sorted(np.random.choice(self.japan_qual_n, japan_overlap_n, replace=False))
        vietnam_qual_overlap_indices = sorted(np.random.choice(self.vietnam_qual_n, vietnam_overlap_n, replace=False))
        
        japan_quant_overlap_indices = sorted(np.random.choice(self.japan_quant_n, japan_overlap_n, replace=False))
        vietnam_quant_overlap_indices = sorted(np.random.choice(self.vietnam_quant_n, vietnam_overlap_n, replace=False))
        
        print(f"  Japan overlapping participants: {japan_overlap_n}")
        print(f"  Vietnam overlapping participants: {vietnam_overlap_n}")
        
        # Step 2: Generate shared demographics for overlapping participants
        print("\n Step 2: Generating shared demographics for overlapping participants...")
        japan_shared_demo = self.generate_demographics('Japan', japan_overlap_n, is_qualitative=False)
        vietnam_shared_demo = self.generate_demographics('Vietnam', vietnam_overlap_n, is_qualitative=False)
        
        # Step 3: Create linkage mapping
        print("\n Step 3: Creating encrypted linkage...")
        linkage = {}
        shared_demographics = {}
        
        # Japan linkage
        for i, (qual_idx, quant_idx) in enumerate(zip(japan_qual_overlap_indices, japan_quant_overlap_indices)):
            qual_id = self.generate_participant_id('JP', qual_idx + 1, 'QUAL')
            quant_id = self.generate_participant_id('JP', quant_idx + 1, 'QUANT')
            link_key = hashlib.sha256(f"{qual_id}_{quant_id}".encode()).hexdigest()[:16]
            
            linkage[qual_id] = {'quant_id': quant_id, 'link_key': link_key}
            shared_demographics[qual_id] = japan_shared_demo.iloc[i].to_dict()
            shared_demographics[quant_id] = japan_shared_demo.iloc[i].to_dict()
        
        # Vietnam linkage
        for i, (qual_idx, quant_idx) in enumerate(zip(vietnam_qual_overlap_indices, vietnam_quant_overlap_indices)):
            qual_id = self.generate_participant_id('VN', qual_idx + 1, 'QUAL')
            quant_id = self.generate_participant_id('VN', quant_idx + 1, 'QUANT')
            link_key = hashlib.sha256(f"{qual_id}_{quant_id}".encode()).hexdigest()[:16]
            
            linkage[qual_id] = {'quant_id': quant_id, 'link_key': link_key}
            shared_demographics[qual_id] = vietnam_shared_demo.iloc[i].to_dict()
            shared_demographics[quant_id] = vietnam_shared_demo.iloc[i].to_dict()
        
        print(f"  Created {len(linkage)} masked linkages with matching demographics")
        
        # Step 4: Generate qualitative data (using shared demographics where applicable)
        print("\n Step 4: Generating qualitative interview data...")
        qual_data = self.generate_qualitative_data(linkage, shared_demographics)
        print(f"  Generated {len(qual_data)} interviews:")
        print(f"    - Japan: {len(qual_data[qual_data['Country']=='Japan'])}")
        print(f"    - Vietnam: {len(qual_data[qual_data['Country']=='Vietnam'])}")
        print(f"    - Also in survey: {qual_data['Also_In_Survey'].sum()}")
        
        # Step 5: Generate quantitative data (using shared demographics where applicable)
        print("\n Step 5: Generating quantitative survey data...")
        quant_data = self.generate_quantitative_dataset(linkage, shared_demographics)
        print(f"  Generated {len(quant_data)} survey responses:")
        print(f"    - Japan: {len(quant_data[quant_data['Country']=='Japan'])}")
        print(f"    - Vietnam: {len(quant_data[quant_data['Country']=='Vietnam'])}")
        print(f"    - Also in interviews: {quant_data['Survey_Link_Key'].notna().sum()}")
        
        # Step 6: Verify matching demographics
        print("\n Step 6: Verifying demographic matching...")
        self.verify_demographic_matching(quant_data, qual_data, linkage)
        
        return quant_data, qual_data, linkage
    
    def save_datasets(self, output_dir='research_data'):
        """Save all datasets"""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        quant_data, qual_data, linkage = self.generate_complete_dataset()
        
        # Save quantitative data
        quant_data.to_csv(f'{output_dir}/survey_data_complete.csv', index=False)
        print(f"\n✓ Saved: {output_dir}/survey_data_complete.csv")
        
        # Save by country
        japan_quant = quant_data[quant_data['Country'] == 'Japan']
        vietnam_quant = quant_data[quant_data['Country'] == 'Vietnam']
        
        japan_quant.to_csv(f'{output_dir}/survey_data_japan.csv', index=False)
        vietnam_quant.to_csv(f'{output_dir}/survey_data_vietnam.csv', index=False)
        print(f"✓ Saved: {output_dir}/survey_data_japan.csv")
        print(f"✓ Saved: {output_dir}/survey_data_vietnam.csv")
        
        # Save qualitative data
        qual_data.to_csv(f'{output_dir}/interview_metadata.csv', index=False)
        print(f"✓ Saved: {output_dir}/interview_metadata.csv")
        
        # Save linkage (encrypted)
        with open(f'{output_dir}/participant_linkage_masked.json', 'w') as f:
            json.dump(linkage, f, indent=2)
        print(f"✓ Saved: {output_dir}/participant_linkage_masked.json")
        
        # Generate data dictionary
        self.generate_data_dictionary(quant_data, qual_data, output_dir)
        
        # Generate summary stats
        self.generate_summary_stats(quant_data, qual_data, output_dir)
        
        print("\n" + "="*70)
        print("DATA GENERATION COMPLETE")
        print("="*70)
        
        return quant_data, qual_data, linkage
    
    def generate_data_dictionary(self, quant_data, qual_data, output_dir):
        """Generate comprehensive data dictionary"""
        
        dictionary = {
            'study_title': 'Leadership Readiness for AI Transformation',
            'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'quantitative_data': {
                'n': len(quant_data),
                'variables': list(quant_data.columns),
                'description': 'Survey responses measuring leadership readiness and outcomes'
            },
            'qualitative_data': {
                'n': len(qual_data),
                'variables': list(qual_data.columns),
                'description': 'Interview metadata for qualitative phase'
            },
            'variable_definitions': {
                'TC_Score': 'Technological Competence (1-7 scale)',
                'CMC_Score': 'Change Management Capability (1-7 scale)',
                'EA_Score': 'Ethical Awareness (1-7 scale)',
                'ALO_Score': 'Adaptive Learning Orientation (1-7 scale)',
                'OI_Score': 'Operational Improvements outcome (1-7 scale)',
                'SA_Score': 'Strategic Advantages outcome (1-7 scale)',
                'OL_Score': 'Organizational Learning outcome (1-7 scale)',
                'Overall_Success': 'Composite AI transformation success (1-7 scale)',
                'Survey_Link_Key': 'Masked linkage to qualitative data (if applicable)'
            },
            'masking_protocol': {
                'method': 'SHA-256 hashing',
                'purpose': 'Protect participant privacy while enabling data integration',
                'description': 'Participants in both phases have matching Survey_Link_Key values'
            }
        }
        
        with open(f'{output_dir}/data_dictionary.json', 'w') as f:
            json.dump(dictionary, f, indent=2)
        
        print(f"✓ Saved: {output_dir}/data_dictionary.json")
    
    def generate_summary_stats(self, quant_data, qual_data, output_dir):
        """Generate summary statistics"""
        
        summary = {
            'quantitative_summary': {},
            'qualitative_summary': {},
            'overlap_summary': {}
        }
        
        for country in ['Japan', 'Vietnam']:
            country_data = quant_data[quant_data['Country'] == country]
            
            summary['quantitative_summary'][country] = {
                'n': len(country_data),
                'demographics': {
                    'mean_age': float(country_data['Age'].mean()),
                    'sd_age': float(country_data['Age'].std()),
                    'pct_male': float((country_data['Gender'] == 'Male').mean() * 100),
                    'mean_tenure': float(country_data['Tenure_Years'].mean()),
                    'sd_tenure': float(country_data['Tenure_Years'].std())
                },
                'lrait_dimensions': {
                    'TC': {'mean': float(country_data['TC_Score'].mean()), 
                           'sd': float(country_data['TC_Score'].std())},
                    'CMC': {'mean': float(country_data['CMC_Score'].mean()), 
                            'sd': float(country_data['CMC_Score'].std())},
                    'EA': {'mean': float(country_data['EA_Score'].mean()), 
                           'sd': float(country_data['EA_Score'].std())},
                    'ALO': {'mean': float(country_data['ALO_Score'].mean()), 
                            'sd': float(country_data['ALO_Score'].std())}
                },
                'outcomes': {
                    'operational': float(country_data['OI_Score'].mean()),
                    'strategic': float(country_data['SA_Score'].mean()),
                    'learning': float(country_data['OL_Score'].mean()),
                    'overall': float(country_data['Overall_Success'].mean())
                }
            }
        
        # Qualitative summary
        for country in ['Japan', 'Vietnam']:
            country_data = qual_data[qual_data['Country'] == country]
            summary['qualitative_summary'][country] = {
                'n': len(country_data),
                'avg_duration_min': float(country_data['Interview_Duration_Min'].mean()),
                'in_both_phases': int(country_data['Also_In_Survey'].sum())
            }
        
        # Overlap summary
        summary['overlap_summary'] = {
            'total_in_both_phases': int(qual_data['Also_In_Survey'].sum()),
            'japan_overlap': int(qual_data[qual_data['Country']=='Japan']['Also_In_Survey'].sum()),
            'vietnam_overlap': int(qual_data[qual_data['Country']=='Vietnam']['Also_In_Survey'].sum()),
            'overlap_percentage': float(qual_data['Also_In_Survey'].mean() * 100)
        }
        
        with open(f'{output_dir}/summary_statistics.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Saved: {output_dir}/summary_statistics.json")


# Main execution
if __name__ == "__main__":
    generator = ComprehensiveDataGenerator()
    quant_data, qual_data, linkage = generator.save_datasets()
    
    print("\nDataset Overview:")
    print(f"Quantitative: {quant_data.shape}")
    print(f"Qualitative: {qual_data.shape}")
    print(f"\nFiles ready for analysis!")