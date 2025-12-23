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
```

**Key changes:**

1. **Education generated before tenure** (affects career start age)

2. **Career start age calculated:**
   - Bachelor's: 22 years old
   - Master's: 24 years old
   - PhD: 28 years old

3. **Maximum possible tenure** = Age - Career Start Age
   - 30-year-old with Bachelor's: max 8 years tenure (30-22)
   - 45-year-old with Master's: max 21 years tenure (45-24)
   - 55-year-old with PhD: max 27 years tenure (55-28)

4. **Tenure constrained** to never exceed maximum possible

**Example outputs:**
```
✓ Age: 30, Education: Bachelor, Tenure: 5.2 years (started at ~25)
✓ Age: 45, Education: Master, Tenure: 12.8 years (started at ~32)
✓ Age: 52, Education: PhD, Tenure: 18.3 years (started at ~34)

✗ Age: 30, Education: Bachelor, Tenure: 15.5 years (IMPOSSIBLE - would need to start at 14.5)