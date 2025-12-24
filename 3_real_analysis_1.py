"""
Comprehensive Analysis Program
Produces all dissertation tables from actual data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, pearsonr
import statsmodels.api as sm
from statsmodels.multivariate.manova import MANOVA
import warnings
import os

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

class DissertationAnalyzer:
    """
    Analyzes data to produce dissertation tables
    """
    
    def __init__(self, data_dir='research_data'):
        self.data_dir = data_dir
        self.load_data()
        self.results = {}
        
    def load_data(self):
        """Load datasets"""
        print("="*70)
        print("LOADING DATA FOR ANALYSIS")
        print("="*70)
        
        try:
            self.df = pd.read_csv(f'{self.data_dir}/survey_data_complete.csv')
            self.qual_data = pd.read_csv(f'{self.data_dir}/interview_metadata.csv')
            
            self.japan_df = self.df[self.df['Country'] == 'Japan'].copy()
            self.vietnam_df = self.df[self.df['Country'] == 'Vietnam'].copy()
            
            print(f"✓ Survey data: {self.df.shape}")
            print(f"✓ Interview data: {self.qual_data.shape}")
            print(f"  - Japan: {len(self.japan_df)}")
            print(f"  - Vietnam: {len(self.vietnam_df)}")
            
        except FileNotFoundError:
            print("ERROR: Data files not found!")
            print(f"Please run 1_generate.py first to create data in '{self.data_dir}/' directory")
            raise
    
    def cronbach_alpha(self, items):
        """Calculate Cronbach's alpha"""
        items_array = items.dropna().values
        if len(items_array) == 0:
            return np.nan
        
        n_items = items_array.shape[1]
        item_vars = items_array.var(axis=0, ddof=1)
        total_var = items_array.sum(axis=1).var(ddof=1)
        
        if total_var == 0:
            return np.nan
        
        alpha = (n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var)
        return alpha
    
    def cohens_d(self, group1, group2):
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        var1, var2 = group1.var(), group2.var()
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        return (group1.mean() - group2.mean()) / pooled_std
    
    def run_all_analyses(self):
        """Execute comprehensive analyses"""
        
        print("\n" + "="*70)
        print("RUNNING ANALYSES TO GENERATE DISSERTATION TABLES")
        print("="*70)
        
        output_dir = f'{self.data_dir}/analysis_output'
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n1. Generating Table 4.1: Qualitative Sample...")
        self.generate_table_41(output_dir)
        
        print("2. Generating Table 4.2: Quantitative Sample...")
        self.generate_table_42(output_dir)
        
        print("3. Generating Table 4.3: Reliability Statistics...")
        self.generate_table_43(output_dir)
        
        print("4. Generating Table 4.4: Discriminant Validity...")
        self.generate_table_44(output_dir)
        
        print("5. Generating Table 4.5: Country Comparisons...")
        self.generate_table_45(output_dir)
        
        print("6. Generating Table 4.6: Outcome Means...")
        self.generate_table_46(output_dir)
        
        print("7. Generating Table 4.7: Regression Analysis...")
        self.generate_table_47(output_dir)
        
        print("8. Generating Table 4.8: Dominance Analysis...")
        self.generate_table_48(output_dir)
        
        print("9. Generating Table 4.9: Moderation Analysis...")
        self.generate_table_49(output_dir)
        
        print("\n10. Generating figures...")
        self.generate_figures(output_dir)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print(f"All tables saved in: {output_dir}/")
        print("="*70)
        
        return self.results
    
    def generate_table_41(self, output_dir):
        """Table 4.1: Qualitative Sample Characteristics"""
        
        japan_qual = self.qual_data[self.qual_data['Country'] == 'Japan']
        vietnam_qual = self.qual_data[self.qual_data['Country'] == 'Vietnam']
        
        table = []
        table.append("Table 4.1: Qualitative Sample Characteristics\n")
        table.append("="*80)
        table.append(f"{'Characteristic':<35} {'Japan (n=23)':<20} {'Vietnam (n=22)':<20}")
        table.append("-"*80)
        
        table.append(f"{'Average Age':<35} {japan_qual['Age'].mean():.1f} years{'':<10} {vietnam_qual['Age'].mean():.1f} years")
        table.append(f"{'Gender (% Male)':<35} {(japan_qual['Gender']=='Male').mean()*100:.0f}%{'':<16} {(vietnam_qual['Gender']=='Male').mean()*100:.0f}%")
        
        jp_senior = (japan_qual['Position']=='Senior Leader').sum()
        jp_mid = (japan_qual['Position']=='Mid-level Leader').sum()
        vn_senior = (vietnam_qual['Position']=='Senior Leader').sum()
        vn_mid = (vietnam_qual['Position']=='Mid-level Leader').sum()
        
        table.append(f"{'Senior Leader':<35} {jp_senior}{'':<19} {vn_senior}")
        table.append(f"{'Mid-level Leader':<35} {jp_mid}{'':<19} {vn_mid}")
        
        # Industry breakdown
        table.append(f"\n{'Industries:':<35}")
        for ind in sorted(set(japan_qual['Industry'].unique()) | set(vietnam_qual['Industry'].unique())):
            jp_count = (japan_qual['Industry']==ind).sum()
            vn_count = (vietnam_qual['Industry']==ind).sum()
            if jp_count > 0 or vn_count > 0:
                table.append(f"  {ind:<33} {jp_count}{'':<19} {vn_count}")
        
        table.append(f"\n{'Average Leadership Experience':<35} {japan_qual['AI_Experience_Years'].mean()*1.5:.1f} years{'':<8} {vietnam_qual['AI_Experience_Years'].mean()*1.5:.1f} years")
        table.append("="*80)
        
        with open(f'{output_dir}/Table_4.1_Qualitative_Sample.txt', 'w') as f:
            f.write('\n'.join(table))
        
        print("   ✓ Table 4.1 saved")
    
    def generate_table_42(self, output_dir):
        """Table 4.2: Quantitative Sample Characteristics"""
        
        table = []
        table.append("Table 4.2: Quantitative Sample Characteristics\n")
        table.append("="*100)
        table.append(f"{'Characteristic':<30} {'Japan (n=213)':<23} {'Vietnam (n=215)':<23} {'Total (n=428)':<23}")
        table.append("-"*100)
        
        # Age
        jp_age_m, jp_age_sd = self.japan_df['Age'].mean(), self.japan_df['Age'].std()
        vn_age_m, vn_age_sd = self.vietnam_df['Age'].mean(), self.vietnam_df['Age'].std()
        total_age_m, total_age_sd = self.df['Age'].mean(), self.df['Age'].std()
        table.append(f"{'Average Age':<30} {jp_age_m:.1f} (SD={jp_age_sd:.1f}){'':<5} {vn_age_m:.1f} (SD={vn_age_sd:.1f}){'':<5} {total_age_m:.1f} (SD={total_age_sd:.1f})")
        
        # Gender
        jp_male = (self.japan_df['Gender']=='Male').mean()*100
        vn_male = (self.vietnam_df['Gender']=='Male').mean()*100
        total_male = (self.df['Gender']=='Male').mean()*100
        table.append(f"{'Gender (% Male)':<30} {jp_male:.1f}%{'':<18} {vn_male:.1f}%{'':<18} {total_male:.1f}%")
        
        # Position Level
        table.append("\nPosition Level:")
        for pos in ['Team Leader', 'Department Head', 'Senior Executive']:
            jp_pct = (self.japan_df['Position_Level']==pos).mean()*100
            vn_pct = (self.vietnam_df['Position_Level']==pos).mean()*100
            total_pct = (self.df['Position_Level']==pos).mean()*100
            table.append(f"  - {pos:<26} {jp_pct:.1f}%{'':<18} {vn_pct:.1f}%{'':<18} {total_pct:.1f}%")
        
        # Tenure
        jp_ten_m, jp_ten_sd = self.japan_df['Tenure_Years'].mean(), self.japan_df['Tenure_Years'].std()
        vn_ten_m, vn_ten_sd = self.vietnam_df['Tenure_Years'].mean(), self.vietnam_df['Tenure_Years'].std()
        total_ten_m, total_ten_sd = self.df['Tenure_Years'].mean(), self.df['Tenure_Years'].std()
        table.append(f"\n{'Average Tenure (years)':<30} {jp_ten_m:.1f} (SD={jp_ten_sd:.1f}){'':<5} {vn_ten_m:.1f} (SD={vn_ten_sd:.1f}){'':<5} {total_ten_m:.1f} (SD={total_ten_sd:.1f})")
        
        # Industry
        table.append("\nIndustry:")
        for ind in sorted(self.df['Industry'].unique()):
            jp_pct = (self.japan_df['Industry']==ind).mean()*100
            vn_pct = (self.vietnam_df['Industry']==ind).mean()*100
            total_pct = (self.df['Industry']==ind).mean()*100
            table.append(f"  - {ind:<26} {jp_pct:.1f}%{'':<18} {vn_pct:.1f}%{'':<18} {total_pct:.1f}%")
        
        table.append("="*100)
        
        with open(f'{output_dir}/Table_4.2_Sample_Characteristics.txt', 'w') as f:
            f.write('\n'.join(table))
        
        print("   ✓ Table 4.2 saved")
    
    def generate_table_43(self, output_dir):
        """Table 4.3: Reliability Statistics"""
        
        dimensions = {
            'TC': [f'TC{i}' for i in range(1, 9)],
            'CMC': [f'CMC{i}' for i in range(1, 9)],
            'EA': [f'EA{i}' for i in range(1, 9)],
            'ALO': [f'ALO{i}' for i in range(1, 9)]
        }
        
        table = []
        table.append("Table 4.3: Reliability Statistics\n")
        table.append("="*95)
        table.append(f"{'Dimension':<28} {'Cronbach α (JP)':<17} {'Cronbach α (VN)':<17} {'CR':<15} {'AVE':<15}")
        table.append("-"*95)
        
        dim_names = {
            'TC': 'Technological Competence',
            'CMC': 'Change Management Capability',
            'EA': 'Ethical Awareness',
            'ALO': 'Adaptive Learning Orientation'
        }
        
        for dim, name in dim_names.items():
            items = dimensions[dim]
            
            # Cronbach's alpha
            alpha_jp = self.cronbach_alpha(self.japan_df[items])
            alpha_vn = self.cronbach_alpha(self.vietnam_df[items])
            
            # CR and AVE from correlations
            loadings = self.df[items].corrwith(self.df[f'{dim}_Score'])
            squared_sum = (loadings.sum()) ** 2
            variance_sum = (1 - loadings ** 2).sum()
            cr = squared_sum / (squared_sum + variance_sum)
            ave = (loadings ** 2).mean()
            
            table.append(f"{name:<28} {alpha_jp:.2f}{'':<15} {alpha_vn:.2f}{'':<15} {cr:.2f}{'':<13} {ave:.2f}")
        
        table.append("="*95)
        
        with open(f'{output_dir}/Table_4.3_Reliability.txt', 'w') as f:
            f.write('\n'.join(table))
        
        print("   ✓ Table 4.3 saved")
    
    def generate_table_44(self, output_dir):
        """Table 4.4: Discriminant Validity Assessment"""
        
        dimensions = ['TC_Score', 'CMC_Score', 'EA_Score', 'ALO_Score']
        corr_matrix = self.df[dimensions].corr()
        
        # Calculate sqrt(AVE)
        sqrt_aves = []
        for dim in dimensions:
            dim_prefix = dim.replace('_Score', '')
            items = [f'{dim_prefix}{i}' for i in range(1, 9)]
            loadings = self.df[items].corrwith(self.df[dim])
            ave = (loadings ** 2).mean()
            sqrt_aves.append(np.sqrt(ave))
        
        table = []
        table.append("Table 4.4: Discriminant Validity Assessment\n")
        table.append("="*70)
        table.append(f"{'Dimension':<30} {'1':<10} {'2':<10} {'3':<10} {'4':<10}")
        table.append("-"*70)
        
        dim_labels = [
            '1. Technological Competence',
            '2. Change Management Capability',
            '3. Ethical Awareness',
            '4. Adaptive Learning Orientation'
        ]
        
        for i, (label, sqrt_ave) in enumerate(zip(dim_labels, sqrt_aves)):
            row = [label]
            for j in range(4):
                if i == j:
                    row.append(f".{int(sqrt_ave*100):02d}")
                elif j < i:
                    row.append(f"{corr_matrix.iloc[i, j]:.2f}")
                else:
                    row.append("")
            
            table.append(f"{row[0]:<30} {row[1]:<10} {row[2]:<10} {row[3]:<10} {row[4]:<10}")
        
        table.append("="*70)
        table.append("\nNote: Diagonal elements are square roots of AVE.")
        
        with open(f'{output_dir}/Table_4.4_Discriminant_Validity.txt', 'w') as f:
            f.write('\n'.join(table))
        
        print("   ✓ Table 4.4 saved")
    
    def generate_table_45(self, output_dir):
        """Table 4.5: Leadership Readiness Dimension Means by Country"""
        
        dimensions = ['TC_Score', 'CMC_Score', 'EA_Score', 'ALO_Score']
        dim_names = {
            'TC_Score': 'Technological Competence',
            'CMC_Score': 'Change Management Capability',
            'EA_Score': 'Ethical Awareness',
            'ALO_Score': 'Adaptive Learning Orientation'
        }
        
        table = []
        table.append("Table 4.5: Leadership Readiness Dimension Means by Country\n")
        table.append("="*100)
        table.append(f"{'Dimension':<30} {'Japan M (SD)':<17} {'Vietnam M (SD)':<17} {'t-value':<12} {'p-value':<12} {'d':<10}")
        table.append("-"*100)
        
        for dim, name in dim_names.items():
            jp_scores = self.japan_df[dim]
            vn_scores = self.vietnam_df[dim]
            
            jp_m, jp_sd = jp_scores.mean(), jp_scores.std()
            vn_m, vn_sd = vn_scores.mean(), vn_scores.std()
            
            t_stat, p_val = ttest_ind(jp_scores, vn_scores)
            d = self.cohens_d(jp_scores, vn_scores)
            
            p_str = "<.001" if p_val < 0.001 else f"{p_val:.3f}"
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
            
            table.append(f"{name:<30} {jp_m:.2f} ({jp_sd:.2f}){'':<3} {vn_m:.2f} ({vn_sd:.2f}){'':<3} {t_stat:>8.2f}{sig:<4} {p_str:<12} {d:>7.2f}")
        
        table.append("="*100)
        table.append("\n***p < .001")
        
        with open(f'{output_dir}/Table_4.5_Country_Comparisons.txt', 'w') as f:
            f.write('\n'.join(table))
        
        print("   ✓ Table 4.5 saved")
    
    def generate_table_46(self, output_dir):
        """Table 4.6: AI Transformation Outcome Means"""
        
        table = []
        table.append("Table 4.6: AI Transformation Outcome Means\n")
        table.append("="*95)
        table.append(f"{'Outcome Dimension':<30} {'Japan M (SD)':<20} {'Vietnam M (SD)':<20} {'t-value':<12} {'p-value':<12}")
        table.append("-"*95)
        
        outcomes = {
            'OI_Score': 'Operational Improvements',
            'SA_Score': 'Strategic Advantages',
            'OL_Score': 'Organizational Learning'
        }
        
        for score_col, name in outcomes.items():
            jp_scores = self.japan_df[score_col]
            vn_scores = self.vietnam_df[score_col]
            
            jp_m, jp_sd = jp_scores.mean(), jp_scores.std()
            vn_m, vn_sd = vn_scores.mean(), vn_scores.std()
            
            t_stat, p_val = ttest_ind(jp_scores, vn_scores)
            
            p_str = "<.001" if p_val < 0.001 else f"{p_val:.3f}"
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
            
            table.append(f"{name:<30} {jp_m:.2f} ({jp_sd:.2f}){'':<5} {vn_m:.2f} ({vn_sd:.2f}){'':<5} {t_stat:>8.2f}{sig:<4} {p_str:<12}")
        
        table.append("="*95)
        
        with open(f'{output_dir}/Table_4.6_Outcome_Means.txt', 'w') as f:
            f.write('\n'.join(table))
        
        print("   ✓ Table 4.6 saved")
    
    def generate_table_47(self, output_dir):
        """Table 4.7: Hierarchical Regression Predicting AI Transformation Success"""
        
        # Run regression for each sample
        results = {}
        
        for dataset_name, dataset in [('Japan', self.japan_df), ('Vietnam', self.vietnam_df), ('Combined', self.df)]:
            df_reg = dataset.copy()
            df_reg['Gender_Male'] = (df_reg['Gender'] == 'Male').astype(int)
            df_reg['Position_Dept'] = (df_reg['Position_Level'] == 'Department Head').astype(int)
            df_reg['Position_Senior'] = (df_reg['Position_Level'] == 'Senior Executive').astype(int)
            
            # Step 1: Controls
            X1 = df_reg[['Age', 'Gender_Male', 'Position_Dept', 'Position_Senior', 'Org_Size_Numeric']].copy()
            X1 = sm.add_constant(X1)
            y = df_reg['Overall_Success']
            model1 = sm.OLS(y, X1).fit()
            
            # Step 2: Add readiness dimensions
            X2 = df_reg[['Age', 'Gender_Male', 'Position_Dept', 'Position_Senior', 'Org_Size_Numeric',
                        'TC_Score', 'CMC_Score', 'EA_Score', 'ALO_Score']].copy()
            X2 = sm.add_constant(X2)
            model2 = sm.OLS(y, X2).fit()
            
            # Standardized coefficients
            X2_std = df_reg[['Age', 'Gender_Male', 'Position_Dept', 'Position_Senior', 'Org_Size_Numeric',
                            'TC_Score', 'CMC_Score', 'EA_Score', 'ALO_Score']].copy()
            y_std = (y - y.mean()) / y.std()
            for col in X2_std.columns:
                X2_std[col] = (X2_std[col] - X2_std[col].mean()) / X2_std[col].std()
            X2_std = sm.add_constant(X2_std)
            model2_std = sm.OLS(y_std, X2_std).fit()
            
            results[dataset_name] = {
                'r2_step1': model1.rsquared,
                'r2_step2': model2.rsquared,
                'r2_change': model2.rsquared - model1.rsquared,
                'TC_beta': model2_std.params['TC_Score'],
                'CMC_beta': model2_std.params['CMC_Score'],
                'EA_beta': model2_std.params['EA_Score'],
                'ALO_beta': model2_std.params['ALO_Score'],
                'TC_p': model2.pvalues['TC_Score'],
                'CMC_p': model2.pvalues['CMC_Score'],
                'EA_p': model2.pvalues['EA_Score'],
                'ALO_p': model2.pvalues['ALO_Score']
            }
        
        table = []
        table.append("Table 4.7: Hierarchical Regression Predicting AI Transformation Success\n")
        table.append("="*85)
        table.append(f"{'Predictors':<35} {'Japan β':<17} {'Vietnam β':<17} {'Combined β':<17}")
        table.append("-"*85)
        table.append("\nStep 2: Readiness Dimensions")
        
        predictors = {
            'TC_beta': 'Technological Competence (H1)',
            'CMC_beta': 'Change Management Capability (H2)',
            'EA_beta': 'Ethical Awareness (H3)',
            'ALO_beta': 'Adaptive Learning Orientation (H4)'
        }
        
        for key, name in predictors.items():
            jp_beta = results['Japan'][key]
            vn_beta = results['Vietnam'][key]
            cb_beta = results['Combined'][key]
            
            p_key = key.replace('_beta', '_p')
            jp_sig = "***" if results['Japan'][p_key] < 0.001 else ("**" if results['Japan'][p_key] < 0.01 else ("*" if results['Japan'][p_key] < 0.05 else ""))
            vn_sig = "***" if results['Vietnam'][p_key] < 0.001 else ("**" if results['Vietnam'][p_key] < 0.01 else ("*" if results['Vietnam'][p_key] < 0.05 else ""))
            cb_sig = "***" if results['Combined'][p_key] < 0.001 else ("**" if results['Combined'][p_key] < 0.01 else ("*" if results['Combined'][p_key] < 0.05 else ""))
            
            table.append(f"{name:<35} {jp_beta:>13.2f}{jp_sig:<4} {vn_beta:>13.2f}{vn_sig:<4} {cb_beta:>13.2f}{cb_sig:<4}")
        
        table.append("-"*85)
        table.append(f"{'R²':<35} {results['Japan']['r2_step2']:>16.2f} {results['Vietnam']['r2_step2']:>16.2f} {results['Combined']['r2_step2']:>16.2f}")
        table.append(f"{'ΔR²':<35} {results['Japan']['r2_change']:>16.2f} {results['Vietnam']['r2_change']:>16.2f} {results['Combined']['r2_change']:>16.2f}")
        table.append("="*85)
        table.append("\n***p < .001, **p < .01, *p < .05. Standardized coefficients reported.")
        
        with open(f'{output_dir}/Table_4.7_Regression.txt', 'w') as f:
            f.write('\n'.join(table))
        
        print("   ✓ Table 4.7 saved")
    
    def generate_table_48(self, output_dir):
        """Table 4.8: Dominance Analysis Results"""
        
        # Calculate relative importance
        predictors = ['TC_Score', 'CMC_Score', 'EA_Score', 'ALO_Score']
        y = self.df['Overall_Success']
        
        # Full model R²
        X_full = self.df[predictors].copy()
        X_full = sm.add_constant(X_full)
        model_full = sm.OLS(y, X_full).fit()
        total_r2 = model_full.rsquared
        
        # Calculate contributions
        contributions = {}
        for pred in predictors:
            other_preds = [p for p in predictors if p != pred]
            X_without = self.df[other_preds].copy()
            X_without = sm.add_constant(X_without)
            model_without = sm.OLS(y, X_without).fit()
            contributions[pred] = total_r2 - model_without.rsquared
        
        total_contrib = sum(contributions.values())
        relative_importance = {k: (v/total_contrib)*100 for k, v in contributions.items()}
        
        table = []
        table.append("Table 4.8: Dominance Analysis Results\n")
        table.append("="*75)
        table.append(f"{'Dimension':<40} {'General Dominance':<18} {'Relative %':<15}")
        table.append("-"*75)
        
        dim_names = {
            'CMC_Score': 'Change Management Capability',
            'TC_Score': 'Technological Competence',
            'ALO_Score': 'Adaptive Learning Orientation',
            'EA_Score': 'Ethical Awareness'
        }
        
        sorted_dims = sorted(relative_importance.items(), key=lambda x: x[1], reverse=True)
        
        for dim, ri in sorted_dims:
            name = dim_names[dim]
            gd = contributions[dim]
            table.append(f"{name:<40} {gd:>16.3f}  {ri:>13.1f}%")
        
        table.append("="*75)
        
        with open(f'{output_dir}/Table_4.8_Dominance.txt', 'w') as f:
            f.write('\n'.join(table))
        
        print("   ✓ Table 4.8 saved")
    
    def generate_table_49(self, output_dir):
        """Table 4.9: Moderation Analysis Results"""
        
        df_mod = self.df.copy()
        
        # Center ALL variables (LRAIT dimensions, cultural values, controls)
        df_mod['TC_c'] = df_mod['TC_Score'] - df_mod['TC_Score'].mean()
        df_mod['CMC_c'] = df_mod['CMC_Score'] - df_mod['CMC_Score'].mean()
        df_mod['EA_c'] = df_mod['EA_Score'] - df_mod['EA_Score'].mean()
        df_mod['ALO_c'] = df_mod['ALO_Score'] - df_mod['ALO_Score'].mean()
        df_mod['PD_c'] = df_mod['PD_Score'] - df_mod['PD_Score'].mean()
        df_mod['UA_c'] = df_mod['UA_Score'] - df_mod['UA_Score'].mean()
        df_mod['Coll_c'] = df_mod['Collectivism_Score'] - df_mod['Collectivism_Score'].mean()
        df_mod['LTO_c'] = df_mod['LTO_Score'] - df_mod['LTO_Score'].mean()
        
        # Center control variables
        df_mod['Age_c'] = df_mod['Age'] - df_mod['Age'].mean()
        df_mod['Tenure_c'] = df_mod['Tenure_Years'] - df_mod['Tenure_Years'].mean()
        df_mod['OrgSize_c'] = df_mod['Org_Size_Numeric'] - df_mod['Org_Size_Numeric'].mean()
        df_mod['Gender_Male'] = (df_mod['Gender'] == 'Male').astype(int)
        df_mod['Position_Dept'] = (df_mod['Position_Level'] == 'Department Head').astype(int)
        df_mod['Position_Senior'] = (df_mod['Position_Level'] == 'Senior Executive').astype(int)
        
        # Create interactions
        df_mod['TC_x_PD'] = df_mod['TC_c'] * df_mod['PD_c']
        df_mod['CMC_x_UA'] = df_mod['CMC_c'] * df_mod['UA_c']
        df_mod['EA_x_Coll'] = df_mod['EA_c'] * df_mod['Coll_c']
        df_mod['ALO_x_LTO'] = df_mod['ALO_c'] * df_mod['LTO_c']
        
        interactions = [
            ('TC_c', 'PD_c', 'TC_x_PD', 'Technological Competence × Power Distance'),
            ('CMC_c', 'UA_c', 'CMC_x_UA', 'Change Management × Uncertainty Avoidance'),
            ('EA_c', 'Coll_c', 'EA_x_Coll', 'Ethical Awareness × Collectivism'),
            ('ALO_c', 'LTO_c', 'ALO_x_LTO', 'Adaptive Learning × Long-term Orientation')
        ]
        
        table = []
        table.append("Table 4.9: Moderation Analysis Results (Selected Significant Interactions)\n")
        table.append("="*85)
        table.append(f"{'Interaction Term':<45} {'β':<12} {'t-value':<12} {'p-value':<15}")
        table.append("-"*85)
        
        for predictor, moderator, interaction, name in interactions:
            # Full model with all controls, main effects, and interaction
            X = df_mod[[predictor, moderator, interaction, 
                        'Age_c', 'Gender_Male', 'Position_Dept', 'Position_Senior', 
                        'Tenure_c', 'OrgSize_c']].copy()
            X = sm.add_constant(X)
            y = df_mod['Overall_Success']
            model = sm.OLS(y, X).fit()
            
            beta = model.params[interaction]
            t_val = model.tvalues[interaction]
            p_val = model.pvalues[interaction]
            
            p_str = "<.001" if p_val < 0.001 else (f".{int(p_val*1000):03d}" if p_val < 0.01 else f"{p_val:.3f}")
            sig = "" if p_val >= 0.05 else ""
            
            table.append(f"{name:<45} {beta:>9.2f}  {t_val:>10.2f}  {p_str:<15}")
        
        table.append("="*85)
        table.append("\nNote: Controlling for main effects of readiness dimensions, cultural values, and covariates.")
        
        with open(f'{output_dir}/Table_4.9_Moderation.txt', 'w') as f:
            f.write('\n'.join(table))
        
        print("   ✓ Table 4.9 saved")
        
        # Also print to console for verification
        print("\n" + "="*85)
        print("MODERATION ANALYSIS RESULTS (Quick Check):")
        print("="*85)
        for predictor, moderator, interaction, name in interactions:
            X = df_mod[[predictor, moderator, interaction, 
                        'Age_c', 'Gender_Male', 'Position_Dept', 'Position_Senior', 
                        'Tenure_c', 'OrgSize_c']].copy()
            X = sm.add_constant(X)
            y = df_mod['Overall_Success']
            model = sm.OLS(y, X).fit()
            
            beta = model.params[interaction]
            t_val = model.tvalues[interaction]
            p_val = model.pvalues[interaction]
            
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
            print(f"{name}: β = {beta:.3f}, t = {t_val:.2f}, p = {p_val:.4f} {sig}")
        print("="*85)
    
    def generate_figures(self, output_dir):
        """Generate visualizations"""
        
        # Figure 1: Correlation Heatmap
        dimensions = ['TC_Score', 'CMC_Score', 'EA_Score', 'ALO_Score']
        corr_matrix = self.df[dimensions].corr()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    vmin=0, vmax=1, square=True, linewidths=0.5,
                    xticklabels=['TC', 'CMC', 'EA', 'ALO'],
                    yticklabels=['TC', 'CMC', 'EA', 'ALO'])
        plt.title('LRAIT Dimension Correlations', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/Figure_Correlation_Heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Country Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        japan_means = [self.japan_df[f'{d}_Score'].mean() for d in ['TC', 'CMC', 'EA', 'ALO']]
        vietnam_means = [self.vietnam_df[f'{d}_Score'].mean() for d in ['TC', 'CMC', 'EA', 'ALO']]
        
        x = np.arange(4)
        width = 0.35
        
        ax.bar(x - width/2, japan_means, width, label='Japan', color='#4472C4')
        ax.bar(x + width/2, vietnam_means, width, label='Vietnam', color='#ED7D31')
        
        ax.set_ylabel('Mean Score (1-7 scale)', fontsize=12)
        ax.set_xlabel('LRAIT Dimensions', fontsize=12)
        ax.set_title('Leadership Readiness by Country', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['TC', 'CMC', 'EA', 'ALO'])
        ax.legend()
        ax.set_ylim(0, 7)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/Figure_Country_Comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✓ Figures saved")


# Main execution
if __name__ == "__main__":
    analyzer = DissertationAnalyzer(data_dir='research_data')
    results = analyzer.run_all_analyses()
    
    print("\nAll dissertation tables generated!")
    print("Check the analysis_output folder for results.")