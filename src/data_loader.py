


def load_dataset(csv_path: Path, multi_output: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame | pd.Series]:
    
    data = pd.read_csv(csv_path)
    
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset. Available columns: {data.columns.tolist()}")
    
    features = data[FEATURE_COLUMNS].copy()
    
    if multi_output:
        
        def categorize_bmi(bmi):
            if bmi < 18.5:
                return 0
            elif bmi < 25:
                return 1
            elif bmi < 30:
                return 2
            else:
                return 3
        obesity_target = features['BMI'].apply(categorize_bmi)
        
        # Get Diabetes target
        diabetes_target = data[TARGET_COLUMN].copy()
 
        combined = pd.concat([features.reset_index(drop=True), diabetes_target.reset_index(drop=True), obesity_target.reset_index(drop=True)], axis=1)
        combined.columns = list(FEATURE_COLUMNS) + ['Diabetes_012', 'Obesity']
        
        diabetes_counts = combined['Diabetes_012'].value_counts()
        max_count = diabetes_counts[0]  # Class 0 has most samples (~185K)
        
        print(f"Before balancing - Diabetes: {dict(diabetes_counts)}")
        
        dfs = []
        for cls in sorted(combined['Diabetes_012'].unique()):
            cls_data = combined[combined['Diabetes_012'] == cls]
            if len(cls_data) < max_count:
                cls_data = resample(cls_data, replace=True, n_samples=max_count, random_state=42)
            dfs.append(cls_data)
        
        combined = pd.concat(dfs, ignore_index=True)
        
        print(f"After balancing - Diabetes: {dict(combined['Diabetes_012'].value_counts())}")
        
        features = combined[FEATURE_COLUMNS].copy()
        targets = combined[['Diabetes_012', 'Obesity']].copy()
        
    else:
        targets = data[TARGET_COLUMN].copy()
    
    return features, targets

