import pandas as pd

def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs basic data cleaning
        - drop the ID column
        - convert temporal features to datetime objects
        - address missing values

    Args:
        df (pd.DataFrame): The input DataFrame to be cleaned.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # Drop the ID column if it exists
    df= df.drop(columns=['ID'], errors='ignore')

    # rename the duplicated MineralFertAppMethod columns (pandas reads the second as '.1')
    df = df.rename(columns={'MineralFertAppMethod': 'MineralFertAppMethod_1',
                            'MineralFertAppMethod.1': 'MineralFertAppMethod_2'})

    # converting temporal features to date time objects
    cols=['CropTillageDate','RcNursEstDate','SeedingSowingTransplanting','Harv_date','Threshing_date']

    for col in cols:
      parsed=pd.to_datetime(df[col],format='%Y-%m-%d',errors='coerce')
      # fail loudly if values were present but could not be parsed (e.g. wrong date format)
      bad = parsed.isna() & df[col].notna()
      if bad.any():
         raise ValueError(f"{col}: {bad.sum()} values do not match YYYY-MM-DD, e.g. {df.loc[bad, col].iloc[0]!r}")
      df[col]=parsed

    # addressing missingness fertilizer/dose application
    fert_cols= ['BasalDAP','BasalUrea','1tdUrea','2tdUrea']
    for col in fert_cols:
       df[col]=df[col].fillna(0)

    df['1appDaysUrea'] = df['1appDaysUrea'].fillna(-1)
    df['2appDaysUrea'] = df['2appDaysUrea'].fillna(-1)

    # addressing missingness organic fertilizer application
    df['OrgFertilizers'] = df['OrgFertilizers'].fillna('No organic fertilizer applied')
    df['Ganaura'] = df['Ganaura'].fillna(0)
    df['CropOrgFYM'] = df['CropOrgFYM'].fillna(0)
    df['PCropSolidOrgFertAppMethod'] = df['PCropSolidOrgFertAppMethod'].fillna('No organic fertilizer applied')

    # addressing missingness in irrigation/water management
    df['TransIrriCost'] = df['TransIrriCost'].fillna(0)
    df['TransplantingIrrigationPowerSource'] = df['TransplantingIrrigationPowerSource'].fillna('No irrigation')
    df['TransplantingIrrigationHours'] = df['TransplantingIrrigationHours'].fillna(0)
    df['TransplantingIrrigationSource'] = df['TransplantingIrrigationSource'].fillna('No irrigation')

   
    return df