import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import TargetEncoder
from sklearn.model_selection import KFold


#set the list of high cardinality and low cardinality features, >10 unique values
HIGH_CARDINALITY_COLS = ["LandPreparationMethod", "CropbasalFerts",
                         "OrgFertilizers", "FirstTopDressFert"]
LOW_CARDINALITY_COLS  = ['District','Block','CropEstMethod','TransplantingIrrigationSource','TransplantingIrrigationPowerSource',
                         'PCropSolidOrgFertAppMethod','MineralFertAppMethod_1','MineralFertAppMethod_2','Harv_method','Threshing_method','Stubble_use']


def build_preprocessor():
    """
    Build a preprocessor for the dataset using ColumnTransformer.
    """
    return ColumnTransformer([
        ("target_enc", TargetEncoder(target_type="continuous",cv=KFold(n_splits=5,shuffle=True, random_state=42)), HIGH_CARDINALITY_COLS),
        ("onehot", OneHotEncoder(handle_unknown="ignore"), LOW_CARDINALITY_COLS),
    ], remainder="passthrough")

def eng_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the input DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame to be processed.

    Returns:
        pd.DataFrame: The DataFrame with engineered features.
    """
    df=df.copy()
    # agronomic features
    df['GrowingSeasonDays'] = (df['Harv_date'] - df['SeedingSowingTransplanting']).dt.days
    # broadcast crops have no nursery, -1 marks not applicable
    df['NurseryToTransplantDays'] = (df['SeedingSowingTransplanting'] - df['RcNursEstDate']).dt.days.fillna(-1)
    df['TillageToTransplantDays'] = (df['SeedingSowingTransplanting'] - df['CropTillageDate']).dt.days
    df['HarvestToThreshingDays'] = (df['Threshing_date'] - df['Harv_date']).dt.days
    df['PlantingMonth'] = df['SeedingSowingTransplanting'].dt.month
    df['HarvestMonth'] = df['Harv_date'].dt.month

    # now drop raw dates and unused features
    df = df.drop(columns=['SeedingSowingTransplanting', 'RcNursEstDate', 'CropTillageDate', 'Harv_date', 'Threshing_date','TransDetFactor','NursDetFactor','Residue_perc'])

    return df