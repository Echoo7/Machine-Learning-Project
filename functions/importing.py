import pandas as pd
import glob


def read_whole_pandas_df(path, alternative_path, sep = ",", enc = None, columns = None, parse_dates=None, format=None, alternative_dataset = False):
    if alternative_dataset:
        all_files = [alternative_path]
    else:
        all_files = glob.glob(path)
    li = []
    for filename in all_files:
        print("Reading file {}".format(filename))
        df = pd.read_csv(filename, encoding = enc, sep=sep, names = columns, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    return frame


def read_meteo_data(path, sep=";", enc="utf-8", parse_dates=True):
    """
    Lit le fichier météo et renvoie un DataFrame Pandas.
    
    Paramètres
    ----------
    path : str
        Chemin du fichier météo.
    sep : str
        Séparateur utilisé dans le CSV.
    enc : str
        Encodage du fichier.
    parse_dates : bool
        Si vrai, convertit automatiquement les colonnes de dates.
    """
    
    print(f"Lecture du fichier météo : {path}")
    
    df = pd.read_csv(path, sep=sep, encoding=enc)
    
    # Si parse_dates = True → on détecte automatiquement les colonnes de dates
    if parse_dates:
        # Essaie de convertir toute colonne contenant 'date' ou 'Date'
        date_cols = [col for col in df.columns if "date" in col.lower()]
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    
    return df