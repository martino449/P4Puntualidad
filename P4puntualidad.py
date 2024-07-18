#-------------------------------
# Nome:        P4puntualidad
# Scopo:       Addestrare un classificatore ad albero decisionale e fare previsioni
#
# Autore:      Martin449
#
# Creato il:   18/07/2024
# dipendenze:
# python = ">=3.10.0,<3.12"
# pandas = "^2.2.2"
# scikit-learn = "^1.5.1"
#
# Licenza:     GNU GENERAL PUBLIC LICENSE v.3, 29 Giugno 2007
#-------------------------------


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Carica il dataset
try:
    partite = pd.read_csv("partite.csv")
except FileNotFoundError:
    raise FileNotFoundError("Il file 'partite.csv' non è stato trovato.")
except pd.errors.EmptyDataError:
    raise ValueError("Il file 'partite.csv' è vuoto.")
except pd.errors.ParserError:
    raise ValueError("Errore nel parsing del file 'partite.csv'. Controlla il formato del file.")

# Separa le caratteristiche (X) e la variabile target (y)
X = partite.drop(columns=["risultato"])
y = partite["risultato"]

# Controlla la presenza di valori mancanti nelle caratteristiche
if X.isnull().any().any():
    raise ValueError("Le caratteristiche contengono valori mancanti. Per favore gestisci i valori mancanti prima dell'addestramento.")

# Codifica le variabili categoriche
label_encoders = {}
for column in X.columns:
    if X[column].dtype == 'object':
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le

# Inizializza il classificatore ad albero decisionale
modello = DecisionTreeClassifier()

# Addestra il modello con le caratteristiche e la variabile target
modello.fit(X, y)

# Richiedi input all'utente
partita = input("Inserisci le squadre che vuoi analizzare (prima quella in casa, poi quella in trasferta e separale con una virgola ad es. inter, milan): \n")

# Prepara l'input dell'utente
squadre = partita.split(', ')
if len(squadre) != 2:
    raise ValueError("L'input deve contenere esattamente due squadre separate da una virgola.")

# Codifica l'input dell'utente
input_df = pd.DataFrame([squadre], columns=['squadra_casa', 'squadra_trasferta'])

# Verifica se le colonne di input corrispondono a quelle del dataset
for column in input_df.columns:
    if column in label_encoders:
        if not input_df[column].isin(label_encoders[column].classes_).all():
            raise ValueError(f"Le squadre inserite non sono presenti nel dataset. Verifica i nomi delle squadre.")
        input_df[column] = label_encoders[column].transform(input_df[column])
    else:
        raise ValueError(f"La colonna '{column}' non è presente nel dataset di addestramento.")

# Fai previsioni basate sull'input dell'utente
try:
    previsione = modello.predict(input_df)
except ValueError as e:
    raise ValueError(f"Errore nella previsione: {e}")

# Mostra le previsioni
print("Previsioni:")
print(previsione[0])
