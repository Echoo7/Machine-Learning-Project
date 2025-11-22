# 🚲 Projet de Machine Learning – Vélo'v Lyon

Ce dépôt contient notre projet de Machine Learning pour le cours **Data Sciences – Furno**, basé sur le jeu de données **Vélo'v Lyon**.

L’objectif de cette organisation est de permettre à chacun de :
- Cloner correctement le dépôt (sans copier-coller manuellement le dossier 🔥)
- Lancer le notebook principal instantanément
- Collaborer facilement en ajoutant des fonctions ou des améliorations au bon endroit

---

## ⚙️ Importer correctement le projet (très important)

> ❗ **Ne copiez-collez surtout pas le dossier du projet depuis GitHub.**  
> Il faut **cloner** le dépôt : cela permet à Git de suivre vos modifications et de synchroniser proprement les mises à jour.

### ✅ Étapes à suivre

1. **Ouvrez Visual Studio Code**

2. Ouvrez la **palette de commandes** (`Ctrl + Shift + P`) et tapez :  
   `Git: Clone`

3. **Collez le lien du dépôt GitHub** (bouton vert “Code” → lien HTTPS, par ex.  
   `https://github.com/Echoo7/Machine-Learning-Project`)

4. **Choisissez un dossier local** où Visual Studio Code téléchargera le projet.

5. Une fois le clonage terminé, cliquez sur **“Ouvrir”** quand VS Code le propose.

6. (Optionnel mais recommandé) Ouvrez un terminal dans VS Code et vérifiez que vous êtes bien dans le dossier du projet :  
   ```bash
   git status

7. Créez l'environnement virtuel grâce au fichier dans le sous-dossier "venv" grâce à la commande :
   conda env create -f venv/DataSciencesFurno.yml
