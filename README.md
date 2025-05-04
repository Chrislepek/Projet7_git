# Implementation_modele_scoring_Pequignot_Christophe

Brève description du projet.

mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifier la demande en crédit accordé ou refusé

## Table des Matières

- [Implementation\_modele\_scoring\_Pequignot\_Christophe](#implementation_modele_scoring_pequignot_christophe)
  - [Table des Matières](#table-des-matières)
  - [Installation](#installation)
  - [Utilisation](#utilisation)
  - [Fonctionnalités](#fonctionnalités)
  - [Contact](#contact)

## Installation

1. Clonez le dépôt : `git clone https://github.com/Chrislepek/Projet7_git.git`
2. Accédez au répertoire du projet : `cd Projet7_git`
3. Installez les dépendances : `pip install -r requirements.txt`
4. Lancer l'api via fastapi : `fastapi.exe dev main.py`
   

## Utilisation

Pour obtenir le scoring vous devez saisir un numéro de client via l'instruction 
/predict/{client_id} : vous pouvez tester avec les id suivants : 101011, 100286, 198356, 109467
l'API prédit un score. en fonction d'un seuil, le credit est soit accepté soit refusé

## Fonctionnalités
en production
- Fonctionnalité 1 : /predict/{client_id} --> outpout client_id, score sous forme de prédicition et une classe (accepté/refusé)
- ...




## Contact

pour toute info complémentaire contatez chris.lepek@gmail.com.
