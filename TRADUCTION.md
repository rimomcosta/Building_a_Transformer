# Traduction française et publication

Les chapitres anglais à la racine sont conservés. La version française se trouve dans `fr/`. Le premier chapitre traduit est le chapitre 4.

## Lire et relire

- [Accueil français](fr/index.md)
- [Chapitre 4](fr/chapter-04-positional-encoding.md)
- [Glossaire](fr/glossaire.md)
- [Notes de traduction](fr/notes-traduction.md)

La traduction est assistée et nécessite une relecture. Les annotations sont traduites, mais les mots anglais des exemples numériques sont conservés pour garder les mêmes positions. Les formules et valeurs numériques du chapitre sont conservées ; les imprécisions identifiées sont signalées dans les notes.

## Vérifier localement

Depuis la racine du dépôt, avec Python 3.12 :

```sh
python -m venv .venv-docs
# Windows PowerShell : .\.venv-docs\Scripts\Activate.ps1
# macOS/Linux : source .venv-docs/bin/activate
python -m pip install -r requirements-docs.txt
python -m mkdocs build --strict
python -m mkdocs serve
```

Le site local est ensuite accessible à `http://127.0.0.1:8000/`. Les outils sont épinglés dans `requirements-docs.txt` pour utiliser les versions vérifiées lors de la préparation.

## Activer GitHub Pages

Avant de publier le cours traduit, vérifier ses conditions de republication avec l'auteur : aucune licence n'a été identifiée à la racine du dépôt examiné. La mention de l'auteur est conservée et aucune nouvelle licence n'est attribuée à son contenu.

1. Dans le fork `lviaud/Building_a_Transformer`, ouvrir **Settings → Pages**.
2. Choisir **GitHub Actions** dans **Build and deployment → Source**.
3. Fusionner la pull request de traduction dans `main`.
4. Si les workflows ne sont pas encore activés dans ce fork, les activer dans l'onglet **Actions**.
5. Vérifier le workflow **Documentation française**. Au besoin, le lancer avec **Run workflow**, sur `main`.

Après un déploiement réussi, l'adresse attendue est :

`https://lviaud.github.io/Building_a_Transformer/`

Cette adresse n'est pas une preuve de déploiement : vérifier le résultat du workflow. Une pull request exécute seulement la vérification de construction. La publication est réservée à `main`.

## Ajouter un chapitre

1. Copier le chapitre traduit dans `fr/`, en gardant son nom d'origine.
2. Mentionner l'auteur, l'URL du chapitre source et le SHA de son commit.
3. Employer le glossaire et conserver les identifiants du code ainsi que les formules.
4. Adapter les liens vers les chapitres français existants. Pour les chapitres encore en anglais, utiliser des liens GitHub absolus et indiquer la langue.
5. Ajouter le chapitre dans `nav` de `mkdocs.yml` et mettre à jour `fr/index.md`.
6. Exécuter `python -m mkdocs build --strict`, puis relire le rendu, les calculs et les exemples linguistiques.

## Suivre les mises à jour de l'original

La traduction du chapitre 4 est fondée sur le commit `d8e275a2c41426889e116bfa6d64090805b3bd9a`. Comparer ce fichier à la nouvelle version avant de modifier la traduction ; conserver les corrections humaines et mettre à jour le SHA source après relecture.

La synchronisation du fork avec le dépôt d'origine ne retraduit pas automatiquement les documents.
