# Chapitre 4 : L'encodage positionnel (enseigner l'ordre des mots)

> Traduction française non officielle du cours de Rimom Costa.
> [Texte original](https://github.com/rimomcosta/Building_a_Transformer/blob/d8e275a2c41426889e116bfa6d64090805b3bd9a/chapter-04-positional-encoding.md).
> Version source : `d8e275a2c41426889e116bfa6d64090805b3bd9a` ; traduction du 2 novembre 2026.
>
> Les phrases anglaises servant aux calculs sont conservées afin de préserver le nombre et la position des mots. Les blocs numériques reprennent les valeurs de la source. Cette traduction conserve aussi les simplifications pédagogiques de l'auteur ; les réserves techniques sont indiquées dans les [notes de traduction](notes-traduction.md).

## Partie 1 : Comprendre le problème fondamental

### La découverte du parallélisme

Commençons par une différence fondamentale entre le fonctionnement des Transformers et notre manière de lire.

**La lecture humaine (séquentielle) :**
```
Étape 1 : lire "I" → le comprendre
Étape 2 : lire "love" → le comprendre  
Étape 3 : lire "pizza" → le comprendre
Durée : 3 étapes successives
```

Nous traitons les mots un à un, dans l'ordre. C'est le **traitement séquentiel**.

**La lecture des Transformers (parallèle) :**
```
Étape 1 : voir TOUS les mots en même temps → les comprendre ensemble
Durée : 1 étape simultanée
```

Le Transformer voit les trois mots au même instant ! C'est le **traitement parallèle**, qui explique la rapidité des Transformers ! 🚀

Mais cette rapidité crée un problème majeur…

### Le problème : l'absence de perception de l'ordre

Au chapitre 3, nous avons converti notre phrase en embeddings, ou représentations vectorielles. **Rappel important : ces nombres sont encore ALÉATOIRES !** Nous les avons initialisés au hasard et ils n'ont pas encore de signification. L'entraînement leur permettra de représenter le sens des mots ; pour l'instant, ce sont simplement des points de départ aléatoires.

```
"I":     [0.01, -0.20, 0.30, 0.15, -0.05, 0.11]  ← Nombres aléatoires (pour l'instant)
"love":  [-0.40, 0.60, 0.00, 0.25, 0.90, -0.30]  ← Nombres aléatoires (pour l'instant)
"pizza": [0.90, -0.10, -0.20, -0.80, 0.50, 0.75]  ← Nombres aléatoires (pour l'instant)
```

**Voici le problème.** Ce ne sont que trois vecteurs, trois listes de nombres. Ils pourraient se présenter dans N'IMPORTE QUEL ordre sans que le Transformer fasse la différence !

**L'analogie du sac de billes :**

Imaginez un sac contenant trois billes colorées :

- 🔴 Une bille rouge
- 🔵 Une bille bleue
- 🟢 Une bille verte

Vous secouez le sac et versez les billes sur une table. Elles peuvent sortir :

- Verte, rouge, bleue
- Ou bleue, verte, rouge
- Ou rouge, bleue, verte

**Le sac ne mémorise pas leur ordre initial !** Une fois sur la table, ce sont simplement trois billes sans ordre intrinsèque.

Il en va de même pour nos embeddings ! Le Transformer voit :
```
Vecteur n°1: [0.01, -0.20, 0.30, 0.15, -0.05, 0.11]
Vecteur n°2: [-0.40, 0.60, 0.00, 0.25, 0.90, -0.30]
Vecteur n°3: [0.90, -0.10, -0.20, -0.80, 0.50, 0.75]
```

**Lequel vient en premier ? Lequel vient en dernier ?** Aucun de ces nombres ne dit « je suis le premier mot » ou « je suis le troisième mot ». Ce ne sont que trois ensembles de nombres sans repère !

### Pourquoi l'ordre compte : le sens en jeu

Voyons pourquoi cela pose un problème avec des exemples concrets.

**Exemple 1 : Qui fait quoi à qui ?**

- « The dog bit the man » 🐕→👨 : le chien a mordu l'homme.
- « The man bit the dog » 👨→🐕 : l'homme a mordu le chien !

Les mêmes mots expriment des sens complètement opposés ! L'ordre change TOUT.

**Exemple 2 : Phrase compréhensible ou dénuée de sens**

- « I love pizza » ✓ : j'aime la pizza, une phrase normale.
- « Pizza love I » ✗ : une tournure digne de Yoda.
- « Love I pizza » ✗ : un assemblage de mots incohérent.

**Exemple 3 : Avertissements et ordres**

- « Don't eat that! » 🚫🍴 : ne mange pas ça, un avertissement urgent !
- « Eat that, don't! » ✗ : une formulation étrange et confuse.
- « That don't eat! » ✗ : une formulation sans sens clair.

**Exemple 4 : Questions et affirmations**

- « You are going? » : une question, « tu y vas ? »
- « Are you going? » : également une question, avec une construction différente.
- « Going you are? » ✗ : une tournure incohérente dans cet exemple.

Sans information de position, le Transformer considérerait ces phrases comme identiques puisqu'elles contiennent les mêmes mots ! 😱

### Ce qu'il nous faut : les exigences

Avant de chercher une solution, définissons les propriétés d'un encodage positionnel IDÉAL.

**Exigence 1 : Chaque position doit être unique**

- La position 1 doit se distinguer de la position 2.
- La position 2 doit se distinguer de la position 3.
- Deux positions ne doivent pas avoir la même « empreinte ».

**Exigence 2 : Rester borné (éviter les valeurs trop grandes !)**

- Nos embeddings contiennent de petits nombres, entre -1 et 1.
- L'information de position ne doit pas écraser le sens du mot.
- Les valeurs doivent rester raisonnables, même à la position 1000.

**Exigence 3 : Fonctionner quelle que soit la longueur de la phrase**

- Phrases courtes : 5 mots.
- Phrases moyennes : 50 mots.
- Documents longs : 1000 mots ou plus.
- Même pour des longueurs jamais rencontrées auparavant !

**Exigence 4 : Des distances cohérentes**

- « Côte à côte » doit toujours avoir le même sens.
- Qu'il s'agisse des mots 5 et 6 ou des mots 100 et 101.
- Le modèle doit pouvoir apprendre des régularités cohérentes.

**Exigence 5 : Des informations à plusieurs échelles**

- Distinguer des mots proches : positions 1 et 2.
- Distinguer aussi des mots éloignés : positions 1 et 100.
- Fournir à la fois des informations locales et globales.

Peut-on satisfaire TOUTES ces exigences ? Essayons !

---

## Partie 2 : Les tentatives infructueuses (apprendre de ses erreurs)

Avant d'aborder la solution, essayons les approches « évidentes » pour comprendre POURQUOI elles échouent. Cela nous permettra d'apprécier l'élégance de la solution finale !

### Tentative 1 : Utiliser simplement les numéros de position (1, 2, 3…)

**« Pourquoi ne pas ajouter le numéro de position à chaque embedding ? »**

Bonne intuition ! Il suffit d'étiqueter chaque mot par sa position. Essayons avec une phrase plus longue pour voir ce qui se produit : « I really love eating delicious homemade pizza with my best friends » (« J'aime vraiment manger de délicieuses pizzas maison avec mes meilleurs amis »).

**Mise en place de l'addition :**

```
Position 0: "I"         → [0.01, -0.20, 0.30, 0.15, -0.05, 0.11] + [0, 0, 0, 0, 0, 0]
Position 1: "really"    → [0.34, 0.12, -0.45, 0.22, 0.67, -0.18] + [1, 1, 1, 1, 1, 1]
Position 2: "love"      → [-0.40, 0.60, 0.00, 0.25, 0.90, -0.30] + [2, 2, 2, 2, 2, 2]
Position 3: "eating"    → [0.55, -0.33, 0.21, -0.12, 0.44, 0.08] + [3, 3, 3, 3, 3, 3]
Position 4: "delicious" → [0.78, 0.05, -0.66, 0.31, -0.22, 0.49] + [4, 4, 4, 4, 4, 4]
Position 5: "homemade"  → [-0.11, 0.88, 0.13, -0.55, 0.37, -0.71] + [5, 5, 5, 5, 5, 5]
Position 6: "pizza"     → [0.90, -0.10, -0.20, -0.80, 0.50, 0.75] + [6, 6, 6, 6, 6, 6]
Position 7: "with"      → [0.02, -0.47, 0.61, 0.18, -0.29, 0.33] + [7, 7, 7, 7, 7, 7]
Position 8: "my"        → [-0.58, 0.24, 0.09, -0.41, 0.73, -0.15] + [8, 8, 8, 8, 8, 8]
Position 9: "best"      → [0.41, -0.69, 0.52, 0.07, -0.35, 0.88] + [9, 9, 9, 9, 9, 9]
Position 10: "friends"  → [0.90, -0.10, -0.20, -0.80, 0.50, 0.75] + [10, 10, 10, 10, 10, 10]
```

**Après l'addition :**

```
Position 0: "I"         → [0.01, -0.20, 0.30, 0.15, -0.05, 0.11]
Position 1: "really"    → [1.34, 1.12, 0.55, 1.22, 1.67, 0.82]
Position 2: "love"      → [1.60, 2.60, 2.00, 2.25, 2.90, 1.70]
Position 3: "eating"    → [3.55, 2.67, 3.21, 2.88, 3.44, 3.08]
Position 4: "delicious" → [4.78, 4.05, 3.34, 4.31, 3.78, 4.49]
Position 5: "homemade"  → [4.89, 5.88, 5.13, 4.45, 5.37, 4.29]
Position 6: "pizza"     → [6.90, 5.90, 5.80, 5.20, 6.50, 6.75]
Position 7: "with"      → [7.02, 6.53, 7.61, 7.18, 6.71, 7.33]
Position 8: "my"        → [7.42, 8.24, 8.09, 7.59, 8.73, 7.85]
Position 9: "best"      → [9.41, 8.31, 9.52, 9.07, 8.65, 9.88]
Position 10: "friends"  → [10.90, 9.90, 9.80, 9.20, 10.50, 10.75]
```

**Voyez-vous le problème ?** Les embeddings initiaux, de petits nombres comme 0.90 et -0.10, sont complètement écrasés par les numéros de position, comme 10 ou 9.90 ! Comprenons pourquoi cela pose problème.

**Problème 1 : Ce que chaque dimension doit apprendre**

Avant de parler des grands nombres, voyons ce qui est CENSÉ se produire après l'entraînement.

**Rappel :** les valeurs des embeddings sont pour l'instant ALÉATOIRES. Après l'entraînement, chaque dimension doit représenter à la fois le sens du mot ET sa position !

Par exemple, après l'entraînement :

- La **dimension 0** pourrait représenter **« Ce mot parle-t-il de nourriture ? » + « Marqueur de position 0 »**.
  - Pour « pizza » à la position 6 : par exemple, 0.85 (nourriture = oui) + 6 (position) = 6.85.
  - Pour « with » à la position 7 : par exemple, 0.02 (nourriture = non) + 7 (position) = 7.02.

- La **dimension 1** pourrait représenter **« Ce mot est-il un verbe ? » + « Marqueur de position 1 »**.
  - Pour « love » à la position 2 : par exemple, 0.90 (verbe = oui) + 2 (position) = 2.90.
  - Pour « pizza » à la position 6 : par exemple, -0.10 (verbe = non) + 6 (position) = 5.90.

- La **dimension 2** pourrait représenter **« Quel degré d'émotion ? » + « Marqueur de position 2 »**.
  - Et ainsi de suite…

**L'idée essentielle :** chaque dimension finira par encoder à la fois une information sémantique, ce que représente le mot, ET une information de position. Le modèle doit apprendre des régularités dans les DEUX composantes de chaque nombre !

**Voici maintenant le problème avec les valeurs 1, 2, 3…**

**Problème 1a : Le signal de position « crie » trop fort — des échelles incompatibles**

Regardons la position 10. Pour le mot « friends », nous avons :
```
Embedding initial : [0.90, -0.10, -0.20, -0.80, 0.50, 0.75]
Ajout position 10 :    [10,    10,    10,    10,    10,   10  ]
                     ─────  ─────  ─────  ─────  ──── ──────
Résultat :             [10.90, 9.90,  9.80,  9.20, 10.50, 10.75]
```

**Réfléchissons à ce que la dimension 0 essaie d'apprendre :**

- Elle VEUT apprendre : « Ce mot parle-t-il de nourriture ? », un petit signal comme 0.90.
- Elle DOIT aussi encoder : « C'est la position 10 », un signal énorme de 10.
- Valeur combinée : 10.90.

**Le modèle voit 10.90 et doit déterminer :**

- « Le sens vaut-il 0.90 et la position 10 ? »
- « Ou le sens vaut-il 10.50 et la position 0.40 ? »
- « Ou encore 5.45 pour chacun ? »

Le numéro de position (10) **domine** le signal ! C'est comme essayer d'entendre un murmure (0.90) près d'un réacteur d'avion (10). Dans cette analogie, le modèle consacrerait 99 % de son attention à « c'est la position 10 » et remarquerait à peine le sens du mot enfoui dans le nombre.

**Problème 1b : Même les dérivées ne nous sauvent pas (le problème technique)**

**Vous vous demandez peut-être :** « Ne peut-on pas utiliser les dérivées pour séparer les deux composantes pendant l'entraînement ? Pendant la rétropropagation, nous calculons les gradients : on devrait pouvoir distinguer position et sens, non ? »

**Bonne question !** C'est exactement ce qu'un débutant attentif pourrait demander. Voici pourquoi cela ne suffit TOUJOURS PAS dans le raisonnement présenté ici.

**Oui, mathématiquement, on PEUT les séparer à l'aide des dérivées :**

```
Combined = WordMeaning + PositionNumber
∂Combined/∂WordMeaning = 1  (le gradient traverse l'addition)
```

Techniquement, le gradient indique donc « comment mettre à jour la composante du sens ». Mais trois problèmes majeurs subsistent.

**Problème A : Un taux d'apprentissage inadapté**

Pendant l'entraînement, nous mettons les paramètres à jour ainsi :
```
new_value = old_value - learning_rate × gradient
```

Un taux d'apprentissage typique : 0.0001, une très petite valeur !

Pour la dimension 0 à la position 1000 :
```
Valeur actuelle : 1000.2 (= sens du mot 0.2 + position 1000)
Le gradient indique : "Augmenter le sens du mot de 0.5"
Mise à jour : 1000.2 - 0.0001 × (-0.5) = 1000.20005
```

La mise à jour est MINUSCULE comparée à 1000.2 : l'auteur décrit un apprentissage extrêmement lent. Le sens devrait changer de 0.5, mais, enfoui dans 1000.2, il évolue à peine avec un taux d'apprentissage de 0.0001 !

**Problème B : La perte de précision numérique**

Les ordinateurs stockent les nombres avec une précision limitée : les flottants sur 32 bits offrent généralement environ 7 chiffres significatifs.

```
Position 1000, sens du mot 0.234567:
Valeur exacte : 1000.234567
Valeur stockée :   1000.2346 (perte de précision sur les derniers chiffres !)
```

Lorsque la position vaut 1000, le texte source affirme que les petites variations du sens, de l'ordre de 0.001, se perdent dans les arrondis. L'ordinateur ne pourrait alors plus représenter les changements précis nécessaires à l'apprentissage du sens.

**Problème C : Le gaspillage de la capacité du réseau**

Le Transformer doit apprendre : « Les valeurs de la dimension 0 proches de 1000 indiquent une position autour de 1000, et il faut examiner la petite partie décimale pour retrouver le sens du mot. »

Cela reviendrait à apprendre une règle DISTINCTE pour chaque plage de positions :

- Positions 0–10 : « Examiner les nombres 0–11 ; le sens est dans la partie décimale. »
- Positions 100–110 : « Examiner les nombres 100–111 ; le sens est dans la partie décimale. »
- Positions 1000–1010 : « Examiner les nombres 1000–1011 ; le sens est dans la partie décimale. »

Le modèle devrait apprendre 1000 « échelles » différentes pour traiter ces plages ! Cela gaspillerait une capacité considérable qui devrait servir à apprendre les structures du langage, plutôt qu'à lire des décimales à différentes échelles.

**L'idée :** même si les dérivées permettent mathématiquement de séparer les composantes, l'apprentissage pratique devient inefficace selon cette explication. C'est comme peser une plume posée sur un éléphant : le poids total change, mais détecter cette différence et en tirer un apprentissage devient difficile !

**Problème 1c : Visualisation du problème**

Voyons ce qui se produit à la position 1000 d'un document long :

```
Mot "amazing" à la position 1000 :
L'embedding veut apprendre : [0.75, -0.30, 0.45, 0.60, -0.20, 0.35]
                                 ↑      ↑      ↑      ↑      ↑      ↑
                             "positif" "nom" "intensité" etc...

Après ajout de la position 1000 :   [1000.75, 999.70, 1000.45, 1000.60, 999.80, 1000.35]
                                 ↑         ↑         ↑         ↑        ↑        ↑
                            La position domine ! Le sens du mot est enfoui !
```

Le Transformer voit des valeurs proches de 1000 dans toutes les dimensions et apprend : « C'est la position 1000 environ ! »

Il a toutefois du mal à apprendre les différences subtiles entre les décimales, comme 0.75, 0.70 et 0.45, qui encodent le sens du mot. Le rapport signal/bruit est mauvais !

**Problème 2 : Aucune borne supérieure**

Quelle est la longueur maximale d'une phrase ? Nous ne le savons pas ! Certaines ont 10 mots, d'autres 1000. Le modèle rencontrerait des échelles très différentes pendant l'entraînement, ce qui compliquerait l'apprentissage de régularités stables.

**Problème 3 : Pas d'information de position relative facilement exploitable**

Le modèle ne peut pas apprendre facilement « ce mot vient trois positions après cet autre mot ». Il devrait apprendre : « Si l'on ajoute +5 à ce vecteur et +2 à celui-là, ils sont séparés de trois positions. » Une relation difficile à apprendre à partir des données !

---

### Tentative 2 : Normaliser les positions (entre 0.0 et 1.0)

**« D'accord, je vois ! Ramenons les positions entre 0 et 1 ! »**

**L'amélioration :** au lieu des positions brutes (1, 2, 3…), divisons par la longueur totale de la phrase pour rester entre 0 et 1.

**Pour la phrase annoncée comme ayant 10 mots dans la source :** « I really love eating delicious homemade pizza with my best friends ».

```
Position 0: 0/10 = 0.0   → "I"
Position 1: 1/10 = 0.1   → "really"
Position 2: 2/10 = 0.2   → "love"
Position 3: 3/10 = 0.3   → "eating"
Position 4: 4/10 = 0.4   → "delicious"
Position 5: 5/10 = 0.5   → "homemade"
Position 6: 6/10 = 0.6   → "pizza"
Position 7: 7/10 = 0.7   → "with"
Position 8: 8/10 = 0.8   → "my"
Position 9: 9/10 = 0.9   → "best"
Position 10: 10/10 = 1.0 → "friends"
```

Tous les nombres sont petits, entre 0.0 et 1.0 : ils n'écraseront donc pas nos embeddings. Essayons !

**Addition aux embeddings :**

```
"I" (position 0.0):
Embedding :  [0.01, -0.20, 0.30, 0.15, -0.05, 0.11]
Position 0.0:    [0.0,   0.0,  0.0,  0.0,   0.0,  0.0]
                  ────   ────  ────  ────   ────  ────
Combinaison :        [0.01, -0.20, 0.30, 0.15, -0.05, 0.11]  ✓ Inchangé (position 0)

"homemade" (position 0.5):
Embedding :  [-0.11, 0.88, 0.13, -0.55,  0.37, -0.71]
Position 0.5:    [ 0.5,  0.5,  0.5,   0.5,   0.5,   0.5]
                  ─────  ────  ────  ─────  ─────  ─────
Combinaison :        [ 0.39, 1.38, 0.63, -0.05,  0.87, -0.21]  ✓ Équilibré !

"friends" (position 1.0):
Embedding :  [0.90, -0.10, -0.20, -0.80,  0.50,  0.75]
Position 1.0:    [1.0,   1.0,   1.0,   1.0,   1.0,   1.0]
                  ────   ────   ────   ────   ────   ────
Combinaison :        [1.90,  0.90,  0.80,  0.20,  1.50,  1.75]  ✓ Raisonnable !
```

**Cela semble bien mieux !** Les positions, entre 0.0 et 1.0, sont du même ordre de grandeur que les embeddings, entre -1 et +1. Aucune composante n'écrase l'autre. Les dimensions pourront encoder à la fois le sens du mot et sa position de manière équilibrée.

**Mais un nouveau problème apparaît !** Comparons deux phrases.

**Phrase courte (5 mots) : « I love eating delicious pizza »**
```
Position 0: 0/5 = 0.0
Position 1: 1/5 = 0.2
Position 2: 2/5 = 0.4
Position 3: 3/5 = 0.6
Position 4: 4/5 = 0.8

Distance entre mots adjacents : 0.2 - 0.0 = 0.2
```

**Phrase plus longue (10 mots) : « I really love eating delicious homemade pizza with my friends »**
```
Position 0: 0/10 = 0.0
Position 1: 1/10 = 0.1
Position 2: 2/10 = 0.2
...
Position 9: 9/10 = 0.9

Distance entre mots adjacents : 0.1 - 0.0 = 0.1
```

**LE PROBLÈME :** les mots adjacents sont séparés de **0.2** dans la phrase de 5 mots, mais de **0.1** seulement dans celle de 10 mots !

**Pourquoi est-ce problématique ?**

**Problème 1 : Une distance incohérente pour le « mot suivant »**

Le modèle essaie d'apprendre : « Deux mots placés côte à côte ont probablement une relation étroite. »

Mais « côte à côte » signifie :

- Un écart de 0.2 dans une phrase de 5 mots.
- Un écart de 0.1 dans une phrase de 10 mots.
- Un écart de 0.05 dans une phrase de 20 mots.
- Un écart de 0.01 dans une phrase de 100 mots.

**Le modèle est désorienté !** Il ne peut pas apprendre une règle cohérente pour « ces mots sont adjacents », puisque leur distance numérique dépend de la longueur totale.

**Un exemple plus détaillé :**

Considérons l'apprentissage de la règle anglaise « l'adjectif précède généralement immédiatement le nom ».

**Dans une phrase de 5 mots :** « I love delicious homemade pizza ».
```
"delicious" (position 0.6) → "pizza" (position 0.8)
Distance: 0.8 - 0.6 = 0.2
Le modèle apprend : "Si la distance ≈ 0.2, les mots sont adjacents et l'adjectif qualifie le nom"
```

**Dans la phrase présentée comme ayant 20 mots :** « Yesterday I was really very hungry so I love eating delicious homemade thin crust pizza from Italy with extra cheese ».
```
"delicious" (position 0.45) → "homemade" (position 0.5)
Distance: 0.5 - 0.45 = 0.05
Le modèle hésite : "Une distance de 0.05 ? C'est minuscule ! Pourtant ces mots SONT adjacents…"
```

Le modèle doit apprendre : « Une distance de 0.2 signifie des mots adjacents dans une phrase courte, mais une distance de 0.05 signifie la même chose dans une phrase longue. » C'est comme attribuer une signification différente à une même distance selon la taille de la ville. Incohérent !

**Problème 2 : L'extrapolation à des séquences plus longues**

**Pendant l'entraînement**, supposons que nous ne rencontrions que des phrases de 50 mots au maximum. Le modèle apprend des encodages compris entre 0.0 et 1.0.

**Pendant l'inférence**, il rencontre un document de 100 mots. Que se passe-t-il ?

Les positions restent comprises entre 0.0 et 1.0, mais :

- À l'entraînement, pour 50 mots, 0.5 correspond au « milieu de la phrase », au mot 25.
- À l'inférence, pour 100 mots, 0.5 correspond au milieu, au mot 50.

**Le modèle a appris :** « La position 0.5 apparaît généralement dans le contexte X », sur la base de phrases de 50 mots.
**Mais il voit désormais :** « La position 0.5 dans un contexte très différent », celui d'un document de 100 mots.

Les mêmes valeurs d'encodage sont réutilisées pour des positions absolues différentes selon la longueur de la phrase, ce qui perturbe les associations apprises.

**Problème 3 : Pas d'information de position absolue**

Le modèle ne peut pas facilement apprendre : « Les mots en début de phrase prennent généralement une majuscule » ou « Les mots interrogatifs apparaissent souvent tôt. »

Car le « début » peut correspondre :

- Aux positions 0.0–0.1 dans une phrase de 10 mots.
- Aux positions 0.0–0.02 dans une phrase de 50 mots.

Le « début » n'est pas une plage numérique cohérente : il dépend de la longueur totale !

**Bilan de la tentative 2 :**

- ✓ Résolu : les valeurs sont bornées entre 0 et 1.
- ✓ Résolu : chaque position est unique.
- ❌ Problème : l'espacement des positions dépend de la longueur de la phrase.
- ❌ Problème : les mots adjacents ont des distances numériques différentes selon les phrases.
- ❌ Problème : le modèle peine à apprendre des régularités cohérentes.
- ❌ Problème : l'extrapolation aux séquences plus longues que celles de l'entraînement est difficile.

Nous progressons, mais il nous faut mieux !

---

### Tentative 3 : Une échelle différente par dimension

**« Et si nous utilisions des nombres DIFFÉRENTS selon les dimensions ? Au lieu d'ajouter la même position partout, chaque dimension pourrait suivre la position à une échelle différente ! »**

**L'idée :** utiliser des multiplicateurs différents pour que les dimensions évoluent à des vitesses différentes.

**Pour nos 6 dimensions, essayons :**

- Dimension 0 : position × 1.0, variation rapide.
- Dimension 1 : position × 0.5, variation moyenne.
- Dimension 2 : position × 0.1, variation lente.
- Dimension 3 : position × 0.05, variation très lente.
- Dimension 4 : position × 0.01, variation à peine perceptible.
- Dimension 5 : position × 0.001, presque immobile.

**Calculons les valeurs pour plusieurs positions.**

**Position 0 :**
```
Dimension 0: 0 × 1.0   = 0.0
Dimension 1: 0 × 0.5   = 0.0
Dimension 2: 0 × 0.1   = 0.0
Dimension 3: 0 × 0.05  = 0.0
Dimension 4: 0 × 0.01  = 0.0
Dimension 5: 0 × 0.001 = 0.0

Encodage positionnel : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

**Position 1 :**
```
Dimension 0: 1 × 1.0   = 1.0    ⚡ (forte variation !)
Dimension 1: 1 × 0.5   = 0.5    🐇 (moyen)
Dimension 2: 1 × 0.1   = 0.1    🐢 (petit)
Dimension 3: 1 × 0.05  = 0.05   🐢
Dimension 4: 1 × 0.01  = 0.01   🐌 (minuscule)
Dimension 5: 1 × 0.001 = 0.001  🐌

Encodage positionnel : [1.0, 0.5, 0.1, 0.05, 0.01, 0.001]
```

**Position 2 :**
```
Dimension 0: 2 × 1.0   = 2.0    ⚡ (fort changement !)
Dimension 1: 2 × 0.5   = 1.0    🐇
Dimension 2: 2 × 0.1   = 0.2    🐢
Dimension 3: 2 × 0.05  = 0.1    🐢
Dimension 4: 2 × 0.01  = 0.02   🐌
Dimension 5: 2 × 0.001 = 0.002  🐌

Encodage positionnel : [2.0, 1.0, 0.2, 0.1, 0.02, 0.002]
```

**Position 5 :**
```
Dimension 0: 5 × 1.0   = 5.0    ⚡
Dimension 1: 5 × 0.5   = 2.5    🐇
Dimension 2: 5 × 0.1   = 0.5    🐢
Dimension 3: 5 × 0.05  = 0.25   🐢
Dimension 4: 5 × 0.01  = 0.05   🐌
Dimension 5: 5 × 0.001 = 0.005  🐌

Encodage positionnel : [5.0, 2.5, 0.5, 0.25, 0.05, 0.005]
```

**Position 10 :**
```
Dimension 0: 10 × 1.0   = 10.0  ⚡
Dimension 1: 10 × 0.5   = 5.0   🐇
Dimension 2: 10 × 0.1   = 1.0   🐢
Dimension 3: 10 × 0.05  = 0.5   🐢
Dimension 4: 10 × 0.01  = 0.1   🐌
Dimension 5: 10 × 0.001 = 0.01  🐌

Encodage positionnel : [10.0, 5.0, 1.0, 0.5, 0.1, 0.01]
```

**C'est bien mieux !** Voyons ce que nous avons obtenu.

**✓ Des empreintes uniques :** chaque position possède une combinaison distincte !
```
Position 1: [1.0, 0.5, 0.1, 0.05, 0.01, 0.001]
Position 2: [2.0, 1.0, 0.2, 0.1,  0.02, 0.002] ← Différent de la position 1 !
Position 5: [5.0, 2.5, 0.5, 0.25, 0.05, 0.005] ← Différent des deux !
```

**✓ Des informations à plusieurs échelles :**

- Les **dimensions à variation rapide** (0, 1) distinguent les positions voisines, comme 1 et 2.
- Les **dimensions à variation lente** (4, 5) distinguent les positions éloignées, comme 10 et 100.

**L'analogie de l'horloge devient claire !**

- Dimension 0 (× 1.0) : la trotteuse, rapide.
- Dimension 2 (× 0.1) : l'aiguille des minutes, à vitesse moyenne.
- Dimension 4 (× 0.01) : l'aiguille des heures, lente.

Ensemble, elles créent des « signatures temporelles » uniques pour chaque position !

**Mais il reste un problème !** Voyons ce qui arrive aux positions plus élevées.

**Position 100 :**
```
Dimension 0: 100 × 1.0   = 100.0  ⚡ (IMMENSE !)
Dimension 1: 100 × 0.5   = 50.0   🐇 (GRAND !)
Dimension 2: 100 × 0.1   = 10.0   🐢 (devient grand…)
Dimension 3: 100 × 0.05  = 5.0    🐢
Dimension 4: 100 × 0.01  = 1.0    🐌 (raisonnable)
Dimension 5: 100 × 0.001 = 0.1    🐌 (encore petit)

Encodage positionnel : [100.0, 50.0, 10.0, 5.0, 1.0, 0.1]
```

**Position 1000 :**
```
Dimension 0: 1000 × 1.0   = 1000.0  ⚡ (MASSIF !)
Dimension 1: 1000 × 0.5   = 500.0   🐇 (ÉNORME !)
Dimension 2: 1000 × 0.1   = 100.0   🐢 (TRÈS GRAND !)
Dimension 3: 1000 × 0.05  = 50.0    🐢 (GRAND !)
Dimension 4: 1000 × 0.01  = 10.0    🐌 (devient grand…)
Dimension 5: 1000 × 0.001 = 1.0     🐌 (acceptable pour l'instant)

Encodage positionnel : [1000.0, 500.0, 100.0, 50.0, 10.0, 1.0]
```

**LE PROBLÈME DE CROISSANCE NON BORNÉE RÉAPPARAÎT !**

Même avec différentes échelles, les valeurs continuent à croître sans limite pour les longues séquences !

**Ajout de la position 1000 à un mot :**
```
Mot "document" à la position 1000 :
Embedding :      [0.45, -0.25,  0.60, -0.15,  0.30,  0.75]  ← Sens du mot (petit)
Encodage positionnel :   [1000,  500,   100,   50,    10,    1   ]  ← Position (IMMENSE !)
                      ─────  ─────  ─────  ─────  ─────  ─────
Combinaison :            [1000.45, 499.75, 100.60, 49.85, 10.30, 1.75]
                       ↑        ↑        ↑       ↑      ↑      ↑
                    La position domine dans les 5 premières dimensions !
```

**Le problème :** même la dimension 4, pourtant « lente », atteint 10.0 à la position 1000 : c'est **10 fois plus grand** que nos embeddings, de l'ordre de ±1. Le signal de position couvre encore le sens du mot !

**Pourquoi sommes-nous bloqués ?**

Chaque dimension doit représenter, après entraînement : **sens du mot + position**.

À la position 1000, la dimension 0 tenterait d'encoder :

- Le sens du mot : par exemple 0.45, son caractère de « document ».
- Le marqueur de position : 1000.
- La combinaison : 1000.45.

Le modèle voit « 1000.45 » et apprend « c'est la position 1000 », mais distingue difficilement la petite composante sémantique 0.45. Nous retrouvons le problème fondamental de la tentative 1 !

**Bilan de la tentative 3 :**

- ✓ Résolu : chaque position possède une combinaison unique.
- ✓ Résolu : l'information est disponible à plusieurs échelles, rapides et lentes.
- ✓ Amélioration : les dimensions lentes croissent moins vite.
- ❌ Problème : les valeurs restent non bornées pour les longues séquences.
- ❌ Problème : elles finissent encore par écraser les embeddings ; cela prend simplement plus de temps.
- ❌ Problème : il n'y a pas de borne supérieure constante.

Nous approchons ! L'idée des différentes échelles est bonne, mais il nous faut des valeurs qui ne dépassent JAMAIS une certaine limite…

---

## Partie 3 : La découverte décisive — comprendre les ondes

**De quoi avons-nous réellement besoin ?** Prenons du recul.

1. **Des valeurs bornées** : rester dans une plage raisonnable, comme -1 à +1.
2. **Des motifs uniques** : une empreinte différente pour chaque position.
3. **Des informations à plusieurs échelles** : certaines dimensions varient vite, d'autres lentement.
4. **Des distances cohérentes** : des écarts comparables entre positions adjacentes.
5. **Un fonctionnement pour toute longueur** : traiter des séquences plus longues que celles de l'entraînement.
6. **Une indépendance vis-à-vis de la longueur** : la position 5 doit toujours signifier « position 5 », pas « 25 % de la phrase ».

**Est-ce seulement possible ?**

Oui ! La réponse vient des **motifs répétitifs**, ou fonctions périodiques.

### Qu'est-ce qu'un motif répétitif ?

Nous en rencontrons partout dans la vie quotidienne.

**Les aiguilles d'une horloge :**
```
12 h → 1 h → 2 h → ... → 11 h → 12 h → 1 h → ...
L'aiguille des heures tourne et se répète toutes les 12 heures
```

**Les jours de la semaine :**
```
Lundi → Mardi → Mercredi → ... → Dimanche → Lundi → Mardi → ...
Le motif se répète tous les 7 jours
```

**Les saisons :**
```
Printemps → Été → Automne → Hiver → Printemps → Été → Automne → Hiver → ...
Le motif se répète chaque année
```

**Les vagues de l'océan :**
```
Haut → Bas → Haut → Bas → Haut → Bas → ...
Le niveau de l'eau monte et descend de manière cyclique
```

### Pourquoi les motifs répétitifs conviennent aux positions

**L'idée essentielle :** ils fournissent des valeurs **bornées**, tout en permettant de créer des **combinaisons uniques** !

**L'analogie du système d'horloges**

Imaginons TROIS horloges de vitesses différentes.

**Horloge 1 : la trotteuse** (très rapide ⚡)

- Elle effectue un tour complet toutes les 60 secondes.
- Sa position change sensiblement à chaque seconde.
- Après 60 secondes, elle revient au départ et recommence.

**Horloge 2 : l'aiguille des minutes** (vitesse moyenne 🐇)

- Elle effectue un tour complet toutes les 60 minutes.
- Sa position change visiblement à chaque minute.
- Elle met beaucoup plus longtemps à se répéter.

**Horloge 3 : l'aiguille des heures** (lente 🐢)

- Elle effectue un tour complet toutes les 12 heures.
- Elle bouge à peine d'une minute à l'autre.
- Elle met très longtemps à se répéter.

**Voici l'astuce :** même si chaque aiguille se répète individuellement, leur COMBINAISON identifie chaque instant dans un cycle de 12 heures !

```
Heure : 10:30:05
Trotteuse : pointe sur 1 (5 seconds)
Minutes : pointe sur 6 (30 minutes)  
Heures : pointe entre 10 et 11

Cette combinaison est unique dans un cycle de 12 heures !
```

**À 10 h 30 min 06 s, une seconde plus tard :**
```
Trotteuse : pointe sur 1.2 (6 seconds) ← Changement !
Minutes : pointe sur 6 (30 minutes) ← Identique
Heures : pointe entre 10 et 11 ← Identique

Combinaison différente ! Les aiguilles des minutes et heures ont à peine bougé.
```

**C'est exactement ce que nous allons faire avec les positions !** Nous utiliserons plusieurs « aiguilles », des motifs répétitifs, à différentes vitesses. Chaque position aura sa combinaison !

### Présentation du sinus et du cosinus : les ondes mathématiques

Le **sinus** et le **cosinus** sont des fonctions mathématiques qui produisent des ondes régulières et répétitives.

**Pas de panique si vous ne connaissez pas la trigonométrie !** Il suffit de comprendre ce que ces fonctions FONT.

**Le sinus :**

- Reçoit un nombre en entrée, comme un angle ou une position sur un cercle.
- Renvoie un nombre entre -1 et +1 : une valeur bornée ! ✓
- Produit une onde régulière qui se répète indéfiniment.

**Description visuelle de l'onde sinusoïdale :**
```
Départ à 0 → montée → sommet (+1) → descente → 
passage par 0 en descendant → creux (-1) → remontée → passage par 0 en montant → 
répétition indéfinie…
```

### Comprendre les valeurs du sinus et du cosinus

**Imaginez une grande roue :**

- Tout en bas : hauteur = -1.
- À mi-hauteur en montant : hauteur = 0.
- Tout en haut : hauteur = +1.
- À mi-hauteur en descendant : hauteur = 0.
- De nouveau en bas : hauteur = -1.

Quand la roue tourne, la hauteur monte et descend régulièrement entre -1 et +1. **C'est ce que fait le sinus !** Il décrit votre hauteur au cours de la rotation.

Le **sinus** ($\sin$) et le **cosinus** ($\cos$) :

- Reçoivent un **angle** en entrée, mesuré en **radians**.
- Renvoient une **valeur** comprise entre -1 et +1.
- Créent des ondes régulières et périodiques.

**📏 À propos des radians :** en apprentissage automatique, les fonctions trigonométriques utilisent les radians plutôt que les degrés.

- 0 radian = 0 degré : le point de départ.
- π radians ≈ 3.14 radians = 180 degrés : un demi-tour.
- 2π radians ≈ 6.28 radians = 360 degrés : un tour complet.

Ainsi, $\sin(1)$ signifie « sinus de 1 radian », environ 57 degrés, et non « sinus de 1 degré » ! Inutile de convertir ici : retenez qu'un cycle complet correspond à $2\pi \approx 6.28$ unités.

**Précision importante :** nous ne changeons PAS la fonction sinus. Elle est fixe et donne toujours la même sortie pour une même entrée. Nous changeons la VITESSE à laquelle ses angles d'entrée progressent lorsque la position augmente. C'est ce qui produit des fréquences « rapides » ou « lentes ».

### L'onde sinusoïdale : observer le motif complet

Regardons des valeurs concrètes. Inutile de savoir les calculer : observez simplement le motif. **Les valeurs ci-dessous sont arrondies à deux décimales.** Ainsi, 1.00 ou -1.00 signifie « très proche du sommet ou du creux », sans être nécessairement la valeur exacte.

```
Angle d'entrée → Valeur du sinus
─────────────────────────
0.0   →   0.00   ← Départ (milieu de l'onde)
0.5   →   0.48   ← Montée…
1.0   →   0.84   ← Toujours en montée…
1.5   →   1.00   ← Près du sommet
2.0   →   0.91   ← Redescente…
2.5   →   0.60   ← Toujours en descente…
3.0   →   0.14   ← Presque au milieu…
3.5   →  -0.35   ← Passage aux valeurs négatives !
4.0   →  -0.76   ← Descente vers le creux…
4.5   →  -0.98   ← Presque au creux…
5.0   →  -0.96   ← Près du creux
5.5   →  -0.71   ← Remontée…
6.0   →  -0.28   ← Toujours en montée…
6.5   →   0.22   ← Retour aux valeurs positives !
7.0   →   0.66   ← Montée…
7.5   →   0.94   ← De nouveau près du sommet !
8.0   →   0.99   ← Presque au sommet !
8.5   →   0.76   ← Début de descente…
9.0   →   0.41   ← Descente…
9.5   →  -0.08   ← Passage aux valeurs négatives…
10.0  →  -0.54   ← De nouveau négatif…
10.5  →  -0.88   ← Descente…
11.0  →  -1.00   ← Tout près du creux de l'onde
11.5  →  -0.88   ← Remontée…
12.0  →  -0.54   ← Toujours en montée…
12.5  →  -0.05   ← Presque de retour à zéro…
```

**Voyez-vous le motif ?**

- Départ à 0.
- Montée vers +1, le sommet.
- Descente et passage par 0.
- Descente vers -1, le creux.
- Remontée vers 0.
- **Puis tout recommence !**

Un cycle complet prend environ 6.28 unités, soit « 2π » en mathématiques, puis le motif se répète indéfiniment.

**Les propriétés qui nous intéressent :**

1. ✓ **Toujours borné :** jamais au-dessus de +1 ni au-dessous de -1.
2. ✓ **Variations régulières :** pas de saut brusque.
3. ✓ **Répétition indéfinie :** le motif continue sans fin.
4. ✓ **Des points distincts :** le texte source affirme ici que deux entrées d'un cycle ne donnent pas la même sortie. Cette affirmation nécessite le couple sinus/cosinus ; voir les notes de traduction.

### L'onde cosinusoïdale : la jumelle du sinus

Le cosinus est presque identique au sinus, mais **commence à un autre point du cycle**. Le tableau suivant est également arrondi à deux décimales.

```
Angle d'entrée → Valeur du cosinus
────────────────────────────
0.0   →   1.00   ← Départ (au sommet !)
0.5   →   0.88   ← Descente…
1.0   →   0.54   ← Descente…
1.5   →   0.07   ← Presque au milieu…
2.0   →  -0.42   ← Passage aux valeurs négatives !
2.5   →  -0.80   ← Descente vers le creux…
3.0   →  -0.99   ← Près du creux
3.5   →  -0.94   ← Remontée…
4.0   →  -0.65   ← Toujours en montée…
4.5   →  -0.21   ← Presque au milieu…
5.0   →   0.28   ← Retour aux valeurs positives !
5.5   →   0.71   ← Montée…
6.0   →   0.96   ← Presque au sommet…
6.5   →   0.99   ← Près du sommet…
7.0   →   0.75   ← Début de descente…
7.5   →   0.35   ← Descente…
8.0   →  -0.15   ← Passage aux valeurs négatives…
8.5   →  -0.61   ← Descente…
9.0   →  -0.91   ← Près du creux…
9.5   →  -0.99   ← Presque au creux…
10.0  →  -0.84   ← Remontée…
10.5  →  -0.53   ← Toujours en montée…
11.0  →  -0.00   ← De nouveau tout près du milieu
11.5  →   0.53   ← Montée…
12.0  →   0.84   ← Montée…
12.5  →   0.99   ← De nouveau presque au sommet…
```

**Retenez :**

- **Sinus :** départ au milieu, à 0, puis montée.
- **Cosinus :** départ au sommet, à +1, puis descente.
- **Même forme d'onde, point de départ différent !**

Comme deux coureurs sur une piste circulaire : l'un part de la ligne de départ, l'autre un quart de tour plus loin. Ils suivent la même boucle, avec un décalage !

### Pourquoi utiliser À LA FOIS le sinus et le cosinus ?

Nous pourrions théoriquement utiliser uniquement le sinus, mais les DEUX ensemble donnent davantage d'informations.

**Analogie : décrire un emplacement sur un cercle**

Imaginez que vous vous trouvez sur une piste circulaire. Pour indiquer EXACTEMENT votre emplacement, il faut DEUX informations.

```
Seulement la distance parcourue : "J'ai parcouru 5 mètres sur la piste"
Problème : dans quel sens ? Sens horaire ou antihoraire ? Tour 1 ou tour 2 ?

Mieux : "Je suis aux coordonnées (3m à l'est, 4m au nord du centre)"
Parfait ! La position est précise, sans ambiguïté.
```

Le sinus et le cosinus jouent le rôle des coordonnées horizontale et verticale d'un cercle. Ensemble, ils indiquent complètement où vous êtes dans le cycle !

**Pour le suivi d'une position, le couple fournit :**

- La valeur du sinus : à quelle hauteur sommes-nous dans l'onde ?
- La valeur du cosinus : sommes-nous dans la phase montante ou descendante ?
- Ensemble : l'emplacement exact dans le cycle, sans ambiguïté !

C'est particulièrement utile lorsque nous combinons plusieurs fréquences : les couples sinus/cosinus aident à distinguer plus précisément les positions.

### Que signifient exactement « vitesse » et « fréquence » ici ?

Clarifions ce point souvent déroutant.

**Rappel :** le sinus reçoit un **angle** et renvoie une **valeur** entre -1 et +1.

```
sin(angle) = valeur
```

Quand nous parlons de « vitesse » ou de « fréquence », la fonction sinus ne change PAS : il s'agit de **la vitesse de progression des angles lorsque la position augmente !**

**Basse fréquence (vitesse lente 🐢) :**
```
Position 0 → angle = 0.0      → sin(0.0)   = 0.00
Position 1 → angle = 0.002    → sin(0.002) = 0.002  (presque immobile !)
Position 2 → angle = 0.004    → sin(0.004) = 0.004  (toujours peu de variation)
Position 3 → angle = 0.006    → sin(0.006) = 0.006
...
Position 100 → angle = 0.2    → sin(0.2)   = 0.199  (variation enfin visible)
```

Les angles progressent LENTEMENT avec la position, et les valeurs du sinus changent donc LENTEMENT. Il faut beaucoup de positions pour observer un changement important !

**Haute fréquence (vitesse rapide ⚡) :**
```
Position 0 → angle = 0        → sin(0)     = 0.00
Position 1 → angle = 1        → sin(1)     = 0.84   (grand saut !)
Position 2 → angle = 2        → sin(2)     = 0.91   (fort changement !)
Position 3 → angle = 3        → sin(3)     = 0.14   (encore différent !)
Position 4 → angle = 4        → sin(4)     = -0.76  (complètement différent !)
```

Les angles progressent RAPIDEMENT avec la position et les valeurs du sinus changent RAPIDEMENT. Chaque position paraît très différente !

**La formule contrôle la vitesse :**

Dans $\sin\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$, le dénominateur $10000^{2i/d_{\text{model}}}$ règle la progression des angles.

- **Grand dénominateur**, comme 464.159 → progression LENTE → basse fréquence 🐢.
  - Position 1 : 1/464.159 = 0.002, un angle minuscule.
  - Position 2 : 2/464.159 = 0.004, toujours minuscule.

- **Petit dénominateur**, comme 1.0 → progression RAPIDE → haute fréquence ⚡.
  - Position 1 : 1/1 = 1, un grand angle.
  - Position 2 : 2/1 = 2, un angle encore plus grand.

**La « vitesse » ou « fréquence » indique donc la rapidité avec laquelle nous parcourons l'onde lorsque la position augmente.**

- Haute fréquence : grands pas dans les angles, fortes variations entre positions.
- Basse fréquence : petits pas dans les angles, faibles variations entre positions.

**Pourquoi différentes vitesses ?** La combinaison des dimensions rapides et lentes crée des empreintes distinctes ! Si toutes les dimensions évoluaient à la même vitesse, différentes positions pourraient trop se ressembler.

**Bilan de la partie 3 :**

- ✓ Le sinus et le cosinus donnent des valeurs **bornées**, entre -1 et +1.
- ✓ Ils produisent des **motifs répétitifs**, comme les aiguilles d'une horloge.
- ✓ Plusieurs **vitesses**, ou fréquences, produisent ensemble des **combinaisons distinctes**.
- ✓ Comme les aiguilles des secondes, minutes et heures, ces dimensions identifient une position par leur combinaison !

Voyons maintenant COMMENT transformer cette idée en nombres…

---

## Partie 4 : Construire la solution étape par étape

Nous pouvons construire la formule de l'encodage positionnel. Examinons chaque composante.

### La formule complète (vue d'ensemble)

Pour la position $\text{pos}$ et la paire de dimensions $i$ :

$$
\text{PE}(\text{pos}, 2i) = \sin\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right) \quad \text{(dimensions paires : 0, 2, 4)}
$$

$$
\text{PE}(\text{pos}, 2i+1) = \cos\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right) \quad \text{(dimensions impaires : 1, 3, 5)}
$$

**Pas de panique !** Décomposons la formule. À la fin, vous comprendrez chaque symbole.

### Étape 1 : Comprendre les paires de dimensions

**Le principe :** les dimensions vont par paires. Chaque paire utilise la même fréquence, mais une dimension emploie le sinus et l'autre le cosinus.

Pour $d_{\text{model}} = 6$, notre exemple :

**Paire 0** ($i = 0$) :

- Dimension 0 : sinus.
- Dimension 1 : cosinus.
- Même fréquence, très rapide, pour les deux.

**Paire 1** ($i = 1$) :

- Dimension 2 : sinus.
- Dimension 3 : cosinus.
- Même fréquence moyenne.

**Paire 2** ($i = 2$) :

- Dimension 4 : sinus.
- Dimension 5 : cosinus.
- Même fréquence lente.

**Pourquoi des paires ?** Le sinus et le cosinus associés à chaque fréquence donnent une information complète sur l'emplacement dans le cycle, sans ambiguïté !

### Étape 2 : Le nombre magique 10 000

L'article original sur le Transformer utilise **10 000** comme base. Pourquoi ?

**Le raisonnement :**

- Nous voulons que l'onde la plus lente évolue TRÈS lentement.
- Elle pourra ainsi distinguer des positions dans de très longues séquences, de plusieurs milliers de mots.
- Selon l'explication de la source, 10 000 est assez grand pour que cette onde ait à peine effectué un cycle à la position 1000.

**Imaginez :**

- Avec 100 comme diviseur : l'onde termine un cycle toutes les 628 positions environ, car 100 × 2π ≈ 628.
- Avec 10 000 comme diviseur : elle termine un cycle toutes les 62 800 positions environ !

Même dans un document de 10 000 mots, l'onde lente fournirait encore une information de position distinctive ! 🎯

Le choix de 10 000 est en partie arbitraire : les chercheurs ont constaté qu'il fonctionnait bien. On pourrait choisir 5000 ou 20 000, mais 10 000 est devenu la valeur habituelle de cette formule.

### Étape 3 : Calculer la fréquence de chaque paire

Quel diviseur, donc quelle fréquence, choisir pour la paire $i$ ?

Nous souhaitons :

- Paire 0 ($i=0$) : un diviseur proche de 1, le plus rapide.
- Paire 1 ($i=1$) : un diviseur intermédiaire.
- Paire 2 ($i=2$) : un grand diviseur ; la source l'annonce proche de 10 000, mais le calcul ci-dessous donne environ 464 pour six dimensions.

**Ces diviseurs doivent être régulièrement répartis sur une échelle logarithmique**, et non concentrés dans une seule plage.

**La formule : une progression exponentielle !**

$$\text{divisor} = 10000^{2i/d_{\text{model}}}$$

Calculons pour $d_{\text{model}} = 6$.

**Paire 0** ($i = 0$, dimensions 0 et 1) :

$$\text{divisor} = 10000^{2 \times 0/6} = 10000^{0/6} = 10000^{0} = 1$$

Toute base non nulle élevée à la puissance 0 vaut 1 ! Nous divisons donc par 1 : l'onde la plus rapide.

**Paire 1** ($i = 1$, dimensions 2 et 3) :

$$\text{divisor} = 10000^{2 \times 1/6} = 10000^{2/6} = 10000^{1/3}$$

$10000^{1/3}$ est la racine cubique de 10 000 :

$$10000^{1/3} = \sqrt[3]{10000} ≈ 21.544$$

Nous divisons donc par environ 21.5 : l'onde intermédiaire.

**Paire 2** ($i = 2$, dimensions 4 et 5) :

$$\text{divisor} = 10000^{2 \times 2/6} = 10000^{4/6} = 10000^{2/3}$$

$10000^{2/3}$ est le carré de la racine cubique de 10 000 :

$$10000^{2/3} = (10000^{1/3})^2 ≈ 21.544^2 ≈ 464.159$$

Nous divisons donc par environ 464 : l'onde lente.

**Observons la progression :**

| Paire | $i$ | Exposant $2i/d_{\text{model}}$ | Diviseur $10000^{\text{exponent}}$ | Vitesse |
|------|-----|-------------------------------|----------------------------------|-------|
| 0    | 0   | 0/6 = 0.000                   | $10000^{0.000}$ = 1              | ⚡⚡⚡ Très rapide |
| 1    | 1   | 2/6 = 0.333                   | $10000^{0.333}$ ≈ 21.5           | 🐇 Moyenne |
| 2    | 2   | 4/6 = 0.667                   | $10000^{0.667}$ ≈ 464            | 🐢 Lente |

Les exposants 0, 0.333 et 0.667 sont régulièrement espacés entre 0 et 1.

En élevant 10 000 à ces puissances, nous obtenons des diviseurs régulièrement espacés sur une **échelle logarithmique** :

- De 1 à 21.5 : multiplication par environ 21.
- De 21.5 à 464 : multiplication par environ 21.

Chaque paire est environ 21 fois plus lente que la précédente ! Cet espacement permet de couvrir progressivement une très grande plage d'échelles.

**Pourquoi une échelle logarithmique ?** Avec une progression linéaire, comme 1, 5000, 10 000, les variations intéressantes seraient concentrées au début et certaines dimensions seraient peu utiles. L'espacement logarithmique apporte de l'information à toutes les échelles !

### Étape 4 : Assembler le tout

Reprenons la formule complète :

$$\text{PE}(\text{pos}, 2i) = \sin\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

**Décodons chaque élément.**

**$\text{PE}(\text{pos}, \text{dim})$**

- L'encodage positionnel à la position « pos », dans la dimension « dim ».
- C'est le nombre que nous calculons.

**$\text{pos}$**

- La position dans la phrase : 0, 1, 2, 3…
- Position 0 : premier mot ; position 1 : deuxième mot, etc.

**$2i$**, pour les dimensions paires :

- $i$ est l'indice de la paire : 0, 1, 2…
- Pour $i=0$ : dimension $2i=0$, paire, donc sinus.
- Pour $i=1$ : dimension $2i=2$, paire, donc sinus.
- Pour $i=2$ : dimension $2i=4$, paire, donc sinus.

**$2i+1$**, pour les dimensions impaires :

- Pour $i=0$ : dimension $2i+1=1$, impaire, donc cosinus.
- Pour $i=1$ : dimension $2i+1=3$, impaire, donc cosinus.
- Pour $i=2$ : dimension $2i+1=5$, impaire, donc cosinus.

**$d_{\text{model}}$**

- Le nombre total de dimensions : 6 ici.
- C'est la taille de chaque vecteur de mot.

**$10000^{2i/d_{\text{model}}}$**

- Le diviseur qui règle la vitesse de l'onde.
- Première paire ($i=0$) : petit diviseur de 1, onde rapide.
- Paire intermédiaire ($i=1$) : diviseur d'environ 21.5, onde intermédiaire.
- Dernière paire ($i=2$) : grand diviseur d'environ 464, onde lente.

**$\sin(\ldots)$ et $\cos(\ldots)$**

- Les fonctions qui créent les motifs répétitifs.
- Sinus pour les dimensions paires, cosinus pour les impaires.
- Toutes deux bornées entre -1 et +1.

**$\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}$**

- L'**angle** donné au sinus ou au cosinus.
- Il augmente avec la position.
- Sa vitesse de progression dépend du diviseur.

**La procédure complète :**

1. Choisir une position, par exemple 2, le troisième mot.
2. Choisir une dimension, par exemple 0.
3. Identifier sa paire : $i = 0$ pour la dimension 0.
4. Calculer le diviseur : $10000^{0/6} = 1$.
5. Calculer l'angle : $\text{pos}/\text{divisor} = 2/1 = 2$.
6. Appliquer le sinus, car la dimension 0 est paire : $\sin(2) ≈ 0.909$.
7. Nous avons l'encodage de la position 2 dans la dimension 0 !

**Récapitulatif des vitesses :**

| Dimension | Paire $i$ | Formule | Diviseur | Vitesse | Rôle |
|-----------|----------|---------|---------|-------|---------|
| 0 (sinus) | 0 | $10000^{0/6}$ | 1.0 | ⚡⚡⚡ | Distinguer les positions adjacentes |
| 1 (cosinus) | 0 | $10000^{0/6}$ | 1.0 | ⚡⚡⚡ | Même rôle, avec déphasage |
| 2 (sinus) | 1 | $10000^{2/6}$ | 21.5 | 🐇 | Distinguer les groupes proches |
| 3 (cosinus) | 1 | $10000^{2/6}$ | 21.5 | 🐇 | Même rôle, avec déphasage |
| 4 (sinus) | 2 | $10000^{4/6}$ | 464.2 | 🐢 | Distinguer les régions éloignées |
| 5 (cosinus) | 2 | $10000^{4/6}$ | 464.2 | 🐢 | Même rôle, avec déphasage |

**L'idée essentielle :** les SIX dimensions encodent ensemble la position du mot. Les dimensions rapides indiquent sa position locale précise, les intermédiaires son voisinage et les lentes sa région globale. Elles forment une « empreinte » à six nombres !

---

## Partie 5 : Calculer les encodages positionnels (à vous de jouer !)

CALCULONS maintenant les encodages ! Nous examinerons chaque dimension pour plusieurs positions afin de voir les motifs apparaître.

### Préparer le calcul

Nous avons :

- Trois positions, 0, 1 et 2, pour « I love pizza ».
- Nous calculerons aussi dix positions, de 0 à 9, pour observer les motifs.
- Six dimensions, de 0 à 5.
- Un encodage positionnel à calculer pour chacune !

### D'abord : calculer tous les diviseurs

Calculons une fois les diviseurs de chaque paire.

**Dimensions 0 et 1, paire $i=0$ :**

$$10000^{2 \times 0 / 6} = 10000^0 = 1.000$$

**Dimensions 2 et 3, paire $i=1$ :**

$$10000^{2 \times 1 / 6} = 10000^{1/3} = \sqrt[3]{10000} ≈ 21.544$$

**Dimensions 4 et 5, paire $i=2$ :**

$$10000^{2 \times 2 / 6} = 10000^{2/3} = (\sqrt[3]{10000})^2 ≈ 464.159$$

**Table de référence :**

| Dimension | Paire $i$ | Diviseur | Vitesse |
|-----------|----------|---------|-------|
| 0 (sinus) | 0 | 1.000 | ⚡⚡⚡ |
| 1 (cosinus) | 0 | 1.000 | ⚡⚡⚡ |
| 2 (sinus) | 1 | 21.544 | 🐇 |
| 3 (cosinus) | 1 | 21.544 | 🐇 |
| 4 (sinus) | 2 | 464.159 | 🐢 |
| 5 (cosinus) | 2 | 464.159 | 🐢 |

Nous pouvons maintenant calculer chaque position.

### Calculer les encodages de « I love pizza »

Calculons en détail les encodages des positions 0, 1 et 2.

**Position 0 (« I ») :**

Pour dim₀, paire, $i=0$ :

$$\text{PE}(0, 0) = \sin\left(\frac{0}{10000^{0/6}}\right) = \sin(0) = 0.0000$$

Pour dim₁, impaire, $i=0$ :

$$\text{PE}(0, 1) = \cos\left(\frac{0}{10000^{0/6}}\right) = \cos(0) = 1.0000$$

Pour dim₂, paire, $i=1$ :

$$\text{PE}(0, 2) = \sin\left(\frac{0}{10000^{2/6}}\right) = \sin(0) = 0.0000$$

Pour dim₃, impaire, $i=1$ :

$$\text{PE}(0, 3) = \cos\left(\frac{0}{10000^{2/6}}\right) = \cos(0) = 1.0000$$

Pour dim₄, paire, $i=2$ :

$$\text{PE}(0, 4) = \sin\left(\frac{0}{10000^{4/6}}\right) = \sin(0) = 0.0000$$

Pour dim₅, impaire, $i=2$ :

$$\text{PE}(0, 5) = \cos\left(\frac{0}{10000^{4/6}}\right) = \cos(0) = 1.0000$$

**Encodage de la position 0 :** $[0.00, 1.00, 0.00, 1.00, 0.00, 1.00]$

**Position 1 (« love ») :**

Rappel des dénominateurs :

- Dimensions 0 et 1 : $10000^{0/6} = 10000^0 = 1$.
- Dimensions 2 et 3 : $10000^{2/6} = 10000^{1/3} = 21.544$.
- Dimensions 4 et 5 : $10000^{4/6} = 10000^{2/3} = 464.159$.

Voici les valeurs :

$$\text{dim}_0 = \sin(1/1) = \sin(1) = 0.8415$$

$$\text{dim}_1 = \cos(1/1) = \cos(1) = 0.5403$$

$$\text{dim}_2 = \sin(1/21.544) = \sin(0.0464) = 0.0464$$

$$\text{dim}_3 = \cos(1/21.544) = \cos(0.0464) = 0.9989$$

$$\text{dim}_4 = \sin(1/464.159) = \sin(0.0022) = 0.0022$$

$$\text{dim}_5 = \cos(1/464.159) = \cos(0.0022) = 1.0000$$

**Encodage de la position 1 :** $[0.84, 0.54, 0.05, 1.00, 0.00, 1.00]$

**Position 2 (« pizza ») :**

$$\text{dim}_0 = \sin(2) = 0.9093$$

$$\text{dim}_1 = \cos(2) = -0.4161$$

$$\text{dim}_2 = \sin(2/21.544) = 0.0927$$

$$\text{dim}_3 = \cos(2/21.544) = 0.9957$$

$$\text{dim}_4 = \sin(2/464.159) = 0.0043$$

$$\text{dim}_5 = \cos(2/464.159) = 1.0000$$

**Encodage de la position 2 :** $[0.91, -0.42, 0.09, 1.00, 0.00, 1.00]$

**Récapitulatif des calculs :**
```
Position 0: [0.00, 1.00, 0.00, 1.00, 0.00, 1.00]
Position 1: [0.84, 0.54, 0.05, 1.00, 0.00, 1.00]
Position 2: [0.91, -0.42, 0.09, 1.00, 0.00, 1.00]
```

Chaque position possède une empreinte UNIQUE à six nombres ! La position 0 diffère de la 1, qui diffère de la 2. Les dimensions rapides 0 et 1 varient beaucoup, tandis que les dimensions lentes 4 et 5 changent à peine à ces premières positions.

### Observer le motif : davantage de positions !

Trois positions ne suffisent pas pour bien VOIR les motifs. Calculons les positions 0 à 9, comme pour une phrase de dix mots, et observons les ondes !

**Table des encodages positionnels**, avec des valeurs arrondies :

| Position | dim₀  | dim₁  | dim₂  | dim₃  | dim₄  | dim₅  |
|----------|-------|-------|-------|-------|-------|-------|
| 0        | 0.00  | 1.00  | 0.00  | 1.00  | 0.00  | 1.00  |
| 1        | 0.84  | 0.54  | 0.05  | 1.00  | 0.00  | 1.00  |
| 2        | 0.91  | -0.42 | 0.09  | 1.00  | 0.00  | 1.00  |
| 3        | 0.14  | -0.99 | 0.14  | 0.99  | 0.01  | 1.00  |
| 4        | -0.76 | -0.65 | 0.19  | 0.98  | 0.01  | 1.00  |
| 5        | -0.96 | 0.28  | 0.23  | 0.97  | 0.01  | 1.00  |
| 6        | -0.28 | 0.96  | 0.28  | 0.96  | 0.01  | 1.00  |
| 7        | 0.66  | 0.75  | 0.32  | 0.95  | 0.02  | 1.00  |
| 8        | 0.99  | -0.15 | 0.37  | 0.93  | 0.02  | 1.00  |
| 9        | 0.41  | -0.91 | 0.42  | 0.91  | 0.02  | 1.00  |

**Regardons les motifs !**

**Dimensions 0 et 1, très rapides ⚡⚡⚡ :**

- Elles changent fortement à chaque position !
- Position 0 : [0.00, 1.00].
- Position 1 : [0.84, 0.54], complètement différente.
- Position 2 : [0.91, -0.42], encore différente.
- Elles oscillent comme une corde à sauter : positif, négatif, positif, négatif…

**Dimensions 2 et 3, intermédiaires 🐇 :**

- Elles changent progressivement.
- Position 0 : [0.00, 1.00].
- Position 5 : [0.23, 0.97], un léger décalage.
- Position 9 : [0.42, 0.91], un changement visible mais doux.

**Dimensions 4 et 5, lentes 🐢 :**

- Elles bougent à peine à ces premières positions !
- Position 0 : [0.00, 1.00].
- Position 9 : [0.02, 1.00], presque identique.
- Elles seront plus utiles pour distinguer les positions 100 et 200.

**L'astuce :** chaque ligne est DIFFÉRENTE ! Deux positions ne partagent pas le même motif à six nombres. Chacune possède sa propre empreinte, son QR code ou son signe de reconnaissance !

Choisissez deux lignes : elles sont toujours différentes. C'est ainsi que le modèle distingue les positions 3 et 7, même si elles se trouvent toutes deux « quelque part au milieu ».

## Partie 6 : Combiner position et sens du mot

Il faut maintenant COMBINER l'information de position avec le sens du mot.

**Au chapitre 3**, nos embeddings étaient :
```
"I":     [0.01, -0.20, 0.30, 0.15, -0.05, 0.11]
"love":  [-0.40, 0.60, 0.00, 0.25, 0.90, -0.30]
"pizza": [0.90, -0.10, -0.20, -0.80, 0.50, 0.75]
```

Ils représentent le SENS de chaque mot.

**À la partie 5, nous avons calculé les encodages positionnels :**
```
Position 0: [0.00, 1.00, 0.00, 1.00, 0.00, 1.00]
Position 1: [0.84, 0.54, 0.05, 1.00, 0.00, 1.00]
Position 2: [0.91, -0.42, 0.09, 1.00, 0.00, 1.00]
```

Ils représentent la POSITION de chaque mot.

**Comment les combiner ?** Par une simple addition composante par composante !

**Qu'est-ce que cela signifie ?** Additionner les valeurs correspondantes :

- Premier nombre + premier nombre.
- Deuxième nombre + deuxième nombre.
- Troisième nombre + troisième nombre.
- Et ainsi de suite…

**Exemple avec de petits nombres :**
```
[1, 2, 3] + [10, 20, 30] = [11, 22, 33]
```

Les composantes correspondantes sont additionnées indépendamment. C'est comme réunir les soldes de deux comptes bancaires !

### Ajouter la position à « I » (position 0)

Procédons dimension par dimension.

```
Embedding du mot "I":    [0.01, -0.20,  0.30,  0.15, -0.05,  0.11]
Encodage position 0 :   [0.00,  1.00,  0.00,  1.00,  0.00,  1.00]
                        ────   ────   ────   ────   ────   ────
Dimension 0:  0.01 + 0.00 = 0.01   (le sens change à peine)
Dimension 1: -0.20 + 1.00 = 0.80   (la position ajoute beaucoup ici !)
Dimension 2:  0.30 + 0.00 = 0.30   (sens inchangé)
Dimension 3:  0.15 + 1.00 = 1.15   (la position ajoute beaucoup ici !)
Dimension 4: -0.05 + 0.00 = -0.05  (sens inchangé)
Dimension 5:  0.11 + 1.00 = 1.11   (la position ajoute beaucoup ici !)
                        ────   ────   ────   ────   ────   ────
Résultat combiné :       [0.01,  0.80,  0.30,  1.15, -0.05,  1.11]
```

**L'idée essentielle : le résultat contient les DEUX signaux !**

- Dimensions 0, 2 et 4 : principalement l'embedding initial, car l'ajout positionnel vaut environ 0.
- Dimensions 1, 3 et 5 : mélange de l'embedding et de la position, car l'ajout vaut 1.0.

Le mot « I » porte maintenant son sens ET sa position !

### Ajouter la position à « love » (position 1)

```
Embedding du mot "love": [-0.40,  0.60,  0.00,  0.25,  0.90, -0.30]
Encodage position 1 :   [ 0.84,  0.54,  0.05,  1.00,  0.00,  1.00]
                        ─────  ─────  ─────  ─────  ─────  ─────
Dimension 0: -0.40 + 0.84 =  0.44   (fort décalage positionnel !)
Dimension 1:  0.60 + 0.54 =  1.14   (les deux contribuent)
Dimension 2:  0.00 + 0.05 =  0.05   (petit signal de position)
Dimension 3:  0.25 + 1.00 =  1.25   (la position domine)
Dimension 4:  0.90 + 0.00 =  0.90   (sens préservé)
Dimension 5: -0.30 + 1.00 =  0.70   (la position le décale)
                        ─────  ─────  ─────  ─────  ─────  ─────
Résultat combiné :       [ 0.44,  1.14,  0.05,  1.25,  0.90,  0.70]
```

Les valeurs de la position 1 diffèrent de celles de la position 0 ! Le mot « love » encode désormais :

- Son sens, issu de l'embedding.
- Son emplacement, issu de l'encodage positionnel.

### Ajouter la position à « pizza » (position 2)

```
Embedding du mot "pizza": [ 0.90, -0.10, -0.20, -0.80,  0.50,  0.75]
Encodage position 2 :    [ 0.91, -0.42,  0.09,  1.00,  0.00,  1.00]
                         ─────  ─────  ─────  ─────  ─────  ─────
Dimension 0:  0.90 + 0.91 =  1.81   (deux grandes valeurs !)
Dimension 1: -0.10 + -0.42 = -0.52  (deux valeurs négatives s'ajoutent)
Dimension 2: -0.20 + 0.09 = -0.11   (compensation partielle)
Dimension 3: -0.80 + 1.00 =  0.20   (la position l'emporte sur le mot)
Dimension 4:  0.50 + 0.00 =  0.50   (sens préservé)
Dimension 5:  0.75 + 1.00 =  1.75   (les deux contribuent)
                         ─────  ─────  ─────  ─────  ─────  ─────
Résultat combiné :        [ 1.81, -0.52, -0.11,  0.20,  0.50,  1.75]
```

**Observation :** la position 2 possède encore un motif distinct, différent des positions 0 et 1.

### La matrice finale tenant compte des positions

Empilons les trois vecteurs combinés.

$$
\mathbf{X}_{\text{pos}} = \begin{bmatrix}
0.01 & 0.80 & 0.30 & 1.15 & -0.05 & 1.11 \\
0.44 & 1.14 & 0.05 & 1.25 & 0.90 & 0.70 \\
1.81 & -0.52 & -0.11 & 0.20 & 0.50 & 1.75
\end{bmatrix}
$$

**Qu'avons-nous obtenu ?**

- Ligne 1 : « I » à la position 0, sens du mot + information de position.
- Ligne 2 : « love » à la position 1, sens du mot + information de position.
- Ligne 3 : « pizza » à la position 2, sens du mot + information de position.

Chaque vecteur **tient maintenant compte de sa position** ! Le Transformer peut exploiter :

- Le sens de chaque mot, provenant de l'embedding initial.
- L'emplacement de chaque mot, provenant de l'encodage positionnel.
- L'ordre « I » d'abord, « love » ensuite, puis « pizza ».

**Analogie :** un badge lors d'une conférence.

- Votre NOM, l'embedding, indique qui vous êtes.
- Votre NUMÉRO DE PLACE, l'encodage positionnel, indique où vous êtes assis.
- Ces deux informations figurent sur le même badge, le vecteur combiné.

Les ondes produisent des motifs distincts pour les positions 0, 1, 2, et ainsi de suite !

---

## Partie 7 : Une question essentielle sur l'entraînement

### L'entraînement ne risque-t-il pas de modifier les positions ?

**« Nous avons AJOUTÉ les valeurs de position aux embeddings. Mais l'entraînement ne va-t-il pas mettre ces nombres à jour ? L'information de position ne va-t-elle pas changer ou disparaître ? »**

**Excellente question !** C'est une source de confusion fréquente.

**RÉPONSE COURTE :** les encodages positionnels sinusoïdaux sont COMPLÈTEMENT FIXES. Ils ne sont JAMAIS mis à jour pendant l'entraînement. Seuls les embeddings des mots le sont dans cette comparaison.

**Voyons ce qui se passe.**

**AVANT l'entraînement, avec les valeurs calculées précédemment :**
```
Embedding du mot "pizza": [0.90, -0.10, -0.20, -0.80,  0.50,  0.75]  ← VALEURS INITIALES ALÉATOIRES
Encodage position 2 :    [0.91, -0.42,  0.09,  1.00,  0.00,  1.00]  ← FORMULE FIXE
                         ─────  ─────  ─────  ─────  ─────  ─────
Combinaison :               [1.81, -0.52, -0.11,  0.20,  0.50,  1.75]  ← ENTRÉE DU TRANSFORMER
```

**APRÈS un peu d'entraînement, les embeddings ont appris de meilleures valeurs :**
```
Embedding du mot "pizza": [0.88, -0.15, -0.18, -0.79,  0.48,  0.73]  ← MODIFIÉ ! (appris)
Encodage position 2 :    [0.91, -0.42,  0.09,  1.00,  0.00,  1.00]  ← TOUJOURS IDENTIQUE !
                         ─────  ─────  ─────  ─────  ─────  ─────
Combinaison :               [1.79, -0.57, -0.09,  0.21,  0.48,  1.73]  ← NOUVEAU VECTEUR COMBINÉ
```

**Qu'est-ce qui a changé ?**

- L'embedding a APPRIS de meilleures valeurs : de [0.90, -0.10…] à [0.88, -0.15…].
- L'encodage positionnel est resté IDENTIQUE : [0.91, -0.42…] n'a pas changé.
- Le résultat combiné a changé uniquement parce que l'embedding a changé.

**Comment est-ce possible ?**

Pendant l'entraînement :

1. Nous calculons le vecteur combiné : embedding + position.
2. Nous le faisons passer dans le Transformer.
3. Nous calculons les gradients pour améliorer le modèle.
4. **Mais**, pour ces deux composantes, seules les valeurs de la matrice d'embeddings sont mises à jour.
5. Les encodages positionnels sont marqués comme « fixes » ; leurs gradients ne sont pas calculés.

**L'analogie du papier quadrillé :**

Imaginez que vous dessinez sur du papier quadrillé.

- Les **lignes du quadrillage**, les encodages positionnels, sont imprimées sur la feuille : elles sont fixes et permanentes.
- Les **points que vous dessinez**, les embeddings, sont ce que vous apprenez à placer correctement.
- En vous exerçant, vous déplacez les points pour trouver les bons emplacements SUR LE QUADRILLAGE FIXE.
- Les lignes ne bougent jamais : elles constituent le référentiel stable.
- Vous apprenez qu'un point en (2, 3) n'a pas la même signification qu'un point en (5, 7).

Le modèle apprend de même à placer les significations sur un référentiel positionnel fixe. Les embeddings s'ajustent, mais le quadrillage reste constant.

**Autre image :** les encodages positionnels sont comme une règle sur votre bureau. Vous ne la déformez pas quand vous apprenez à mesurer : elle reste fixe, et vous apprenez à la lire et à l'utiliser !

**Les mathématiques, pour les curieux :**

Lors de la rétropropagation :

$$\text{Combined} = \text{Embedding} + \text{PositionEncoding}$$

La dérivée par rapport à l'embedding est :

$$\frac{\partial \text{Combined}}{\partial \text{Embedding}} = 1$$

Le gradient traverse directement l'addition ! Comme pour $(x + 5)$ : quand on met $x$ à jour, le 5 reste constant.

**En pratique :** les encodages positionnels sont marqués « requires_grad=False », ou conservés comme données non entraînables, pour que l'optimiseur ne les mette pas à jour.

**Pourquoi cela fonctionne :**

1. Les positions sont CALCULÉES par une formule, et ne sont pas des paramètres appris.
2. Elles fournissent un référentiel stable.
3. Les embeddings apprennent à encoder le sens DANS ce référentiel.
4. Le modèle apprend à interpréter une valeur, par exemple 0.5 dans la dimension 1, en tenant compte de la position et du contexte.

---

## Partie 8 : Pourquoi cette solution est élégante

Récapitulons les propriétés des encodages sinus/cosinus.

- **✓ Bornés :** leurs valeurs restent entre -1 et +1, ce qui limite leur amplitude.
- **✓ Distinctifs :** chaque position de notre exemple possède une empreinte à six nombres.
- **✓ Plusieurs échelles :** les dimensions rapides, à haute fréquence, et lentes, à basse fréquence, donnent des combinaisons distinctes.
- **✓ Positions relatives :** les relations mathématiques entre les ondes aident à représenter « ces mots sont séparés de trois positions ».
- **✓ Longueurs arbitraires :** la formule peut être évaluée pour des séquences plus longues que celles de l'entraînement ; cela ne garantit pas la qualité de l'extrapolation.
- **✓ Aucun apprentissage nécessaire pour cet encodage :** les valeurs sont calculées par une formule.
- **✓ Encodages fixes pendant l'entraînement :** les positions restent constantes pendant que les embeddings apprennent.

---

## Partie 9 : Récapitulatif et points à retenir

Avant de poursuivre, assurez-vous de comprendre :

1. **Le problème :** le traitement parallèle ne fournit pas à lui seul l'ordre des mots ; une information de position est nécessaire.
2. **Les solutions simples échouent :** 1, 2, 3… pose un problème d'échelle ; la normalisation dépend de la longueur de la phrase.
3. **La solution sinus/cosinus :** elle crée des empreintes positionnelles bornées et à plusieurs échelles.
4. **Les positions fixes :** ces encodages ne changent JAMAIS pendant l'entraînement ; les embeddings apprennent.
5. **Une addition, pas un remplacement :** nous AJOUTONS l'information de position aux embeddings ; les deux signaux coexistent.
6. **Le modèle apprend à les interpréter :** les embeddings s'ajustent pour fonctionner avec les signaux de position fixes.

**L'idée principale :** les encodages constituent un système de coordonnées fixe. Les embeddings apprennent à s'organiser de manière utile dans ce système. Les positions fournissent un référentiel stable !

### Les alternatives modernes (pour compléter)

Ce tutoriel utilise les encodages sinusoïdaux, mais d'autres approches existent.

- **Embeddings positionnels appris :** les positions sont apprises comme les embeddings des mots. L'approche est simple, mais une table fixe ne fournit pas directement les positions au-delà de sa taille.
- **RoPE (Rotary Position Embedding) :** la position est encodée par des rotations plutôt que par l'ajout d'un vecteur distinct. Les distances relatives deviennent plus faciles à représenter ; cette approche est notamment utilisée dans LLaMA et GPT-NeoX.
- **ALiBi (Attention with Linear Biases) :** l'information de position est ajoutée directement aux scores d'attention, une approche simple et efficace.

Chaque méthode a ses compromis. L'encodage sinusoïdal reste une bonne manière de comprendre les concepts fondamentaux !

---

### 🎓 Note avancée : l'élégance mathématique (pour les curieux)

**Cette section est FACULTATIVE !** Vous pouvez la passer si les empreintes positionnelles bornées vous suffisent. Pour comprendre la propriété mathématique sous-jacente, poursuivez !

**La question :** les encodages sinus/cosinus sont bornés, distinctifs et à plusieurs échelles. Mais une propriété supplémentaire les rend particulièrement élégants : **un décalage de position s'exprime par une transformation linéaire**.

**Qu'est-ce que cela signifie ?**

Pour un décalage fixe $k$, par exemple « trois positions plus loin », il existe une matrice $M_k$ telle que :

$$\text{PE}(\text{pos} + k) = M_k \times \text{PE}(\text{pos})$$

**En langage courant :** l'encodage de « position + 3 » se calcule à partir de celui de « position » par une multiplication matricielle, quelle que soit la position initiale !

**Pourquoi est-ce intéressant ?**

**Sans cette propriété :**

- Le modèle devrait apprendre ce que « +3 positions » signifie à la position 5, puis 50, puis 500…
- Il faudrait une règle pour chaque position de départ !
- Cela demanderait une infinité de régularités à apprendre.

**Avec cette propriété :**

- UNE matrice $M_3$ exprime « +3 positions ».
- Elle fonctionne aux positions 5, 50, 500, partout !
- Une même transformation peut traiter tous les cas.

**La raison mathématique : les identités trigonométriques.**

$$\sin(a + b) = \sin(a)\cos(b) + \cos(a)\sin(b)$$

$$\cos(a + b) = \cos(a)\cos(b) - \sin(a)\sin(b)$$

Elles s'écrivent comme des opérations matricielles ! Pour le décalage $k$ :

$$\begin{bmatrix} \sin(\text{pos} + k) \\ \cos(\text{pos} + k) \end{bmatrix} = \begin{bmatrix} \cos(k) & \sin(k) \\ -\sin(k) & \cos(k) \end{bmatrix} \begin{bmatrix} \sin(\text{pos}) \\ \cos(\text{pos}) \end{bmatrix}$$

La matrice

$$
\begin{bmatrix} \cos(k) & \sin(k) \\ -\sin(k) & \cos(k) \end{bmatrix}
$$

est la matrice de rotation correspondant au décalage $k$ dans la convention d'ordre sinus/cosinus !

**La conséquence pratique :** le mécanisme d'attention utilise des multiplications matricielles et peut exploiter une relation telle que « pour comparer le mot A à la position $p$ et le mot B à $p+3$, rechercher le motif X ». La relation « +3 » s'exprime sous une forme matricielle que le modèle peut apprendre à exploiter !

**Le sinus et le cosinus ne sont donc pas seulement une astuce :** ils donnent une construction mathématique fondée sur les identités trigonométriques. La source les oppose ici aux polynômes et aux exponentielles ; cette généralisation est à nuancer, comme l'expliquent les notes de traduction.

Cette propriété donne aux encodages sinusoïdaux leur élégance théorique. Elle motive leur choix dans l'article original « Attention Is All You Need », au-delà de leur amplitude bornée et de leur capacité à distinguer les positions.

**Pas d'inquiétude si cela semble avancé !** Il n'est pas nécessaire de maîtriser ces détails pour utiliser les Transformers. C'est un complément pour approfondir !

---

**Navigation dans le cours :** [Précédent : chapitre 3 — Embeddings (anglais)](https://github.com/lviaud/Building_a_Transformer/blob/main/chapter-03-embeddings.md) | [Suivant : chapitre 5 — Auto-attention multi-têtes (anglais)](https://github.com/lviaud/Building_a_Transformer/blob/main/chapter-05-multi-head-self-attention.md)

