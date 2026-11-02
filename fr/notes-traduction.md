# Notes de traduction et précisions techniques

Cette page distingue la traduction du chapitre des précisions apportées lors de sa préparation. Le texte original est celui de **Rimom Costa**, dans le dépôt [Building_a_Transformer](https://github.com/rimomcosta/Building_a_Transformer), au commit `d8e275a2c41426889e116bfa6d64090805b3bd9a`.

## Choix de traduction

- Les neuf parties du chapitre sont conservées, ainsi que les tableaux, formules et exemples numériques.
- Les phrases anglaises associées aux vecteurs restent en anglais pour conserver leurs positions. Par exemple, « I love pizza » comporte trois mots ; sa traduction française ne conserverait pas nécessairement ce découpage.
- Les légendes et explications sont traduites. Les identifiants symboliques du pseudo-code restent inchangés.
- Les valeurs arrondies proviennent de la source. Des égalités numériques sont parfois approximatives : lire « ≈ » lorsque les valeurs affichées sont arrondies.
- Les liens vers les chapitres 3 et 5 conduisent aux versions anglaises du fork, tant qu'ils ne sont pas traduits.

## Précisions sur le chapitre original

### Longueurs et indices de certains exemples

La phrase « I really love eating delicious homemade pizza with my best friends » contient **11 mots**, indexés de 0 à 10, bien que la source parle de 10 mots et divise par 10. Pour une normalisation par la longueur, on diviserait par 11 ; pour faire correspondre les deux extrémités à 0 et 1, on peut diviser par 10, soit le dernier indice. Ces conventions sont différentes.

Dans « I love delicious homemade pizza », « delicious » est à l'indice 2 et « homemade » à l'indice 3. Le bloc source associe la position normalisée 0.6 à « delicious », alors qu'elle correspond à « homemade » pour un diviseur de 5. Les valeurs sont conservées dans la traduction pour permettre la comparaison.

### Le sinus seul ne détermine pas un angle

Le sinus prend plusieurs fois une même valeur dans un cycle. Par exemple, $\sin(\pi/6)=\sin(5\pi/6)=1/2$. C'est le **couple** sinus/cosinus qui détermine l'angle modulo $2\pi$. La traduction le précise à l'endroit de l'affirmation ambiguë.

### La base 10 000 n'est pas le dernier diviseur

Pour une dimension paire $d$, le dernier indice de paire est $i=d/2-1$. Le dernier diviseur est donc $10000^{1-2/d}$, inférieur à 10 000. Avec $d=6$, il vaut environ 464.159 et sa période est d'environ $2\pi\times464.159\approx2916.4$ positions. La période de 62 800 positions évoquée dans le texte correspondrait à un diviseur de 10 000, et non au cas à six dimensions calculé ensuite.

### Gradients et précision numérique

Pour $x+p$ avec $p$ constant, la dérivée par rapport à $x$ est 1. Le seul fait d'ajouter une grande constante ne rend donc pas automatiquement les mises à jour de $x$ plus petites. L'exemple du taux d'apprentissage exprime une intuition d'échelle, sans démontrer à lui seul un ralentissement.

En float32, autour de 1000, le pas entre valeurs représentables vaut $2^{-14}\approx0.000061$. Une variation de 0.001 n'est donc pas systématiquement perdue. Les erreurs dépendent de l'amplitude et du format utilisé. La fonction [NumPy spacing](https://numpy.org/doc/stable/reference/generated/numpy.spacing.html) permet de mesurer cet espacement.

### Calculer une position ne garantit pas l'extrapolation

La formule sinusoïdale peut être évaluée à des positions absentes de l'entraînement. Cela ne garantit ni la qualité sur de longues séquences ni une distinction parfaite de toutes les positions dans une précision numérique finie. Les affirmations sur des longueurs « infinies » décrivent la formule, et non une garantie de performance du modèle.

### Quels paramètres sont entraînés ?

La comparaison du chapitre oppose les embeddings aux encodages sinusoïdaux fixes. Les autres paramètres du Transformer, notamment les projections d'attention et les couches de calcul, sont également entraînés. Dire que « seuls les embeddings apprennent » doit se lire dans le cadre de cette comparaison.

### RoPE et transformations relatives

RoPE applique les rotations aux vecteurs de requêtes et de clés de l'attention ; la description comme une simple rotation des embeddings de mots est un raccourci. Voir l'article [RoFormer](https://arxiv.org/abs/2104.09864).

La matrice de la note avancée correspond à la fréquence 1 et à l'ordre de coordonnées sinus/cosinus. Pour chaque autre paire, remplacer $k$ par $k/10000^{2i/d_{\text{model}}}$. La transformation complète est formée de ces blocs. Son existence ne signifie pas que le Transformer apprend explicitement une matrice nommée $M_k$.

Enfin, les sinus et cosinus ne sont pas les seules fonctions ayant une relation de décalage linéaire : les exponentielles et une base polynomiale adaptée en possèdent aussi. Leur intérêt ici réside notamment dans la combinaison de valeurs bornées et de plusieurs fréquences.

La section 3.5 de [Attention Is All You Need](https://arxiv.org/html/1706.03762v7#S3.SS5) présente la construction sinusoïdale et l'hypothèse selon laquelle ces relations facilitent l'apprentissage des positions relatives. Les précisions et calculs ci-dessus sont des notes de relecture, distinctes du cours original.
