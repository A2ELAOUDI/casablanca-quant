# Casablanca Quant Lab

**Un laboratoire de recherche quantitative pour la Bourse de Casablanca (CSE).**

## C'est quoi ce projet ?

La question de départ : *la bourse marocaine est-elle prédictible, et si oui, comment
en profiter intelligemment ?*

Les grands fonds quantitatifs ne « devinent » pas les cours. Ils font trois choses,
systématiquement :

1. **mesurer** si le marché contient de la structure exploitable (inefficiences) ;
2. **tester** des stratégies candidates dans des conditions brutalement réalistes
   (coûts de transaction, liquidité, aucune information du futur) ;
3. **refuser** de trader tout signal qui ne survit pas aux tests statistiques — parce
   que le pire ennemi d'un quant, c'est de se mentir avec un backtest trop beau.

Ce projet reproduit cette discipline pour la Bourse de Casablanca : un marché
frontière peu arbitré (donc potentiellement inefficient — c'est l'opportunité), mais
avec une liquidité fine, pas de vente à découvert, et des coûts élevés qui tuent les
stratégies trop actives (c'est la contrainte).

Concrètement, le lab prend les historiques de cours (exports Excel de la Bourse de
Casablanca), mesure l'inefficience du marché, backteste une bibliothèque de
stratégies, entraîne un moteur de prévision par apprentissage automatique, et
produit : un **top 10 d'actions par trimestre avec les raisons de chaque choix**,
des **prévisions de 1 mois à 1 an** avec leur incertitude, et un **portefeuille**
aux poids optimisés. Le tout piloté depuis un dashboard.

## Pourquoi la Bourse de Casablanca ? Un peu de théorie

L'**hypothèse d'efficience des marchés** (Fama, 1970) affirme que les prix intègrent
à tout instant l'ensemble de l'information publique disponible : si c'est vrai,
aucune stratégie basée sur cette information ne peut battre durablement le marché
après coûts — les cours suivent une marche aléatoire, et le passé ne dit rien sur
l'avenir. C'est le théorème fondamental contre lequel tout quant se bat.

Empiriquement, cette hypothèse est *approximativement* vraie sur les grands marchés
très arbitrés (New York, Londres, Tokyo) : des milliers de fonds systématiques y
traquent la moindre inefficience, et en la tradant, la font disparaître. Mais elle
est *fréquemment violée* sur les **marchés frontières**, et pour des raisons
structurelles précises :

- **faible couverture par les analystes** : l'information circule lentement, les
  prix mettent des jours ou des semaines à l'intégrer (source de momentum) ;
- **flux dominé par les particuliers et les institutionnels locaux** : biais
  comportementaux (sur-réaction, ancrage, effet de disposition) non arbitrés ;
- **absence d'arbitrageurs systématiques** : personne pour corriger les anomalies ;
- **contraintes structurelles** : pas de vente à découvert (impossible de parier à
  la baisse, donc les mauvaises nouvelles s'intègrent lentement), seuils de variation
  journaliers (les grands mouvements s'étalent sur plusieurs séances — de
  l'autocorrélation mécanique).

La Bourse de Casablanca (~75 valeurs, indice MASI) présente exactement ce profil.
C'est l'opportunité. La contrepartie : une liquidité fine (beaucoup de valeurs ne
traitent pas chaque séance) et des coûts de transaction élevés (~1 % l'aller simple,
courtage + spread) qui condamnent les stratégies à rotation rapide. L'enjeu
quantitatif du projet est donc double : **détecter l'inefficience** (étape 1) et
**vérifier qu'elle reste exploitable net de frictions** (étapes 2 et 3) — une
anomalie qui ne survit pas aux coûts n'est pas une opportunité, c'est un mirage.

## ⚠️ Ce n'est que le début — le projet a besoin de données

**État actuel : un seul titre chargé (Managem, ~740 séances).** Or toute la puissance
de l'approche est *cross-sectionnelle* : comparer les ~75 valeurs de la cote entre
elles à chaque date. Avec une action, le moteur tourne mais se limite à de l'analyse
de tendance ; avec toute la cote :

- l'audit dira **combien de titres rejettent la marche aléatoire** (la carte des
  inefficiences du marché marocain) ;
- le top 10 trimestriel aura un **vrai backtest** (panier vs univers, trimestre après
  trimestre) ;
- les modèles apprendront sur ~35 000 observations au lieu de ~700.

**Pour améliorer le projet : déposer les exports de cours des autres valeurs dans
`data/raw/`** (même format que les exports officiels, un fichier par valeur ou tout
dans un seul), puis relancer le pipeline.

## Installation et démarrage

```bash
pip install -e .

# 1. Déposer les fichiers Excel/CSV dans data/raw/
# 2. Pipeline complet
python -m casablanca_quant.cli all --capital 100000
# 3. Dashboard
python -m streamlit run app/streamlit_app.py

# Pas encore de données ? Marché synthétique de validation :
python -m casablanca_quant.cli demo
```

L'ingestion reconnaît automatiquement les exports officiels CSE (`Séance`, `Ticker`,
`Cours ajusté`, `Ouverture`, `+haut/+bas du jour`, `Nombre de titres échangés`...),
les formats long et large, les nombres français (`1 234,56`) et les multi-feuilles.
Le **cours ajusté** est préféré au cours brut : sans correction des dividendes et
splits, tous les rendements sont faux. Rendements logarithmiques :
$r_{i,t} = \ln(P_{i,t}/P_{i,t-1})$.

---

# Les modèles — l'idée, puis les maths

Chaque étape est expliquée en deux temps : **l'idée** (ce que ça fait et pourquoi),
puis **les maths** (comment c'est calculé exactement).

## Étape 1 — L'audit : le marché est-il prédictible ? (`audit.py`)

### Variance ratio de Lo-MacKinlay

**L'idée.** Si les cours suivent une marche aléatoire, le risque sur 5 jours vaut
exactement 5 fois le risque sur 1 jour. Si les tendances persistent (momentum), le
risque sur 5 jours est *plus grand* que ça ; si les cours sur-réagissent puis
reviennent (réversion), il est *plus petit*. On compare donc les deux — c'est le test
académique de référence de la marche aléatoire.

**Les maths.** $VR(q) = \dfrac{\mathrm{Var}(r_t + \dots + r_{t-q+1})}{q\,\mathrm{Var}(r_t)}$ ;
$VR>1$ momentum, $VR<1$ réversion. La statistique de test utilise la variance robuste
à l'hétéroscédasticité :

$$\hat\theta(q) = \sum_{k=1}^{q-1}\left[\frac{2(q-k)}{q}\right]^2 \hat\delta_k,
\qquad
\hat\delta_k = \frac{\sum_t (r_t-\bar r)^2 (r_{t-k}-\bar r)^2}{\left[\sum_t (r_t-\bar r)^2\right]^2 / n}$$

indispensable car la volatilité des actions arrive en grappes (effet GARCH) : sans
cette correction, on rejetterait la marche aléatoire à tort.

### Ljung-Box

**L'idée.** Le rendement d'aujourd'hui dépend-il de ceux des 10 derniers jours ?

**Les maths.** $Q = n(n+2)\sum_{k=1}^{10} \frac{\hat\rho_k^2}{n-k} \sim \chi^2_{10}$
sous l'hypothèse d'indépendance.

### Spread momentum

**L'idée.** Test direct, sans machinerie : à chaque date, on classe les actions par
leur performance des 6 derniers mois, et on regarde si les gagnantes battent les
perdantes le mois suivant. Si oui régulièrement, il y a du momentum à récolter.

**Les maths.** Signal $\ln(P_{t-21}/P_{t-126})$ (on saute le dernier mois qui
sur-réagit), spread = rendement 21 j du quintile haut moins le quintile bas, t-stat
de Student sur des périodes non chevauchantes.

## Étape 2 — Les stratégies candidates (`signals.py`, `backtest.py`)

### La bibliothèque de facteurs

**L'idée.** Dix stratégies issues de la littérature, choisies pour leur pertinence en
marché frontière dominé par les particuliers. Chacune donne un score à chaque action
chaque jour, en n'utilisant que le passé.

| Facteur | L'idée | La définition | Référence |
|---|---|---|---|
| `momentum_126_21` | acheter ce qui monte depuis 6 mois | $\ln(P_{t-21}/P_{t-126})$ | Jegadeesh & Titman 1993 |
| `reversal_5d` | acheter ce qui vient de baisser (sur-réaction) | $-\sum_{s=t-4}^{t} r_s$ | Jegadeesh 1990 |
| `low_volatility` | les valeurs calmes rapportent plus par unité de risque | $-\hat\sigma_{63}$ | Ang et al. 2006 |
| `high_52w` | l'ancrage : proche du plus-haut annuel, ça continue | $P_t / \max_{[t-252,t]} P_s$ | George & Hwang 2004 |
| `amihud_illiquidity` | l'illiquidité paie une prime de rendement | $\ln(1 + \overline{\|r\|/\mathrm{Vol}^{MAD}} \cdot 10^6)$ | Amihud 2002 |
| `abnormal_volume` | un volume inhabituel précède le mouvement | $\ln(\overline{V}_5 / \overline{V}_{63})$ | Gervais et al. 2001 |
| `trend_ma_20_100` | suivi de tendance par moyennes mobiles | $MA_{20}/MA_{100} - 1$ | Moskowitz et al. 2012 |
| `composite` | la moyenne des signaux, standardisés | moyenne des z-scores | — |

### Le moteur de backtest

**L'idée.** Rejouer l'histoire comme si on avait tradé : chaque semaine, acheter les
20 % d'actions les mieux notées parmi les suffisamment liquides, en payant les frais
à chaque changement. Deux règles anti-triche absolues : la décision du jour t est
exécutée au cours de **demain** (aucune information du futur), et le portefeuille est
**long-only** (pas de vente à découvert à la CSE).

**Les maths.** Rendement net du portefeuille :

$$r_{p,t} = \mathbf{w}_{t-1}^\top \mathbf{r}_t \;-\; c \cdot \|\mathbf{w}_{t-1} - \mathbf{w}_{t-2}\|_1$$

avec $c = 100$ pb par côté (courtage marocain + spread). Éligibilité : ≥ 60 % de
séances traitées sur 63 jours et volume médian ≥ 50 000 MAD.

## Étape 3 — Les quatre portes statistiques

**L'idée.** Quand on essaie 10 stratégies, la meilleure a *forcément* l'air bonne —
même sur un marché purement aléatoire. C'est le piège n°1 de la finance quantitative
(le « data-snooping »). Avant de déployer quoi que ce soit, le système exige donc
quatre preuves indépendantes ; au moindre échec, verdict : rester sur le portefeuille
de base. **Un système qui ne sait pas dire non perd de l'argent.**

**Les maths.**

1. **Sharpe net positif** : $\widehat{SR} = \hat\mu/\hat\sigma \cdot \sqrt{252} > 0$ après coûts.
2. **Surperformance bootstrap** : IC à 95 % (2 000 rééchantillonnages) sur la moyenne
   des rendements actifs vs le baseline equal-weight ; on exige $CI_{5\%} > 0$.
3. **Reality Check de White (2000)** : statistique $V = \max_k \sqrt{n}\,\bar f_k$
   ($\bar f_k$ = surperformance moyenne du facteur $k$), comparée à sa distribution
   sous $H_0$ par bootstrap par blocs mobiles recentré. Répond littéralement à :
   *« après avoir essayé K stratégies, quelle est la probabilité que la meilleure
   paraisse aussi bonne par pur hasard ? »*
4. **Sharpe déflaté** (Bailey & López de Prado 2014) :

$$DSR = \Phi\!\left(\frac{\widehat{SR} - SR^*}{\sqrt{\frac{1 - \hat\gamma_3 \widehat{SR} + \frac{\hat\gamma_4 - 1}{4}\widehat{SR}^2}{n-1}}}\right)$$

   où $SR^*$ est le Sharpe maximal attendu de $K$ essais sur du bruit pur
   (approximation de Gumbel), et $\hat\gamma_3, \hat\gamma_4$ pénalisent la
   non-normalité des rendements. Seuil : $DSR > 0.90$.

Validation sur marché synthétique : le système **refuse correctement de déployer**
— les coûts de 100 pb détruisent les stratégies à rotation rapide, et le Reality
Check rejette le reste. C'est le comportement voulu.

## Étape 4 — Le moteur alpha : prévisions par apprentissage (`alpha.py`)

### Le problème d'apprentissage

**L'idée.** Prédire le rendement de chaque action à 1, 3, 6 et 12 mois à partir de 12
caractéristiques calculées uniquement avec le passé (momentum, volatilité, liquidité,
état du marché...). Deux modèles aux erreurs différentes votent : une régression
linéaire régularisée (relations simples) et un gradient boosting (interactions,
non-linéarités).

**Les maths.** Cible $y_{i,t}^{(h)} = \ln(P_{i,t+h}/P_{i,t})$, $h \in \{21, 63, 126, 252\}$.
Prédicteur : $\hat y = \tfrac12\,\mathrm{Ridge}_{\alpha=10}(\mathbf{x}) + \tfrac12\,\mathrm{HGB}(\mathbf{x})$,
features standardisées, boosting profondeur 3 / 120 itérations / L2.

### La validation purgée — le point le plus important du projet

**L'idée.** Les cibles se chevauchent : le rendement « à 3 mois » du 1er mars et
celui du 2 mars partagent 62 jours sur 63. Une validation croisée classique met ces
deux points l'un en train, l'autre en test → le modèle « connaît » déjà la réponse →
score gonflé → backtest illusoire. C'est LA raison pour laquelle 90 % des modèles de
prédiction boursière amateurs semblent marcher et perdent en réel.

**Les maths.** Protocole walk-forward avec purge et embargo (López de Prado 2018) : à
chaque date de réentraînement $T$ (cadence 42 j), le train ne contient que les
observations dont la fenêtre de cible se termine avant $T$ ; la prédiction porte sur
$[T, T+42)$. Aucun chevauchement train/test, jamais.

### Le rétrécissement par compétence prouvée

**L'idée.** Le modèle mesure sa propre compétence hors échantillon. S'il n'a rien
prouvé, ses prévisions collent au rendement moyen du marché — et l'affichent. Pas de
fausse confiance : c'est la différence entre un outil de décision et un vendeur de rêve.

**Les maths.** Compétence = information coefficient $IC$ (corrélation de rang de
Spearman prédiction/réalisé, par date en cross-section, t-stat sur la série).
Prévision publiée :

$$\hat y_{\mathrm{final}} = \mu_M + \lambda\,(\hat y_{\mathrm{modèle}} - \mu_M),
\qquad \lambda = \mathrm{clip}(5 \cdot IC_{OOS},\; 0,\; 0.5)$$

$IC \le 0 \Rightarrow \lambda = 0$. Les fourchettes 5 %/95 % viennent des quantiles
des résidus hors échantillon (aucune hypothèse gaussienne).

### Régimes de marché et évaluation trimestrielle

**L'idée.** Le marché alterne entre périodes calmes et périodes de stress où les
corrélations montent et le momentum casse — un mélange gaussien à 2 états le détecte
et l'état courant est affiché à côté de chaque recommandation. Et pour coller à un
usage réel « j'investis chaque trimestre dans 10 valeurs », le top 10 est backtesté
exactement à cette cadence.

**Les maths.** GMM à 2 composantes sur les rendements du marché, état = moyenne des
probabilités a posteriori sur 21 j. Backtest : toutes les 63 séances hors
échantillon, excès du panier top-10 vs univers, t-stat sur la série des excès.

## Étape 5 — Le portefeuille : Hierarchical Risk Parity (`portfolio.py`)

**L'idée.** Une fois les 10 valeurs choisies, combien mettre sur chacune ?
L'optimisation classique (Markowitz) exige d'inverser une matrice de covariance —
numériquement explosive avec peu d'historique : elle concentre tout sur 2-3 titres
pour de mauvaises raisons. HRP fait plus robuste : regrouper les actions qui se
ressemblent en familles (clustering), puis répartir le budget de risque entre
familles, les plus stables recevant plus.

**Les maths.**

1. Covariance **Ledoit-Wolf (2004)** : $\hat\Sigma = \delta F + (1-\delta) S$,
   rétrécissement optimal (MSE) de la covariance empirique $S$ vers une cible
   structurée $F$, intensité $\delta$ estimée des données.
2. **HRP (López de Prado 2016)** : distance $d_{ij} = \sqrt{\tfrac12(1-\rho_{ij})}$ →
   clustering hiérarchique (single linkage) → quasi-diagonalisation (réordonnancement
   selon le dendrogramme) → **bisection récursive** : à chaque scission, budget
   réparti en proportion inverse de la variance des sous-clusters
   ($w \propto 1/\sigma^2$, variance de cluster au portefeuille inverse-variance interne).

Garde-fous : 12 % max par ligne, position ≤ 3 jours de volume médian (contrainte
d'impact), repli inverse-volatilité si HRP indisponible. Si aucun facteur n'a passé
les portes : equal-weight de l'univers liquide.

---

## Le dashboard

`python -m streamlit run app/streamlit_app.py` — six onglets : **Marché** (cours
ajusté, MM50, volumes, performances), **Top 10** (le panier du trimestre et pourquoi
chaque valeur y est), **Prévisions** (1M→12M avec fourchettes et poids de confiance),
**Modèles** (Sharpe par facteur, portes, IC par horizon), **Audit** (inefficience du
marché), **Portefeuille** (allocation HRP, export CSV). Bouton **« Tout analyser »**
= pipeline complet en un clic.

## Structure du code

```
src/casablanca_quant/
├── ingest.py      # lecture Excel/CSV flexible + générateur synthétique
├── audit.py       # tests d'efficience du marché
├── signals.py     # bibliothèque de facteurs
├── backtest.py    # moteur walk-forward + Reality Check + Sharpe déflaté
├── alpha.py       # ensemble ML purgé, prévisions, régimes, top 10
├── portfolio.py   # HRP + Ledoit-Wolf
└── cli.py         # ligne de commande
app/streamlit_app.py   # dashboard
tests/                 # pytest : lookahead, parsing, coûts, HRP...
```

## Feuille de route

- [x] Ingestion des exports officiels CSE + générateur synthétique
- [x] Audit d'efficience, 10 facteurs, 4 portes statistiques
- [x] Moteur alpha multi-horizons + régimes + top 10 trimestriel
- [x] Portefeuille HRP + dashboard
- [ ] **Charger toute la cote (~75 valeurs)** ← priorité absolue
- [ ] Historique 5-10 ans (les horizons 6M/12M manquent d'échantillons)
- [ ] Features fondamentales (PER, dividende, flottant, secteur)
- [ ] Indice MASI comme benchmark explicite (au lieu du proxy equal-weight)
- [ ] Journal de production : recommandations horodatées vs réalisé

## Références

- Lo & MacKinlay (1988). *Stock Market Prices Do Not Follow Random Walks*. RFS.
- Jegadeesh & Titman (1993). *Returns to Buying Winners and Selling Losers*. JF.
- Amihud (2002). *Illiquidity and Stock Returns*. JFM.
- White (2000). *A Reality Check for Data Snooping*. Econometrica.
- Ledoit & Wolf (2004). *A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices*. JMVA.
- Bailey & López de Prado (2014). *The Deflated Sharpe Ratio*. JPM.
- López de Prado (2016). *Building Diversified Portfolios that Outperform Out-of-Sample*. JPM.
- López de Prado (2018). *Advances in Financial Machine Learning*. Wiley.

## Avertissement

Outil de recherche et d'aide à la décision — **pas un conseil en investissement**.
Les performances passées ou simulées ne préjugent pas des performances futures.
Investir en actions comporte un risque de perte en capital.
