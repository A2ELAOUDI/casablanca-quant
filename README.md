# Casablanca Quant Lab

**Un laboratoire de recherche quantitative pour la Bourse de Casablanca.**

## C'est quoi ce projet ?

La question de départ est simple : *la bourse marocaine est-elle prédictible, et si
oui, comment en profiter intelligemment ?*

Les grands fonds quantitatifs (Renaissance Technologies, Two Sigma, Citadel...) ne
« devinent » pas les cours. Ils font trois choses, systématiquement :

1. **mesurer** si le marché contient de la structure exploitable (inefficiences) ;
2. **tester** des stratégies candidates dans des conditions brutalement réalistes
   (coûts, liquidité, aucune information du futur) ;
3. **refuser** de trader tout signal qui ne survit pas aux tests statistiques —
   parce que le pire ennemi d'un quant, c'est de se mentir à soi-même avec un
   backtest trop beau.

Ce projet reproduit cette discipline, adaptée à la réalité de la Bourse de
Casablanca : marché frontière peu arbitré (donc potentiellement inefficient — c'est
une opportunité), mais liquidité fine, pas de vente à découvert, et des coûts de
transaction élevés qui tuent les stratégies trop actives.

Concrètement, le lab prend des historiques de cours (fichiers Excel de la Bourse de
Casablanca), les nettoie, mesure l'inefficience du marché, backteste une bibliothèque
de stratégies, entraîne un moteur de prévision par apprentissage automatique, propose
un **top 10 d'actions par trimestre avec les raisons de chaque choix**, des
**prévisions de 1 mois à 1 an** avec leur marge d'incertitude, et construit un
**portefeuille** aux poids optimisés. Le tout piloté depuis un dashboard.

## ⚠️ Ce n'est que le début — le projet a besoin de données

**L'état actuel : un seul titre est chargé (Managem).** Or toute la puissance de
l'approche est *cross-sectionnelle* : comparer ~75 valeurs de la cote entre elles à
chaque date pour repérer lesquelles sont en avance ou en retard. Avec une seule
action, le moteur tourne mais se limite à de l'analyse de tendance ; avec toute la
cote, il peut vraiment travailler :

- l'audit dira **combien de titres rejettent la marche aléatoire** (la carte des
  inefficiences du marché marocain) ;
- le classement top 10 trimestriel aura un **vrai backtest** (paniers comparés à
  l'univers, trimestre après trimestre) ;
- les modèles apprendront sur ~35 000 observations (75 titres × 500 séances) au lieu
  de 700.

**Comment contribuer / améliorer : déposer les exports de cours des autres valeurs de
la cote dans `data/raw/`** (même format que les exports officiels de la Bourse de
Casablanca, un fichier par valeur ou tout dans un seul fichier), puis relancer le
pipeline. Plus l'historique est long et large, plus les conclusions sont fiables.

## Installation et démarrage

```bash
pip install -e .

# 1. Déposer les fichiers Excel/CSV de cours dans data/raw/
# 2. Pipeline complet : ingestion → audit → backtest → alpha → portefeuille
python -m casablanca_quant.cli all --capital 100000
# 3. Dashboard
python -m streamlit run app/streamlit_app.py
```

Pas encore de données ? `python -m casablanca_quant.cli demo` génère un marché
synthétique (échanges clairsemés, seuils de variation, momentum injecté) pour tester
toute la chaîne.

## Les données acceptées

L'ingestion reconnaît automatiquement les exports officiels de la Bourse de
Casablanca : colonnes `Séance`, `Ticker`, `Cours ajusté` (préféré au cours brut car
corrigé des dividendes et splits), `Ouverture`, `+haut/+bas du jour`, `Nombre de
titres échangés`... Elle gère aussi le format « large » (une colonne par valeur), les
nombres français (`1 234,56`), les dates `jj/mm/aaaa`, plusieurs fichiers et
plusieurs feuilles Excel. Sortie : `data/processed/prices.csv`.

## Les modèles, expliqués simplement

### Étape 1 — L'audit : le marché est-il prédictible ? (`audit`)

| Test | Question posée |
|---|---|
| **Variance ratio (Lo-MacKinlay)** | Les cours suivent-ils une marche aléatoire ? VR > 1 = les tendances persistent (momentum), VR < 1 = les cours sur-réagissent puis reviennent (mean reversion) |
| **Ljung-Box** | Les rendements d'aujourd'hui dépendent-ils de ceux d'hier ? |
| **Spread momentum** | Les gagnants des 6 derniers mois battent-ils les perdants le mois suivant ? |
| **Carte de liquidité** | Sur quelles valeurs peut-on réellement trader sans bouger le cours ? |

### Étape 2 — Les stratégies candidates (`backtest`)

Dix familles issues de la littérature académique et de la pratique des fonds, choisies
pour leur pertinence en marché frontière :

| Facteur | Idée |
|---|---|
| `momentum_126_21` / `momentum_63_10` | Acheter ce qui monte depuis 3-6 mois (en sautant le dernier mois, qui sur-réagit) |
| `reversal_5d` / `reversal_21d` | Acheter ce qui vient de baisser : la sur-réaction des particuliers se corrige |
| `low_volatility` | Les valeurs calmes rapportent plus par unité de risque (anomalie documentée mondialement) |
| `high_52w` | Les valeurs proches de leur plus-haut 52 semaines continuent de dériver vers le haut (ancrage psychologique) |
| `amihud_illiquidity` | Les valeurs illiquides paient une prime de rendement pour compenser |
| `abnormal_volume` | Un volume inhabituel signale l'arrivée d'information avant son plein effet sur le prix |
| `trend_ma_20_100` | Suivi de tendance classique par croisement de moyennes mobiles |
| `composite` | Moyenne des signaux ci-dessus, standardisés |

Le moteur de backtest est volontairement impitoyable : signal du jour t exécuté à la
clôture de **t+1** (aucune information du futur), **long-only** (pas de vente à
découvert à la CSE), filtre de liquidité, et **100 points de base de coûts par côté**
prélevés sur chaque rotation du portefeuille. Une stratégie qui a l'air géniale sans
coûts et qui meurt avec — c'est le backtest qui a raison.

### Étape 3 — Les quatre portes statistiques

Un facteur n'est déployé en production que s'il passe **les quatre** :

1. Sharpe net de coûts positif ;
2. bat le portefeuille equal-weight (le « marché ») avec un intervalle de confiance
   bootstrap à 95 % ;
3. **Reality Check de White** : quand on essaie 10 stratégies, la meilleure a l'air
   bonne par pur hasard — ce test bootstrap corrige exactement ce biais ;
4. **Sharpe déflaté** (Bailey & López de Prado) > 0.90 : la probabilité que le Sharpe
   observé soit réel compte tenu du nombre d'essais effectués.

Sinon, verdict : rester sur le baseline. Un système qui ne sait pas dire « non » perd
de l'argent.

### Étape 4 — Le moteur alpha : prévisions par apprentissage automatique (`alpha`)

- 12 **features point-in-time** par (date, valeur) : momentum multi-horizons,
  reversal, volatilité, distance au plus-haut 52 semaines, illiquidité, chocs de
  volume, bêta, état du marché ;
- **ensemble de modèles** (régression Ridge + Gradient Boosting) — deux familles
  d'erreurs différentes, moyennées ;
- **validation walk-forward purgée avec embargo** (López de Prado) : l'entraînement ne
  voit jamais de données dont la cible chevauche la période de test. C'est LA
  technique qui sépare un backtest honnête d'un backtest illusoire ;
- **prévisions rétrécies par la compétence prouvée** : le modèle mesure son propre
  skill hors échantillon (information coefficient) ; sans skill démontré, ses
  prévisions collent au rendement moyen du marché **et l'affichent**. Pas de fausse
  confiance ;
- **horizons 1M / 3M / 6M / 12M** avec fourchettes pire cas / meilleur cas ;
- **détection de régime** (calme vs stress) par mélange gaussien ;
- **top 10 trimestriel** avec, pour chaque valeur, les raisons de sa sélection.

### Étape 5 — Le portefeuille (`portfolio`)

**Hierarchical Risk Parity** (López de Prado) sur covariance **Ledoit-Wolf** : la
méthode d'allocation robuste utilisée quand l'optimisation classique de Markowitz
explose sur de petits échantillons. Garde-fous : 12 % maximum par ligne, 3 jours de
volume médian maximum par position. Sortie : poids, montants en MAD, nombre d'actions.

## Le dashboard

Six onglets : **Marché** (cours, moyennes mobiles, volumes, performances), **Top 10**
(le panier du trimestre et pourquoi chaque valeur y est), **Prévisions** (1 mois →
1 an par valeur avec incertitude), **Modèles** (Sharpe par facteur, portes de
déploiement, skill du moteur alpha), **Audit** (inefficience du marché),
**Portefeuille** (allocation, export CSV). Bouton **« Tout analyser »** = pipeline
complet en un clic.

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
tests/                 # pytest (lookahead, parsing, coûts, HRP...)
```

## Feuille de route

- [x] Ingestion des exports officiels CSE + générateur synthétique
- [x] Audit d'efficience, 10 facteurs, 4 portes statistiques
- [x] Moteur alpha multi-horizons + régimes + top 10 trimestriel
- [x] Portefeuille HRP + dashboard
- [ ] **Charger toute la cote (~75 valeurs)** ← la priorité absolue
- [ ] Historique plus profond (5-10 ans) pour les horizons 6M/12M
- [ ] Données fondamentales (PER, dividendes, flottant) comme features
- [ ] Suivi en production : journal des recommandations vs réalisé

## Avertissement

Outil de recherche et d'aide à la décision — **pas un conseil en investissement**.
Les performances passées ou simulées ne préjugent pas des performances futures.
Investir en actions comporte un risque de perte en capital.
