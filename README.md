# Casablanca Quant Lab

**Laboratoire de recherche quantitative pour la Bourse de Casablanca (CSE)** —
ingestion des historiques de cours, tests d'efficience de marché, bibliothèque de
facteurs backtestés sous frictions réalistes, moteur de prévision par ensemble
d'apprentissage avec validation purgée, allocation Hierarchical Risk Parity, et
dashboard de pilotage.

Le principe directeur est celui des fonds quantitatifs systématiques : **aucun signal
n'est déployé s'il ne survit pas à une batterie de tests conçus pour le détruire**
(coûts de transaction, correction du data-snooping, déflation du Sharpe). Le système
préfère dire « pas de signal » que d'halluciner de l'alpha.

---

## 1. Motivation

L'hypothèse d'efficience des marchés (Fama, 1970) prédit qu'aucune stratégie basée
sur l'information publique ne bat durablement le marché après coûts. Empiriquement,
cette hypothèse est *approximativement* vraie sur les grands marchés très arbitrés,
et *fréquemment violée* sur les marchés frontières : faible couverture par les
analystes, flux dominé par des investisseurs particuliers et institutionnels locaux,
absence d'arbitrageurs systématiques, contraintes structurelles (pas de vente à
découvert, seuils de variation journaliers).

La Bourse de Casablanca (~75 valeurs cotées, MASI) présente exactement ce profil.
La contrepartie : liquidité fine (de nombreuses valeurs ne traitent pas chaque
séance) et coûts de transaction élevés (~1 % aller simple courtage + spread), qui
condamnent toute stratégie à rotation rapide. L'enjeu quantitatif est donc double :
**détecter l'inefficience** et **vérifier qu'elle est exploitable net de frictions**.

## 2. Architecture du pipeline

```
Excel/CSV (exports CSE)
   └─ ingest.py    → panel long normalisé (date, ticker, OHLCV, cours ajusté)
        └─ audit.py     → le marché dévie-t-il de la marche aléatoire ?
        └─ signals.py   → scores factoriels S_k(t, i), strictement point-in-time
             └─ backtest.py → rendements nets walk-forward + inférence statistique
        └─ alpha.py     → E[r_{t→t+h}] par ensemble ML, validation purgée
             └─ portfolio.py → poids HRP sous contraintes de liquidité
                  └─ app/streamlit_app.py → dashboard
```

## 3. Données

L'ingestion reconnaît automatiquement les exports officiels de la Bourse de
Casablanca (`Séance`, `Ticker`, `Cours ajusté`, `Ouverture`, `+haut/+bas du jour`,
`Nombre de titres échangés`, ...), les formats long et large, les nombres français
et les multi-feuilles Excel. Le **cours ajusté** (corrigé des dividendes et
opérations sur titres) est systématiquement préféré au cours brut : les rendements
calculés sur cours non ajustés sont biaisés à chaque détachement.

Les rendements sont logarithmiques : $r_{i,t} = \ln(P_{i,t}/P_{i,t-1})$, avec
report avant (forward-fill) plafonné à 10 séances pour les valeurs non traitées, et
un indicateur `traded` pour distinguer vrai rendement nul et absence d'échange.

**État actuel : un seul titre chargé (Managem).** Toute l'approche cross-sectionnelle
(section 6) nécessite l'univers complet — voir la feuille de route.

## 4. Audit d'efficience (`audit.py`)

### 4.1 Variance ratio de Lo-MacKinlay (1988)

Sous marche aléatoire, la variance des rendements agrégés sur $q$ périodes croît
linéairement : $\mathrm{Var}(r_t^{(q)}) = q\,\mathrm{Var}(r_t)$. Le test mesure la
déviation :

$$VR(q) = \frac{\mathrm{Var}(r_t + \dots + r_{t-q+1})}{q\,\mathrm{Var}(r_t)}$$

$VR > 1$ signale de l'autocorrélation positive (momentum), $VR < 1$ de la réversion.
L'implémentation utilise la statistique **robuste à l'hétéroscédasticité** $z^*(q)$,
où la variance de $VR$ est estimée par

$$\hat\theta(q) = \sum_{k=1}^{q-1}\left[\frac{2(q-k)}{q}\right]^2 \hat\delta_k,
\qquad
\hat\delta_k = \frac{\sum_t (r_t-\bar r)^2 (r_{t-k}-\bar r)^2}{\left[\sum_t (r_t-\bar r)^2\right]^2 / n}$$

ce qui rend le test valide sous volatilité stochastique (clustering GARCH), omniprésente
sur actions.

### 4.2 Ljung-Box

$Q = n(n+2)\sum_{k=1}^{h} \frac{\hat\rho_k^2}{n-k} \sim \chi^2_h$ sous l'hypothèse
nulle d'absence d'autocorrélation jusqu'au retard $h$ (ici $h=10$).

### 4.3 Spread momentum cross-sectionnel

Test direct de Jegadeesh-Titman : à chaque date de formation, classement par
$\ln(P_{t-21}/P_{t-126})$, puis rendement moyen à 21 jours du quintile supérieur
moins le quintile inférieur, sur des périodes **non chevauchantes** (t-stat de
Student sur la série des spreads). C'est la matière première du facteur momentum,
mesurée sans machinerie de backtest.

## 5. Bibliothèque factorielle et backtest (`signals.py`, `backtest.py`)

### 5.1 Définition des facteurs

Chaque facteur est une fonction $S_k : (t, i) \mapsto \mathbb{R}$ n'utilisant que
l'information $\mathcal{F}_t$ :

| Facteur | Définition | Référence |
|---|---|---|
| `momentum_126_21` | $\ln(P_{t-21}/P_{t-126})$ | Jegadeesh & Titman (1993) |
| `reversal_5d` | $-\sum_{s=t-4}^{t} r_s$ | Jegadeesh (1990) |
| `low_volatility` | $-\hat\sigma_{63}$ (écart-type roulant) | Ang et al. (2006) |
| `high_52w` | $P_t / \max_{s \in [t-252, t]} P_s$ | George & Hwang (2004) |
| `amihud_illiquidity` | $\ln\!\big(1 + \overline{|r|/\mathrm{Vol}^{MAD}}_{63} \cdot 10^6\big)$ | Amihud (2002) |
| `abnormal_volume` | $\ln(\overline{V}_5 / \overline{V}_{63})$ | Gervais et al. (2001) |
| `trend_ma_20_100` | $MA_{20}/MA_{100} - 1$ | Moskowitz et al. (2012) |
| `composite` | moyenne des z-scores cross-sectionnels des facteurs cœurs | — |

### 5.2 Moteur d'exécution

Portefeuille long-only (la VAD n'existe pas à la CSE) : à chaque rebalancement
(défaut : 5 séances), poids égaux sur le quantile supérieur (20 %) des titres
**éligibles** (≥ 60 % de séances traitées sur 63 jours, volume médian ≥ 50 000 MAD).
Le rendement net s'écrit :

$$r_{p,t} = \mathbf{w}_{t-1}^\top \mathbf{r}_t \;-\; c \cdot \|\mathbf{w}_{t-1} - \mathbf{w}_{t-2}\|_1$$

avec $c = 100$ pb par côté (paramétrable). Le décalage $\mathbf{w}_{t-1}$ impose
l'exécution à la clôture de $t\!+\!1$ : **aucune information contemporaine n'entre
dans la décision**.

### 5.3 Inférence : les quatre portes

Un facteur n'est éligible à la production que s'il franchit les quatre tests.

1. **Sharpe net positif** : $\widehat{SR} = \hat\mu / \hat\sigma \cdot \sqrt{252} > 0$.

2. **Surperformance bootstrap** : IC à 95 % (2 000 rééchantillonnages) sur la moyenne
   des rendements actifs $r_{p,t} - r_{bench,t}$ contre le baseline equal-weight
   soumis aux mêmes frictions ; on exige $CI_{5\%} > 0$.

3. **Reality Check de White (2000)** : soit $\bar f_k$ la moyenne des rendements
   actifs du facteur $k$. La statistique $V = \max_k \sqrt{n}\,\bar f_k$ est comparée
   à sa distribution sous $H_0$ obtenue par bootstrap par blocs mobiles
   (blocs de 10 jours, recentrage $\bar f_k^* - \bar f_k$). Ce test répond à la
   question que le p-value individuel ignore : *« après avoir essayé $K$ stratégies,
   quelle est la probabilité que la meilleure paraisse aussi bonne par pur hasard ? »*

4. **Deflated Sharpe Ratio** (Bailey & López de Prado, 2014) :

$$DSR = \Phi\!\left(\frac{\widehat{SR} - SR^*}{\sqrt{\frac{1 - \hat\gamma_3 \widehat{SR} + \frac{\hat\gamma_4 - 1}{4}\widehat{SR}^2}{n-1}}}\right)$$

   où $SR^*$ est l'espérance du maximum de $K$ Sharpe sous $H_0$ (approximation
   Gumbel via $E[\max] \approx (1-\gamma)\Phi^{-1}(1-\tfrac1K) + \gamma\,\Phi^{-1}(1-\tfrac1{Ke})$,
   $\gamma$ constante d'Euler-Mascheroni), et $\hat\gamma_3, \hat\gamma_4$ les
   skewness et kurtosis empiriques — la non-normalité des rendements gonfle la
   variance du Sharpe estimé et doit être pénalisée. Seuil retenu : $DSR > 0.90$.

Échec à une porte ⇒ verdict : **baseline equal-weight**. Sur données synthétiques de
validation, le système refuse correctement de déployer (les coûts de 100 pb
détruisent les facteurs à rotation rapide — comportement attendu et souhaité).

## 6. Moteur alpha (`alpha.py`)

### 6.1 Problème d'apprentissage

Cible : $y_{i,t}^{(h)} = \ln(P_{i,t+h}/P_{i,t})$ pour $h \in \{21, 63, 126, 252\}$
jours. Features $\mathbf{x}_{i,t} \in \mathbb{R}^{12}$ : momentum (3 horizons),
réversion 5 j, volatilités (niveau et ratio court/long), distance au plus-haut 52
semaines, illiquidité d'Amihud, choc de volume, bêta roulant 63 j vs marché
equal-weight, momentum et volatilité du marché (features de régime).

Prédicteur : moyenne d'un **Ridge** ($\alpha=10$, features standardisées) et d'un
**HistGradientBoosting** (profondeur 3, 120 itérations, lr 0.05, régularisation L2).
La combinaison linéaire + arbres couvre les composantes additives et les
interactions/non-linéarités, avec des profils d'erreur décorrélés.

### 6.2 Validation walk-forward purgée avec embargo (López de Prado, 2018)

Les cibles à horizon $h$ se **chevauchent** sur $h$ jours : une validation croisée
naïve fait fuiter l'information du test vers le train et produit des IC massivement
gonflés. Protocole appliqué : à chaque date de réentraînement $T$ (cadence 42 j),
le train est restreint aux observations dont la **fenêtre de cible est intégralement
antérieure** à $T$ (purge + embargo), la prédiction porte sur $[T, T+42)$.

### 6.3 Compétence mesurée et rétrécissement des prévisions

Compétence hors échantillon = **information coefficient** : corrélation de rang de
Spearman entre prédiction et réalisé, calculée par date en cross-section (t-stat sur
la série des IC), en repli sur l'IC poolé temporel avec taille effective
$n_{\mathrm{eff}} \approx n/21$ si l'univers est trop étroit.

La prévision publiée est **rétrécie vers le marché** proportionnellement au skill
prouvé :

$$\hat y_{\mathrm{final}} = \mu_M + \lambda\,(\hat y_{\mathrm{modèle}} - \mu_M),
\qquad \lambda = \mathrm{clip}(5 \cdot IC_{OOS},\; 0,\; 0.5)$$

où $\mu_M$ est le rendement moyen du marché à l'horizon considéré. Conséquence
structurelle : $IC \le 0 \Rightarrow \lambda = 0$ — un horizon sans skill démontré
publie le rendement de marché, et l'affiche. Les intervalles (5 %/95 %) proviennent
des quantiles des **résidus hors échantillon**, pas d'une hypothèse gaussienne.

### 6.4 Régimes de marché

Mélange gaussien à 2 composantes sur les rendements quotidiens du marché ; l'état
courant (CALME/STRESS) est la moyenne des probabilités a posteriori sur 21 jours.
En régime de stress, corrélations et queues s'épaississent et le momentum se dégrade
— information affichée à côté de toute recommandation.

### 6.5 Évaluation à la cadence d'investissement réelle

Backtest trimestriel du top 10 : toutes les 63 séances hors échantillon, panier des
10 meilleures prévisions vs moyenne de l'univers ; t-stat sur la série des excès
trimestriels. C'est la métrique qui correspond à une utilisation « j'investis chaque
trimestre dans 10 valeurs ».

## 7. Allocation : Hierarchical Risk Parity (`portfolio.py`)

Markowitz requiert l'inversion de la covariance — numériquement instable quand
$T/N$ est faible. Deux remèdes combinés :

1. **Covariance Ledoit-Wolf (2004)** : $\hat\Sigma = \delta F + (1-\delta) S$,
   rétrécissement optimal (minimisation MSE) de la covariance empirique $S$ vers une
   cible structurée $F$, avec intensité $\delta$ estimée des données.

2. **HRP (López de Prado, 2016)**, sans aucune inversion :
   - distance de corrélation $d_{ij} = \sqrt{\tfrac12(1 - \rho_{ij})}$, clustering
     hiérarchique (single linkage) ;
   - quasi-diagonalisation : réordonnancement des titres selon les feuilles du
     dendrogramme ;
   - **bisection récursive** : à chaque scission, le budget de risque est réparti
     entre les deux sous-clusters en proportion inverse de leur variance
     ($w \propto 1/\sigma^2_{cluster}$, variance de cluster au portefeuille
     inverse-variance interne).

Contraintes finales : 12 % maximum par ligne, position plafonnée à 3 jours de volume
médian en dirhams (contrainte d'impact), repli inverse-volatilité si l'HRP est
indisponible. Si aucun facteur n'a passé les portes de la section 5.3, l'allocation
porte sur l'univers liquide entier.

## 8. Dashboard

`python -m streamlit run app/streamlit_app.py` — six onglets : Marché (cours ajusté,
MM50, volumes, performances), Top 10 (avec attribution des raisons par valeur),
Prévisions (1M→12M avec intervalles et poids de confiance), Modèles (Sharpe par
facteur, portes, IC par horizon), Audit, Portefeuille (allocation HRP, export CSV).
Bouton « Tout analyser » = pipeline complet.

## 9. Utilisation

```bash
pip install -e .

python -m casablanca_quant.cli ingest      # lit data/raw/
python -m casablanca_quant.cli audit
python -m casablanca_quant.cli backtest --cost-bps 100 --rebalance-days 5
python -m casablanca_quant.cli alpha
python -m casablanca_quant.cli portfolio --capital 100000
python -m casablanca_quant.cli all         # tout enchaîner
python -m casablanca_quant.cli demo        # marché synthétique de validation
python -m pytest tests/ -q                 # 12 tests (lookahead, parsing, coûts, HRP...)
```

## 10. Limites actuelles et feuille de route

Le point critique est la **donnée**. Un seul titre est chargé : l'intégralité de la
couche cross-sectionnelle (IC par date, top 10 backtesté, Reality Check sur univers
réel) attend l'historique des ~75 valeurs de la cote.

- [ ] **Univers complet CSE (~75 valeurs)** — priorité absolue : passe de ~700 à
      ~35 000 observations d'apprentissage
- [ ] Profondeur d'historique 5-10 ans (les horizons 6M/12M manquent d'échantillons
      hors chevauchement)
- [ ] Features fondamentales (PER, rendement du dividende, flottant, secteur)
- [ ] Indice MASI comme benchmark de marché explicite (au lieu du proxy equal-weight)
- [ ] Journal de production : recommandations horodatées vs réalisé (tracking error
      du lab lui-même)

## Références

- Lo, A., MacKinlay, C. (1988). *Stock Market Prices Do Not Follow Random Walks*. RFS.
- Jegadeesh, N., Titman, S. (1993). *Returns to Buying Winners and Selling Losers*. JF.
- Amihud, Y. (2002). *Illiquidity and Stock Returns*. JFM.
- White, H. (2000). *A Reality Check for Data Snooping*. Econometrica.
- Ledoit, O., Wolf, M. (2004). *A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices*. JMVA.
- Bailey, D., López de Prado, M. (2014). *The Deflated Sharpe Ratio*. JPM.
- López de Prado, M. (2016). *Building Diversified Portfolios that Outperform Out-of-Sample* (HRP). JPM.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.

## Avertissement

Outil de recherche — pas un conseil en investissement. Les performances simulées ne
préjugent pas des performances futures ; risque de perte en capital.
