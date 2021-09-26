import scipy
from scipy.stats import ks_2samp
import scipy.stats as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters
from sklearn.metrics import roc_curve  
register_matplotlib_converters()

def plot_roc_curve(fper, tper):  
    sns.set(style="whitegrid")
    plt.plot(fper, tper, color='cornflowerblue', label='ROC')
    plt.plot([0, 1], [0, 1], color='slategrey', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

def corré_quant(serie_1,serie_2):
    print('Covariance empirique : ',np.cov(serie_1,serie_2,ddof=0)[1,0])
    print('Coefficient de Pearson : ',st.pearsonr(serie_1,serie_2)[0])
    if (st.pearsonr(serie_1,serie_2)[0] < 0.5)|(st.pearsonr(serie_1,serie_2)[0] > 0.5):
        print('')
        print('Les valeurs suivent une distribution linéaire')
    else:
        print('')
        print('Les valeurs suivent une distribution non-linéaire')

def get_stationarity(timeseries):
    
    # Statistiques mobiles
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()
    
    # tracé statistiques mobiles
    plt.figure(figsize=(14,5))
    original = plt.plot(timeseries, color='blue', label='Origine')
    mean = plt.plot(rolling_mean, color='red', label='Moyenne Mobile')
    std = plt.plot(rolling_std, color='black', label='Ecart-type Mobile')
    plt.legend(loc='best')
    plt.title('Moyenne et écart-type Mobiles')
    plt.show(block=False)    
    
    # Test Dickey–Fuller :
    result = adfuller(timeseries)
    print(' ')
    print(' ')
    print('Statistiques ADF : {}'.format(result[0]))
    print('p-value : {}'.format(result[1]))
    print('Valeurs Critiques :')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))        
    print(' ')
    if result[1] < 0.05:
        print('La série temporelle est stationnaire')
    else:
        if result[1] < 0.2:
            print("La série temporelle est 'presque' stationnaire")
        else:
            print("La série temporelle n'est pas stationnaire")

def indep_chi_2(dataframe,X,Y,VC,alpha):
    chi = dataframe[[X,Y]].pivot_table(index=X,columns=Y,aggfunc=len,margins=True,margins_name="Total")
    
    tx = chi.loc[:,["Total"]]
    ty = chi.loc[["Total"],:]
    n = len(dataframe)
    indep = tx.dot(ty) / n

    c = chi.fillna(0) # On remplace les valeurs nulles par 0
    measure = (c-indep)**2/indep
    xi_n = measure.sum().sum()
    table = (measure/xi_n)*100
    sns.heatmap(table.iloc[:4,0:-1],annot=True, cmap='Reds')
    plt.title('Heat map montrant la corrélation entre l\'âge des clients et les catégories de produits')
    plt.show()
    chi2, pvalue, degrees, expected = st.chi2_contingency(chi)
    
    if VC < chi2: 
        print(' ')
        print('Rejet de l\'hypothèse 0 pour un risque de',alpha)
        print(' ')
    else:
        print(' ')
        print('Non-Rejet de l\'hypothèse 0 pour un risque de',alpha)
        print(' ')
    print('_ _ '*10)
    if pvalue < alpha: 
        print(' ')
        print('Le test est significatif')
        print(' ')
    else: 
        print(' ')
        print('Le test est non-significatif')
        print(' ')
        return chi
        
def adequation_kolmo_shap(var):
    print('Résultat du test d\'adéquation :')
    print('')
    print('- Test de Kolmogorov-Smirnov :')
    print(' ')
    if ks_2samp(var,list(np.random.normal(np.mean(var), np.std(var), 1000)))[1] > 0.05: 
        print('L\'hypothèse 0 n\'est pas rejetée pour un risque de 5 %.')
        print('')
    else: 
        print('L\'hypothèse 0 est rejetée pour un risque de 5 %.')
        if ks_2samp(var,list(np.random.normal(np.mean(var), np.std(var), 1000)))[1] > 0.01:
            print('L\'hypothèse 0 n\'est pas rejetée pour un risque de 1 %.')
            print('')
        else: 
            print('L\'hypothèse 0 est rejetée.')
            print('')

def shapiro(var):
    print('- Test de Shapiro-Wilk :')
    print(' ')
    print('Valeur critique : ',scipy.stats.shapiro(var)[1])
    print(' ')
    if scipy.stats.shapiro(var)[1] > 0.05: 
        print('Sachant que les données sont normalement distribuées, l\'hypothèse 0 n\'est pas rejetée.')
    else:
        print('L\'hypothèse 0 est rejetée.')


def compa(var1,var2):
    print('Résultat du test de comparaison des moyennes et variances :')
    print(' ')
    print('- Comparaison des moyennes :')
    print(' ')
    print('Valeur critique : ',scipy.stats.ttest_ind(var1,var2, equal_var=True)[1])
    print(' ')
    if scipy.stats.ttest_ind(var1,var2, equal_var=True)[1] > 0.05: 
        print('L\'hypothèse 0 n\'est pas rejetée pour un risque de 5 %.')
    else: 
        print('L\'hypothèse 0 est rejetée pour un risque de 5 %.')
        if scipy.stats.ttest_ind(var1,var2, equal_var=True)[1] > 0.01:
            print('L\'hypothèse 0 n\'est pas rejetée pour un risque de 1 %.')
            print('')
        else: 
            print('L\'hypothèse 0 est rejetée.')
            print('')
    print('- Comparaison des variances :')
    print(' ')
    if scipy.stats.bartlett(var1,var2)[1] > 0.05: 
        print('L\'hypothèse 0 n\'est pas rejetée pour un risque de 5 %.')
    else: 
        print('L\'hypothèse 0 est rejetée pour un risque de 5 %.')
        if scipy.stats.bartlett(var1,var2)[1] > 0.01:
            print('L\'hypothèse 0 n\'est pas rejetée pour un risque de 1 %.')
        else: 
            print('L\'hypothèse 0 est rejetée.')
            

def anov(var1,var2):
    p = scipy.stats.levene(var1,var2)[1]
    print('Valeur critique : ',scipy.stats.levene(var1,var2)[1])
    print(' ')
    if p > 0.05: 
        print('L\'hypothèse 0 n\'est pas rejetée pour un risque de 5 %.')
    else: 
        print('L\'hypothèse 0 est rejetée pour un risque de 5 %.')
        if p > 0.01:
            print('L\'hypothèse 0 n\'est pas rejetée pour un risque de 1 %.')
        else: 
            print('L\'hypothèse 0 est rejetée.')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(7,6))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)
        
def display_factorial_planes(X_projected, pca, axis_ranks, labels=None, alpha=1):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure       
            fig = plt.figure(figsize=(7,6))
        
            

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.savefig('Eboulis.png')
    plt.show(block=False)