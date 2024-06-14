


"""                                            PARAMETRES D'ENTREE                                                                       """
"""======================================================================================================================================"""
Longueur_éprouvette = 100 #mm (distance entre la sortie des mors)

montrer_que_certaines_eprouvettes = [2,3,4,5,6,7,8,9,10]#[26] # soit une liste du numéro d'éprouvette, où alors une liste vide [] quand l'on veut toute les eprouvettes

Rp = 0.2 #%

nom_csv = "C:/Users/ecreach/Documents/PFE Caratérisation impression 3D/test diperssion, 10 eprouvette identique_1.csv"
#nom_csv = "C:/Users/ecreach/Documents/PFE Caratérisation impression 3D/12-06-2024/test 1 eprouvette charge decharge pas 0.1mm_1.csv"
"""======================================================================================================================================"""

print("\n",nom_csv)


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import ticker
import numpy as np
import statistics
#nom_csv = str(input("fichier .csv: "))

inf=9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999

def lire_csv_sans_lib(nom_fichier):
    donnees = []
    with open(nom_fichier, 'r') as fichier:
        for ligne in fichier:
            # Supprime les caractères " de la ligne
            ligne = ligne.replace('"', '')
            # Supprime les espaces en début et fin de ligne et les sauts de ligne
            ligne = ligne.strip()
            # Sépare la ligne en utilisant la virgule comme séparateur
            elements = ligne.split(';')
            # Ajoute la liste d'éléments à la liste principale
            donnees.append(elements)
    return donnees

def commencer_à_0_MPA(liste_deplacement, liste_contrainte, longueur_initiale=100):
    deplacement = np.array(liste_deplacement)
    contrainte = np.array(liste_contrainte)

    offset_deplacement = 0
    cnt0 =  liste_contrainte[0]
    i=1
    cnt1 = liste_contrainte[i]
    while cnt1 <0:
        cnt0 = cnt1
        i+=1
        cnt1 = liste_contrainte[i]

    A = [ liste_deplacement[i-1], liste_contrainte[i-1] ]
    B = [ liste_deplacement[i],   liste_contrainte[i] ]
    offset_deplacement = interpolation( A,B,0)

    deplacement -= offset_deplacement
    longueur_initiale += offset_deplacement

    deformation = deplacement/longueur_initiale*100

    return list(deformation)
        
def interpolation( A,B,y_cible):
    a= (B[1]-A[1]) / (B[0]-A[0])
    b = A[1] - a*A[0]
    x_cible = (y_cible - b) / a
    return x_cible
     
def Contrainte_max(liste_deformation,liste_contrainte):
    max_index = liste_contrainte.index(max(liste_contrainte))
    max_point = [liste_deformation[max_index], liste_contrainte[max_index]]
    return max_point

def ecrire_liste_dans_csv(liste, nom_fichier):
    with open(nom_fichier, 'w') as fichier:
        for element in liste:
            ligne = ";".join(map(str, element)) + "\n"
            fichier.write(ligne)
"""
def calcule_Rp02(liste_contrainte, liste_deformation, module_d_Young):
    cnt=liste_contrainte
    eps=liste_deformation
    E=module_d_Young
    cnt0 = cnt[0]
    cnt1 = cnt[1]
    for i in range(len(liste_contrainte)):
        cnt02 = E*eps[i] - E*(0.2/100)
        if cnt[i] <= cnt02:
            rp02=cnt[i]
            return eps[i], rp02
"""
def calcule_Rp02_sans_E(liste_contrainte, liste_deformation, contrainte_max,Rp=0.2):
    
    cnt=liste_contrainte
    eps=liste_deformation
    cntm=contrainte_max
    cnt0 = cnt[0]
    cnt1 = cnt[1]
    #Calcule du module d'Young
    zone_stable1 = [10,60] #%  de la contrainte max
    cnt_min = min(liste_contrainte)
    zone_stable2=[ ( cntm-cnt_min) * zone_stable1[0]/100+cnt_min,
                   ( cntm-cnt_min) * zone_stable1[1]/100+cnt_min]#MPa
    #print("cnt_min",cnt_min)
    #print("zone_stable2",zone_stable2)
    #print([cntm*zone_stable[0]/100,cntm*zone_stable[1]/100])#valeurs de contraintes utiliser pouyr calculer le module d'Young
    valmin_trouver = False
    valmax_trouver = False
    for i in range(len(liste_contrainte)):

        if valmin_trouver == False:
            if cnt[i]>= zone_stable2[0]:
                valmin = [eps[i], cnt[i]]
                valmin_trouver = True
                #print("valmin_trouver")
                
        elif valmax_trouver == False:
            if cnt[i]>= zone_stable2[1]:
                valmax = [eps[i], cnt[i]]
                valmax_trouver = True
                #print("valmax_trouver")
    E = ( valmax[1] - valmin[1] ) / ( valmax[0] - valmin[0] ) *100

    eps02, cnt02 = calcule_Rp02_avec_E(liste_contrainte, liste_deformation,E,Rp=Rp)
    return eps02, cnt02, E
    """
    for i in range(len(liste_contrainte)):
        cnt02 = E*eps[i] - E*Rp
        if cnt[i] < cnt02:
            P11 = [Rp , 0]
            P12 = [Rp+1 , E]
            P21 = [eps[i-1] , cnt[i-1]]
            P22 = [eps[i] , cnt[i]]

            eps02, cnt02 = intersection_droites( P11, P12, P21, P22)
            return eps02, cnt02, E*100
        elif cnt[i] == cnt02:
            return eps[i] , cnt[i]
    """
def calcule_Rp02_avec_E(liste_contrainte, liste_deformation,E,Rp=0.2): #def en %, cnt en MPa, E en MPa, RP en %
    cnt=liste_contrainte
    eps=liste_deformation
    E /= 100 
    for i in range(len(liste_contrainte)):
        cnt02 = E*eps[i] - E*Rp
        if cnt[i] < cnt02:
            P11 = [Rp , 0]
            P12 = [Rp+1 , E]
            P21 = [eps[i-1] , cnt[i-1]]
            P22 = [eps[i] , cnt[i]]

            eps02, cnt02 = intersection_droites( P11, P12, P21, P22)
            return eps02, cnt02
        elif cnt[i] == cnt02:
            return eps[i] , cnt[i]

def calcule_Rp02_adaptatif(liste_contrainte, liste_deformation,Rp=0.2,force_moyenne=15,seuille_linéarité=10): #def en %, cnt en MPa, en MPa, RP en %, seuille_linéarité en %
    #on decale la deformation quand elle est stabiliser
    #Rp=0
    cnt=liste_contrainte
    eps=liste_deformation
    E, deps, dcnt = calcule_E_adaptatif(eps, cnt, force_moyenne=force_moyenne, return_derivé=True)
    E /= 100
    deps_moy, dcnt_moy = moyenner_courbe(deps, dcnt, force=force_moyenne)
    
    #plt.plot(deps,dcnt)

    ddeps = deps_moy[:-1]
    ddcnt = []
    for i in range(len(dcnt_moy)-1): #dérivation
        ddcnt.append((dcnt_moy[i+1] - dcnt_moy[i]) / (deps_moy[i+1] - deps_moy[i]))
    ddeps_moy, ddcnt_moy = moyenner_courbe(ddeps, ddcnt, force=force_moyenne)
    #plt.plot(ddeps_moy,ddcnt_moy)

    #seuille_linéarité = 10 #%
    seuille_linéarité/=100

    ddcnt_max = max(ddcnt_moy)
    i_max = ddcnt_moy.index(ddcnt_max)
    ddeps_moy = ddeps_moy[i_max:]# on prend que la partie de la courbe qui descend
    ddcnt_moy = ddcnt_moy[i_max:]


    #plt.plot(ddeps_moy,ddcnt_moy)
    #plt.show()
    
    #determination de l'offset de Rp a faire
    Rp_offset = 0
    seuille_linéarité *= ddcnt_max
    print("seuille_linéarité",seuille_linéarité)
    continuer = True
    i = 0
    while continuer :
        if i >= len(ddcnt_moy):
            continuer =False
        elif ddcnt_moy[i] <= seuille_linéarité:
            #Rp_offset = ddeps_moy[i]
            continuer = False
            Rp_offset = interpolation( [ddeps_moy[i-1],ddcnt_moy[i-1]] , [ddeps_moy[i],ddcnt_moy[i]] , seuille_linéarité )
        else:
            i +=1
    Rp += Rp_offset
    print("Rp_offset:",Rp_offset,"\nNouveau Rp:",Rp)

    #determination du point a partir duqeul la courbe de traction est ~linéaire
    continuer = True
    i = 0
    while continuer :
        if i >= len(eps):
            continuer =False
        elif eps[i] >= Rp_offset:
            continuer = False
            cnt_lineaire = interpolation( [cnt[i-1],eps[i-1]] , [cnt[i],eps[i]] , Rp_offset )
        else:
            i +=1
    
    P11 = [  Rp       , cnt_lineaire]   
    P12 = [P11[0]+100 , P11[1]+E*100]
    a,b = equation_droite(P11,P12)

    #affichage de la droite du module d'Young
    #P10 = [P11[0]-100 , P11[1]-E*100]
    #plt.plot( [P10[0],P12[0]] , [P10[1],P12[1]] )
    
    for i in range(len(liste_contrainte)):
        #cnt02 = E*eps[i] - E*Rp
        cnt02 = a*eps[i] + b
        if cnt[i] < cnt02:
            P21 = [eps[i-1] , cnt[i-1]]
            P22 = [eps[i] , cnt[i]]

            eps02, cnt02 = intersection_droites( P11, P12, P21, P22)
            return eps02, cnt02
        elif cnt[i] == cnt02:
            return eps[i] , cnt[i]
     
def extraire_donee_par_eprouvettes(donnee, nb_eprouvette):
    liste_deplacement = []
    liste_force = []
    liste_contraintes = []
    liste_deformation = []
    liste_eprouvette_vide = []

    nb_eprouvette_reel = nb_eprouvette
    numéro_eprouvette = 1
    ligne = taille_entete+3
    while numéro_eprouvette <= nb_eprouvette: #on iterre sur toute les eprouvettes
        #print("numéro_eprouvette:",numéro_eprouvette)
        liste_contraintes.append([])
        liste_deformation.append([])
        liste_deplacement.append([])
        liste_force.append([])
        #print("donnee[ligne]: ",donnee[ligne])
        while donnee[ligne][0] == '' and len(donnee[ligne])>1 :
            liste_deplacement[-1].append(float(donnee[ligne][2]))
            liste_force[-1].append(      float(donnee[ligne][3]))
            liste_contraintes[-1].append(float(donnee[ligne][4]))
            liste_deformation[-1].append(float(donnee[ligne][5]))
            
            ligne += 1
        if numéro_eprouvette != nb_eprouvette: #si on est pas a la fin des eprouvettes
            ligne += 2
        if liste_contraintes[-1] == []: #si il n'y avait aucune donnée dans l'eprouvette
            print("/!\\ L'éprouvette n°",numéro_eprouvette,"n'as aucune donnée! /!\\")
            del liste_deplacement[-1] #on enleve toutes les liste inutiles
            del liste_force[-1]
            del liste_contraintes[-1]
            del liste_deformation[-1]
            nb_eprouvette_reel -= 1
            liste_eprouvette_vide.append(numéro_eprouvette)
        numéro_eprouvette += 1

    if nb_eprouvette_reel != nb_eprouvette:
        print("Nombre d'éprouvette avec des données:",nb_eprouvette_reel)
    return liste_deplacement, liste_force, liste_contraintes, liste_deformation, nb_eprouvette_reel, liste_eprouvette_vide

def coefficient_directeur( p1, p2):
    a = (p2[1]-p1[1]) / (p2[0]-p1[0])
    return a

def calculer_E_a_partir_de_la_decharge(liste_déformation, liste_contrainte, critere_stabilité=10, force=15):
    #liste_déformation, liste_contrainte = moyenner_courbe( liste_déformation, liste_contrainte,force=force)
    liste_deformation_decharge = []
    liste_contrainte_decharge = []
    ind_cnt_max = liste_contrainte.index( max(liste_contrainte))
    
    i = ind_cnt_max# on commence à la décharge
    #critere_stabilité: % de variation

    pente0 = coefficient_directeur( [liste_déformation[i-2]/100 , liste_contrainte[i-2]],
                                    [liste_déformation[i-1]/100 , liste_contrainte[i-1]])

    P_haut = [] #point utiliser pour calculer le modul d'young, situé en haut de la courbe
    P_bas = []  #point utiliser pour calculer le modul d'young, situé en bas de la courbe
    while i < len(liste_contrainte): 
        pente1 = coefficient_directeur( [liste_déformation[i-1]/100 , liste_contrainte[i-1]],
                                        [liste_déformation[ i ]/100 , liste_contrainte[ i ]])
        if pente1 != 0: #manoucherie pour eviter une division par 0
            variation = abs(pente0 / pente1)*100 #%
        else:
            variation = inf
            
        if variation < critere_stabilité and P_haut == []:
            P_haut = [liste_déformation[ i ] , liste_contrainte[ i ]]
                      
        elif (variation > critere_stabilité and P_haut != [] and P_bas == []): #or i == len(liste_contrainte)-1:
            P_bas = [liste_déformation[ i ] , liste_contrainte[ i ]]

            print("P_bas, P_haut",P_bas, P_haut)
            E = abs(coefficient_directeur( P_bas, P_haut))
            print("E",E)
            return E
        pente0 = pente1
        i += 1
    return False #si on ne pas calculer le module d'Young

def calculer_E_a_partir_de_la_ISO_527(liste_déformation, liste_contrainte):
    eps1 = 0.05 #%
    eps2 = 0.25 #%

    if min(liste_déformation) <= eps1 and max(liste_déformation) >= eps2:
        continuer_boucle = True
        i = 0
        while continuer_boucle:
            eps = liste_déformation[i]
            if eps == eps1:
                sig1 = liste_contrainte[i]
            elif eps > eps1:
                sig1 = interpolation( [liste_contrainte[i-1],liste_déformation[i-1]] , [liste_contrainte[i],liste_déformation[i]] , eps1)
                
                continuer_boucle = False
            i+=1
                
        continuer_boucle = True
        while continuer_boucle:
            eps = liste_déformation[i]
            if eps == eps2:
                sig2 = liste_contrainte[i]
            elif eps > eps2:
                sig2 = interpolation( [liste_contrainte[i-1],liste_déformation[i-1]] , [liste_contrainte[i],liste_déformation[i]] , eps2)
                continuer_boucle = False
            i += 1
        print([eps1,sig1] , [eps2,sig2])
        E = coefficient_directeur( [eps1/100,sig1] , [eps2/100,sig2] )
        return E
    return None

def calcule_E_adaptatif(liste_déformation, liste_contrainte, delta_dcnt=0.25, force_moyenne=15, return_derivé=False): # eps (%) ; cnt (MPa)
    X = liste_déformation
    Y = liste_contrainte
    i_Y_max = Y.index(max(Y))

    X = X[:i_Y_max]#on s'interesse qu'a la première partie de la courbe
    Y = Y[:i_Y_max]

    
    dX = X[:-1]
    dY=[]
    for i in range(len(Y)-1):
        dY.append((Y[i+1] - Y[i]) / (X[i+1] - X[i]))

    dX_moy, dY_moy = moyenner_courbe(dX, dY, force=force_moyenne)
    dY_max = max(dY_moy)
    i_dY_max = dY_moy.index(dY_max)

    imin = i_dY_max
    while dY_moy[imin] > dY_max-delta_dcnt:
        imin -= 1

    imax = i_dY_max
    while dY_moy[imax] > dY_max-delta_dcnt:
        imax += 1

    n_val = ( imax - imin +1)
    E = 0
    for i in range( imin, imax+1):
        E += dY_moy[i]
    E = E/n_val
    print("n_val",n_val)
    print([ liste_déformation[imin] , liste_déformation[imax] ])

    if return_derivé:
        return E*100, dX, dY
    else:
        return E*100
    
        
def moyenner_courbe( X, Y,force=5): #la force du moyennage est le nombre de points pris dans la moyenne de part e d'autre de chaque point
    force = int(force)
    X_moy = X[force : -force]
    Y_moy = []
    for i in range(len(X)-force*2):
        i += force
        moy=0
        for j in range(-force , force+1):
            moy += Y[i+j]
        moy = moy / (force*2+1)
        Y_moy.append(moy)
    return X_moy, Y_moy
        
def equation_droite(P1,P2):
    a = coefficient_directeur(P1, P2)
    b = P1[1] - a*P1[0]
    return a,b

def intersection_droites( P11, P12, P21, P22):
    a1, b1 = equation_droite(P11,P12)
    a2, b2 = equation_droite(P21,P22)

    x = (b2 - b1) / (a1 - a2)
    y = a1 * x + b1

    return x ,y

def depouiller_essais_traction_simple(nom_csv):
    
    donnee = lire_csv_sans_lib(nom_csv ) #ouverture du fichier .csv exporter depuis la machine de traction

    #determination de la taille de l'entette et du nombre d'eprouvettes
    taille_entete=3 # en comptant a partir de 0
    nb_eprouvette = 1

    while donnee[taille_entete+1][0].isnumeric() == True:
        #print("donnee[taille_entete+1][0] ",donnee[taille_entete+1][0])
        taille_entete+=1
        nb_eprouvette += 1

    taille_entete += 5

    print("\nnombre d'éprouvettes: ",nb_eprouvette)

    #Extraction des donées pour chaque eprouvettes:
    liste_deplacement, liste_force, liste_contraintes, liste_deformation, nb_eprouvette_reel, liste_eprouvette_vide = extraire_donee_par_eprouvettes(donnee, nb_eprouvette)


    #création de la liste d'élémentexporter dan le rapport
    export_data=[["Longueur entre la sortie des mors (mm)",Longueur_éprouvette],
                 [""],
                 ["n°éprouvette","nom éprouvette","Vitesse d'essai","Contrainte max","Deformation à contrainte max","Module de Young machine","Module de Young 'adaptatif'","Module de Young ISO 527","Rp0,2"],
                 ["","",donnee[2][2],donnee[2][3],donnee[2][4],donnee[2][5],donnee[2][5],donnee[2][5],donnee[2][5]]]
    
    #création de la palette de couleurs utiliser dans les graphique
    palette = [ 'navy','mediumblue', 'dodgerblue', 'deepskyblue', 'royalblue','tab:blue','cornflowerblue','lightskyblue','lightsteelblue','lavender'] # couleurs utiliser sur le graph
    #for style in ['classic', 'seaborn-v0_8-bright', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white']:
    #for style in plt.style.available:
        #print("style:",style)


    #affichage des courbes sur un graphique   
    fig, ax = plt.subplots()
    #plt.style.use(style)

    list_point_cnt_max = []
    list_cnt_max = []
    list_def_cnt_max = []
    list_rp02 = []
    l_rp02 = []
    ep_rp02 = []
    list_E = []

    window_size = ['', '', '', '']

    centrer_deformation_à_0 = input("\nDeplacer les deformation en 0? [ENTER / N] ")#demande à l'utilisateur si il veut décaler l'axe des déformation pour compensser la mise en tension du filament au début de l'essais
    if centrer_deformation_à_0 == "n" or centrer_deformation_à_0 == "N":
        centrer_deformation_à_0 = False
    else:
        centrer_deformation_à_0 = True
        
    print("liste_eprouvette_vide",liste_eprouvette_vide)
    print("nb_eprouvette:",nb_eprouvette)
    ep_sauter = 0
    for ep in range(nb_eprouvette):
        print("\nep",ep)
        if not( (ep+1) in liste_eprouvette_vide ):#on vérifie que l'éprouvette à des données associées
            if (montrer_que_certaines_eprouvettes == []) or (ep+1 in montrer_que_certaines_eprouvettes):
                print("n° eprouvette:",ep+1)
                k = ep%len(palette)
                if centrer_deformation_à_0:
                    deformation = commencer_à_0_MPA(liste_deformation[ep-ep_sauter],liste_contraintes[ep-ep_sauter],longueur_initiale=Longueur_éprouvette)
                else:
                    deformation = liste_deformation[ep-ep_sauter]
                X = np.array(deformation)
                Y = np.array(liste_contraintes[ep-ep_sauter])
                nom = ''.join(['ep n°',str(ep+1)])
                plt.plot(X, Y, color=palette[k])# ,label=nom)
                cnt_max=Contrainte_max(list(X),list(Y))
                list_point_cnt_max.append(cnt_max)
                list_cnt_max.append(cnt_max[1])
                list_def_cnt_max.append(cnt_max[0])

                print("nb de points de l'éprouvette",len(liste_contraintes[ep-ep_sauter]))
                ligne_entete = 3+ep
                E_machine = donnee[ligne_entete][5]
                print("E (machine de traction)",E_machine)
                E_ISO_527 = calculer_E_a_partir_de_la_ISO_527(deformation, liste_contraintes[ep-ep_sauter])
                print("E ISO 527",E_ISO_527)
                #Calcule du Rp0.2 et E
                #print("len(deformation)",len(deformation))
                #print("cnt_max[1]",cnt_max[1])
                E_adaptatif = calcule_E_adaptatif(deformation, liste_contraintes[ep-ep_sauter])
                print("E_adaptatif",E_adaptatif)
                eps02, rp02, E=calcule_Rp02_sans_E(liste_contraintes[ep-ep_sauter], deformation, cnt_max[1])
                #eps02, rp02 =  calcule_Rp02_avec_E(liste_contraintes[ep-ep_sauter], deformation, E_ISO_527)
                
                eps02, rp02 = calcule_Rp02_avec_E(liste_contraintes[ep-ep_sauter], deformation, E_adaptatif)
                eps02, rp02 = calcule_Rp02_adaptatif(liste_contraintes[ep-ep_sauter], deformation, Rp=Rp)
                
                list_rp02.append([eps02, rp02])
                l_rp02.append(rp02)
                ep_rp02.append(eps02)
                print("E (Rp0.2)",E)
                list_E.append(E)

                #determinasion de la taille de la fenetre d'affichage:
                X = X.tolist()
                Y = Y.tolist()
                if (window_size[0] == '') or (min(X) < window_size[0]):
                    window_size[0] = min(X)
                if (window_size[2] == '') or (min(Y) < window_size[2]):
                    window_size[2] = min(Y)
                if (window_size[3] == '') or (max(Y) > window_size[3]):
                    window_size[3] = max(Y)

                if ((window_size[1] == '') or
                    (max(X) > window_size[1]) and Y[X.index(max(X))]>2):
                    window_size[1] = max(X)
                elif X[Y.index(max(Y))] > window_size[1]:
                    window_size[1] = X[Y.index(max(Y))]

                
                

                
                #on met dans le rapport les élément de l'entete qui nous interesse
                
                export_data.append([ donnee[ligne_entete][0], donnee[ligne_entete][1], donnee[ligne_entete][2], cnt_max[1], cnt_max[0], E,E_adaptatif, E_ISO_527, rp02])

        else:
            ep_sauter +=1

#rajout d'une marge sur la fenetre d'affichage
marge = 5 #%
dx = window_size[1] - window_size[0]
window_size[0] = window_size[0]- dx * (marge/100)
window_size[1] = window_size[1]+ dx * (marge/100)
dy = window_size[3] - window_size[2]
window_size[2] = window_size[2]- dy * (marge/100)
window_size[3] = window_size[3]+ dy * (marge/100)

#plt.axis([-2, 20, -3, 65])
                                       
plt.plot(X, Y, color=palette[k])

#affichage des contraintes max pour chaque courbes
for cnt_max in list_point_cnt_max:
    plt.plot(cnt_max[0],cnt_max[1], '+', color = 'crimson')

#affichage des Rp0,2 pour chque courbes
for pnt in list_rp02:
    plt.plot(pnt[0],pnt[1], '+', color = 'red')
    



# moyenne contrainte max
if len(list_cnt_max) >= 2:
    moyenne_contrainte_max = statistics.mean(list_cnt_max)
    ecart_type_contrainte_max = statistics.stdev(list_cnt_max)

    moyenne_deformation_contrainte_max = statistics.mean(list_def_cnt_max)
    ecart_type_deformation_contrainte_max = statistics.stdev(list_def_cnt_max)
else:
    moyenne_contrainte_max = list_cnt_max[0]
    ecart_type_contrainte_max = 0

    moyenne_deformation_contrainte_max = list_def_cnt_max[0]
    ecart_type_deformation_contrainte_max = 0

nb_ar=2

print("\n            ===\n")
print("nombre d'éprouvettes utilisé:",len(list_cnt_max),"\n")
print("médiane de la contrainte max:",statistics.median(list_cnt_max))
print("")
print("moyenne_contrainte_max:",moyenne_contrainte_max)
print("ecart_type:",ecart_type_contrainte_max)
print("")
print("Dans   68% des cas σmax ∈",[round(moyenne_contrainte_max-ecart_type_contrainte_max, nb_ar) , round(moyenne_contrainte_max+ecart_type_contrainte_max, nb_ar)],"MPa")
print("Dans   95% des cas σmax ∈",[round(moyenne_contrainte_max-2*ecart_type_contrainte_max, nb_ar) , round(moyenne_contrainte_max+2*ecart_type_contrainte_max, nb_ar)],"MPa")
print("Dans 99.7% des cas σmax ∈",[round(moyenne_contrainte_max-3*ecart_type_contrainte_max, nb_ar) , round(moyenne_contrainte_max+3*ecart_type_contrainte_max, nb_ar)],"MPa")


export_data.append([''])
export_data.append(["Moyenne de la contrainte max",moyenne_contrainte_max])
export_data.append(["Ecart type",ecart_type_contrainte_max])
export_data.append(['Dans   68% des cas Contrainte max €',round(moyenne_contrainte_max-ecart_type_contrainte_max, nb_ar) , round(moyenne_contrainte_max+ecart_type_contrainte_max, nb_ar)])
export_data.append(['Dans   95% des cas Contrainte max €',round(moyenne_contrainte_max-2*ecart_type_contrainte_max, nb_ar) , round(moyenne_contrainte_max+2*ecart_type_contrainte_max, nb_ar)])
export_data.append(['Dans 99.7% des cas Contrainte max €',round(moyenne_contrainte_max-3*ecart_type_contrainte_max, nb_ar) , round(moyenne_contrainte_max+3*ecart_type_contrainte_max, nb_ar)])
export_data.append([''])
export_data.append(['Mediane de la contrainte max',statistics.median(list_cnt_max)])
                   
#affichage moyenne contrainte max
x_txt = moyenne_deformation_contrainte_max*1 + 6*ecart_type_deformation_contrainte_max
y_txt = moyenne_contrainte_max*1 + 4*ecart_type_contrainte_max
txt = ''.join([r"$\sigma_{max}$ ∈ [",str(round(moyenne_contrainte_max-2*ecart_type_contrainte_max, nb_ar))," ; ",str(round(moyenne_contrainte_max+2*ecart_type_contrainte_max, nb_ar)),'] à 95%'])
plt.text(x_txt, y_txt, txt)
p1 = patches.FancyArrowPatch( (moyenne_deformation_contrainte_max, moyenne_contrainte_max), (x_txt, y_txt),arrowstyle='<-', mutation_scale=20)
plt.gca().add_patch(p1)

n=2
xerr = [ecart_type_deformation_contrainte_max*n]# , ecart_type_deformation_contrainte_max*n]
yerr = [ecart_type_contrainte_max*n]#, ecart_type_contrainte_max*n]
plt.errorbar([moyenne_deformation_contrainte_max], [moyenne_contrainte_max], xerr=xerr, yerr=yerr, capsize=3, fmt="o", ecolor = "darkolivegreen", color = "gold",label = "moyenne des contraintes max avec\nun intervale de confiance à 95%")

# moyenne Rp0.2
if len(l_rp02) >= 2:
    moyenne_rp02= statistics.mean(l_rp02)
    ecart_type_rp02 = statistics.stdev(l_rp02)

    moyenne_deformation_rp02= statistics.mean(ep_rp02)
    ecart_type_deformation_rp02 = statistics.stdev(ep_rp02)

    moyenne_E = statistics.mean(list_E)
    ecart_type_E = statistics.stdev(list_E)
else:
    moyenne_rp02= l_rp02[0]
    ecart_type_rp02 = 0

    moyenne_deformation_rp02= ep_rp02[0]
    ecart_type_deformation_rp02 = 0

    moyenne_E = list_E[0]
    ecart_type_E = 0

nb_ar=2

print("\n            ===\n")
print("nombre d'éprouvettes utilisé:",len(list_cnt_max),"\n")
print("médiane d Rp0.2:",statistics.median(l_rp02))
print("")
print("moyenne_Rp0.2:",moyenne_rp02)
print("ecart_type_rp02:",ecart_type_rp02)
print("")
print("Dans   68% des cas Rp0.2 ∈",[round(moyenne_rp02-ecart_type_rp02, nb_ar) , round(moyenne_rp02+ecart_type_rp02, nb_ar)],"MPa")
print("Dans   95% des cas Rp0.2 ∈",[round(moyenne_rp02-2*ecart_type_rp02, nb_ar) , round(moyenne_rp02+2*ecart_type_rp02, nb_ar)],"MPa")
print("Dans 99.7% des cas Rp0.2 ∈",[round(moyenne_rp02-3*ecart_type_rp02, nb_ar) , round(moyenne_rp02+3*ecart_type_rp02, nb_ar)],"MPa")


export_data.append([''])
export_data.append(["Moyenne des Rp0.2",moyenne_rp02])
export_data.append(["Ecart type des Rp0.2",ecart_type_rp02])
export_data.append(['Dans   68% des cas Rp0.2 €',round(moyenne_rp02-ecart_type_rp02, nb_ar) , round(moyenne_rp02+ecart_type_rp02, nb_ar)])
export_data.append(['Dans   95% des cas Rp0.2 €',round(moyenne_rp02-2*ecart_type_rp02, nb_ar) , round(moyenne_rp02+2*ecart_type_rp02, nb_ar)])
export_data.append(['Dans 99.7% des cas Rp0.2 €',round(moyenne_rp02-3*ecart_type_rp02, nb_ar) , round(moyenne_rp02+3*ecart_type_rp02, nb_ar)])
export_data.append([''])
export_data.append(['Mediane du Rp0.2',statistics.median(l_rp02)])
#affichage moyenne Rp0.2

x_txt = moyenne_deformation_rp02*1 + 7*ecart_type_deformation_rp02
y_txt = moyenne_rp02*1 - 3*ecart_type_rp02
txt = ''.join([" \n$Rp_{{{}}}$ ∈ [".format(Rp)
               ,str(round(moyenne_rp02-2*ecart_type_rp02, nb_ar))," ; ",str(round(moyenne_rp02+2*ecart_type_rp02, nb_ar)),'] à 95%'])#\n(E ∈ [',str(round(moyenne_E-2*ecart_type_E, nb_ar))," ; ",str(round(moyenne_E+2*ecart_type_E, nb_ar)),'] à 95%)'])
plt.text(x_txt, y_txt, txt)
p1 = patches.FancyArrowPatch( (moyenne_deformation_rp02, moyenne_rp02), (x_txt, y_txt),arrowstyle='<-', mutation_scale=20)
plt.gca().add_patch(p1)

n=2
xerr = [ecart_type_deformation_rp02*n]# , ecart_type_deformation_contrainte_max*n]
yerr = [ecart_type_rp02*n]#, ecart_type_contrainte_max*n]
plt.errorbar([moyenne_deformation_rp02], [moyenne_rp02], xerr=xerr, yerr=yerr, capsize=3, fmt="o", ecolor = "saddlebrown", color = "lime",label = "moyenne des Rp0.2 avec\nun intervale de confiance à 95%")



ecrire_liste_dans_csv(export_data,"test.csv")


#plt.axis([-2, 20, -3, 65])
plt.axis(window_size)

#cadrillage
#https://stackoverflow.com/questions/24943991/change-grid-interval-and-specify-tick-labels
maj_posy = ticker.MultipleLocator(5)   # major ticks for every 5 units
min_posy = ticker.MultipleLocator(1)    # minor ticks for every 1 units
maj_posx = ticker.MultipleLocator(0.5)   
min_posx = ticker.MultipleLocator(0.1)    

ax.xaxis.set(major_locator=maj_posx, minor_locator=min_posx)
ax.yaxis.set(major_locator=maj_posy, minor_locator=min_posy)

ax.tick_params(axis='both', which='minor', length=0)   # remove minor tick lines

# different settings for major & minor gridlines
ax.grid(which='major', alpha=0.5)
ax.grid(which='minor', alpha=0.2, linestyle='--')

plt.ylabel('Contrainte (MPa)')
plt.xlabel('Déformation de traction (%)')
plt.title('Essai de traction sur filament PLA noir Francofil\n10 éprouvette Ø1.75 L100')#\n10/06/2024')
plt.legend()
#plt.grid()
plt.show()

ep_sauter=0
force_moyenne = 15
for ep in range(nb_eprouvette):
    if not( (ep+1) in liste_eprouvette_vide ):
        if (montrer_que_certaines_eprouvettes == []) or (ep+1 in montrer_que_certaines_eprouvettes):
            print("n° eprouvette:",ep+1)
            k = ep%len(palette)
            if centrer_deformation_à_0:
                deformation = commencer_à_0_MPA(liste_deformation[ep-ep_sauter],liste_contraintes[ep-ep_sauter],longueur_initiale=Longueur_éprouvette)
            else:
                deformation = liste_deformation[ep-ep_sauter]
            X2 = deformation
            Y2 = liste_contraintes[ep-ep_sauter]
            dX = X2[:-1]
            dY=[]
            for i in range(len(Y2)-1):
                dY.append((Y2[i+1] - Y2[i]) / (X2[i+1] - X2[i]))

            dX_moy, dY_moy = moyenner_courbe(dX, dY, force=force_moyenne)
            
            E = calculer_E_a_partir_de_la_decharge( X2, Y2)
            
            nom = ''.join(['ep n°',str(ep+1)])
            #plt.plot(dX, dY,linewidth=0.25, linestyle='dotted',color=palette[k])# ,label=nom) color="gray" #   DERIVE BRUTE
            plt.plot(dX_moy, dY_moy,linewidth=0.3,color=palette[k]) #                                          DERIVE MOYENNER
            #plt.plot(dX, dY, color=palette[k])# ,label=nom)
            
            ddX_moy = dX_moy[:-1]
            ddY_moy = []
            for i in range(len(dY_moy)-1):
                ddY_moy.append((dY_moy[i+1] - dY_moy[i]) / (dX_moy[i+1] - dX_moy[i]))
            #print(len(ddX_moy))
            #print(len(ddY_moy))
            ddX_moy_moy, ddY_moy_moy = moyenner_courbe(ddX_moy, ddY_moy, force=force_moyenne)
            plt.plot(ddX_moy_moy, ddY_moy_moy,linewidth=0.5,color=palette[k],linestyle='dotted')
            
    else:
        ep_sauter +=1


fig.show()

plt.show()

fig2, ax2 = plt.subplots()
#DERIVER
for ep in range(nb_eprouvette):
    k = ep%len(palette)
    print("centrer_deformation_à_0",centrer_deformation_à_0)
    if centrer_deformation_à_0:
        deformation = commencer_à_0_MPA(liste_deformation[ep],liste_contraintes[ep],longueur_initiale=Longueur_éprouvette)
    else:
        deformation = liste_deformation[ep]
    X2 = deformation
    Y2 = liste_contraintes[ep]
    dX = X[:-1]
    dY=[]
    for i in range(len(Y2)-1):
        dY.append((Y2[i+1] - Y2[i]) / (X2[i+1] - X2[i]))
        

    
    nom = ''.join(['ep n°',str(ep+1)])
    plt.plot(dX, dY, color=palette[k])# ,label=nom)
plt.axis([min(dX), max(dX), -5, max(dY)])

ax2.xaxis.set(major_locator=maj_posx, minor_locator=min_posx)
ax2.yaxis.set(major_locator=maj_posy, minor_locator=min_posy)

ax2.tick_params(axis='both', which='minor', length=0)   # remove minor tick lines

# different settings for major & minor gridlines
ax2.grid(which='major', alpha=0.5)
ax2.grid(which='minor', alpha=0.2, linestyle='--')

plt.ylabel('Dérivé Contrainte (MPa)')
plt.xlabel('Déformation de traction (%)')
plt.show()





































