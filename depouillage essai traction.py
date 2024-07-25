


"""                                            PARAMETRES D'ENTREE                                                                       """
"""======================================================================================================================================"""
Longueur_éprouvette = 100 #mm (distance entre la sortie des mors)

montrer_que_certaines_eprouvettes = [0,1,3,4,5]#[1,3,4,5]#[26] # soit une liste du numéro d'éprouvette, où alors une liste vide [] quand l'on veut toute les eprouvettes

Rp = 0.2 #%
  

"""======================================================================================================================================"""




import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import ticker
import numpy as np
import statistics
import math
from math import log10, floor
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

def lire_parametres(nom_fichier_para):
    list_true = ["True","true","Yes","yes","Oui","oui"]
    list_false= ["False","false","No","no","Non","non"]
    para = []
    para_final = []
    with open(nom_fichier_para, 'r') as fichier:
        n_ligne = 0
        for ligne in fichier:
            # Supprime les espaces en début et fin de ligne et les sauts de ligne
            ligne = ligne.strip()
            if ligne != "":
                # Sépare la ligne en utilisant la virgule comme séparateur
                elements = ligne.split(':')
                # Ajoute la liste d'éléments à la liste principale
                para.append(elements)
            else:
                para.append(ligne)
            n_ligne+=1

    i = 0
    noms_csv = []
    while para[i] != '':
        #print(para[i])
        noms_csv.append(":".join(para[i]))
        i+=1
        
        
    i+=1
    while i < n_ligne:
        #print("para[i]:",para[i])
        donnée = para[i][1].split(';')
        #print("donnée",donnée)
        #print("len(donnée)",len(donnée))
        for j in range(len(donnée)):

            for true in list_true:#On transforme les charatère en valeur booléenne True ou False et on prend en compte différentes écritture possible
                if true in str(donnée[j]):
                    donnée[j] = True
            for false in list_false:
                if false in str(donnée[j]):
                    donnée[j] = False
            if '[]'  in str(donnée[j]):
                donnée[j] = []
            elif '[' in str(donnée[j]): #On recréer les liste de liste
                liste = donnée[j].split('[')[1]
                liste = liste.split(']')[0]
                liste = liste.split(',')
                for k in range(len(liste)):
                    if str_is_numeric(liste[k]):
                        liste[k] = converti_en_chiffre(liste[k])
                donnée[j] = liste
            if str_is_numeric(donnée[j]):# on convertie les string en chiffre
                donnée[j] = converti_en_chiffre(donnée[j])
            elif type(donnée[j]) == str:
                if '\\n' in donnée[j]:
                    print(donnée[j])
                    donnée[j] = donnée[j].replace('\\n', '\n')

        
            
        
        if len(donnée) == 1:
            donnée = donnée[0]
            
        para_final.append([para[i][0],donnée])
        i+=1

    #print("noms_csv",noms_csv)
    para_reformater = []
    noms_csv_reformater = []
    for i in range(len(para_final)):
        para_reformater +=para_final[i]
    for i in range(len(noms_csv)):
        noms_csv_reformater += noms_csv[i]
    return noms_csv, para_reformater

def str_is_numeric(string):
    try:
        string = string.replace(' ', '')
        string = string.replace('.', '')
        string = string.replace(',', '')
        return string.isnumeric()
    except:
        return False

def converti_en_chiffre(string):
    if str_is_numeric(string):
        string = string.replace(' ', '')
        if '.' in str(string):
            return float(string)
        else:
            return int(string)
    else:
        return "ERROR"
        

def commencer_à_0_MPA(liste_deplacement, liste_contrainte, longueur_initiale=100, zéro=0):
    deplacement = np.array(liste_deplacement)
    contrainte = np.array(liste_contrainte)

    offset_deplacement = 0
    cnt0 =  liste_contrainte[0]
    i=1
    cnt1 = liste_contrainte[i]
    while cnt1 <zéro:
        cnt0 = cnt1
        i+=1
        cnt1 = liste_contrainte[i]

    A = [ liste_deplacement[i-1], liste_contrainte[i-1] ]
    B = [ liste_deplacement[i],   liste_contrainte[i] ]
    offset_deplacement = interpolation( A,B,zéro)

    deplacement -= offset_deplacement
    longueur_initiale += offset_deplacement

    deformation = deplacement/longueur_initiale*100

    return list(deformation)

def commencer_à_0_N(liste_deplacement,liste_force, zéro=0.0005):
    #return liste_deplacement

    np_deplacement = np.array(liste_deplacement)
    force = np.array(liste_force)
    if liste_force[0]==zéro: #si on est déja au dela de 0 en force
        offset_deplacement = liste_deplacement[0]
        
    else:
            
        offset_deplacement = zéro
        f0 =  liste_force[0]
        i=1
        f1 = liste_force[i]
        while f1 <zéro:
            f0 = f1
            i+=1
            f1 = liste_force[i]

        A = [ liste_deplacement[i-1], liste_force[i-1] ]
        B = [ liste_deplacement[i],   liste_force[i] ]
        while B[1] == A[1]:
            i+=1
            B = [ liste_deplacement[i],   liste_force[i] ]
        offset_deplacement = interpolation( A,B,zéro)
        #print("offset_deplacement",offset_deplacement)
    np_deplacement = np_deplacement - offset_deplacement

    #longueur_initiale += offset_deplacement

    return list(np_deplacement), offset_deplacement


def interpolation( A,B,y_cible):
    if (B[0]-A[0]) != 0:
        a= (B[1]-A[1]) / (B[0]-A[0])
    else:
        global inf
        a= inf
    b = A[1] - a*A[0]
    
    x_cible = (y_cible - b) / a
    return x_cible
     
def Contrainte_max(liste_deformation,liste_contrainte):
    max_index = liste_contrainte.index(max(liste_contrainte))
    max_point = [liste_deformation[max_index], liste_contrainte[max_index]]
    return max_point

def Point_Y_max(liste_X,liste_Y):
    max_index = liste_Y.index(max(liste_Y))
    max_point = [liste_X[max_index], liste_Y[max_index]]
    return max_point

def ecrire_liste_dans_csv(liste, nom_fichier):
    with open(nom_fichier, 'w') as fichier:
        for element in liste:
            ligne = ";".join(map(str, element)) + "\n"
            fichier.write(ligne)
    print("raport écris dans",nom_fichier)
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
def calcule_Rp02_sans_E(liste_contrainte, liste_deformation, contrainte_max,Rp=0.2,calcule_de="E"):
    
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

    #print("E ou K",E)
    #print("Rp",Rp)
    #print("liste_contrainte:",np.array(liste_contrainte))
    #print("liste_deformation:",np.array(liste_deformation))

    eps02, cnt02 = calcule_Rp02_avec_E(liste_contrainte, liste_deformation,E,Rp=Rp,calcule_de=calcule_de)
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
def calcule_Rp02_avec_E(liste_contrainte, liste_deformation,E,Rp=0.2,calcule_de="E"): #def en %, cnt en MPa, E en MPa, RP en %
    cnt=liste_contrainte
    eps=liste_deformation
    E = (E/100)
    Rp = float(Rp)
    #print("len(liste_contrainte)",len(liste_contrainte))
    
    for i in range(len(liste_contrainte)):
        cnt02 = E*eps[i] - E*Rp
        if cnt[i] < cnt02:
            P11 = [Rp , 0]
            P12 = [Rp+1 , E]
            P21 = [eps[i-1] , cnt[i-1]]
            P22 = [eps[i] , cnt[i]]

            eps02, cnt02 = intersection_droites( P11, P12, P21, P22)

            #print("eps02, cnt02",eps02, cnt02)
            return eps02, cnt02
        elif cnt[i] == cnt02:
            #print("eps[i] , cnt[i]",eps[i] , cnt[i])
            return eps[i] , cnt[i]
        else:
            print("/!\\ calcule_Rp02_avec_E pas réaliser,la valeur de",calcule_de,"est fausse /!\\")
            return 0,0 

def calcule_Rp02_adaptatif(liste_contrainte, liste_deformation,Rp=0.2,force_moyenne=15,seuille_linéarité=10,debug=False): #def en %, cnt en MPa, en MPa, RP en %, seuille_linéarité en %
    #on decale la deformation quand elle est stabiliser
    #Rp=0
    cnt=liste_contrainte
    eps=liste_deformation
    E, deps, dcnt = calcule_E_adaptatif(eps, cnt, force_moyenne=force_moyenne, return_derivé=True,debug=debug)
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
    if debug:
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
    if debug:
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
#def Calcule_point_inflection(X,Y,force_moyennage=15):
    
def extraire_donee_par_eprouvettes(donnee, nb_eprouvette,taille_entete):
    liste_deplacement = []
    liste_force = []
    liste_contraintes = []
    liste_deformation = []
    liste_eprouvette_vide = []
    liste_autre = []

    nb_eprouvette_reel = nb_eprouvette
    numéro_eprouvette = 1
    ligne = taille_entete+3
    while numéro_eprouvette <= nb_eprouvette: #on iterre sur toute les eprouvettes
        #print("numéro_eprouvette:",numéro_eprouvette)
        liste_contraintes.append([])
        liste_deformation.append([])
        liste_deplacement.append([])
        liste_force.append([])
        liste_autre.append([])
        #print("donnee[ligne]: ",donnee[ligne])
        while donnee[ligne][0] == '' and len(donnee[ligne])>1 :
            liste_deplacement[-1].append(float(donnee[ligne][2]))
            liste_force[-1].append(      float(donnee[ligne][3]))
            liste_contraintes[-1].append(float(donnee[ligne][4]))
            liste_deformation[-1].append(float(donnee[ligne][5]))
            liste_autre[-1].append(donnee[ligne][6:])
            
            ligne += 1
            #print("donnee[ligne]: ",donnee[ligne])
        if numéro_eprouvette != nb_eprouvette: #si on est pas a la fin des eprouvettes
            ligne += 2
        if liste_contraintes[-1] == []: #si il n'y avait aucune donnée dans l'eprouvette
            print("/!\\ L'éprouvette n°",numéro_eprouvette,"n'as aucune donnée! /!\\")
            del liste_deplacement[-1] #on enleve toutes les liste inutiles
            del liste_force[-1]
            del liste_contraintes[-1]
            del liste_deformation[-1]
            del liste_autre[-1]
            nb_eprouvette_reel -= 1
            liste_eprouvette_vide.append(numéro_eprouvette)
        numéro_eprouvette += 1

    if nb_eprouvette_reel != nb_eprouvette:
        print("Nombre d'éprouvette avec des données:",nb_eprouvette_reel)
    return liste_deplacement, liste_force, liste_contraintes, liste_deformation, nb_eprouvette_reel, liste_eprouvette_vide, liste_autre

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

def calculer_E_a_partir_de_la_ISO_527(liste_déformation, liste_contrainte,debug=False):
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
        if debug:
            print([eps1,sig1] , [eps2,sig2])
        E = coefficient_directeur( [eps1/100,sig1] , [eps2/100,sig2] )
        return E
    return None

def calcule_E_adaptatif(liste_déformation, liste_contrainte, delta_dcnt=0.25, force_moyenne=15, return_derivé=False,debug=False): # eps (%) ; cnt (MPa)
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
    while (dY_moy[imin] > dY_max-delta_dcnt) and (imin -1>=0):
        imin -= 1

    imax = i_dY_max
    while (dY_moy[imax] > dY_max-delta_dcnt) and (imax+1<len(dY_moy)):
        imax += 1

    n_val = ( imax - imin +1)
    E = 0
    for i in range( imin, imax+1):
        E += dY_moy[i]
    E = E/n_val
    if debug:
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
def combiner_plusieurs_échantillons(list_nom_csv,nom_export):
    list_donnee = []
    champ1 = []
    champ2 = []
    list_fichier_finale =[]
    for i in range(len(list_nom_csv)):
        donnee= lire_csv_sans_lib(list_nom_csv[i])
        
        list_donnee.append(donnee)
        taille_entete=3 # en comptant a partir de 0
        
        deb_champ1 =3 # en comptant a partir de 0
        fin_champ1 = deb_champ1
        nb_eprouvette = 1

        while donnee[fin_champ1+1][0].isnumeric() == True:
            fin_champ1+=1
            nb_eprouvette += 1
        #print(fin_champ1)
            
        if i == 1: #si on a encore rien écris
            list_fichier_finale += donnee[:3]
            tab_resul2 = donnee[fin_champ1+2:fin_champ1+6]
        champ1 += donnee[3:fin_champ1+1]
        champ2 += donnee[fin_champ1+6:-1]



    moyenne = [''] * len(champ1[0])
    moyenne[0] = 'Moyenne'
    for col in range(len(champ1[0])):
        if col>1:
            somme = 0
            for ligne in range(len(champ1)):
                somme += float(champ1[ligne][col])
            moyenne[col] = str(somme/len(champ1))

    list_fichier_finale += champ1
    list_fichier_finale += [moyenne]
    list_fichier_finale += tab_resul2
    list_fichier_finale += champ2
    list_fichier_finale += [[''],['']]
    #print("champ1:\n",champ1)
    #print("moyenne:\n",moyenne)
    #print("\n\n")
    #or j in list_fichier_finale:
        #print(j)
    #print("\n\n")
    #print("moyenne:\n",moyenne)


    for i in range(len(list_fichier_finale)):
        list_fichier_finale[i] = ';'.join(list_fichier_finale[i])
    str_sortie = '\n'.join(list_fichier_finale)

    f = open(nom_export, "w")
    f.write(str_sortie)
    f.close()
    
    return list_fichier_finale
        
    
def round_to_1(x): #arondi a 1 chiffre significatif
    #https://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
    return round(x, -int(floor(log10(abs(x)))))

def base_10(x):
    return 10**( int(floor(log10(abs(x)))))
def round_to_base(x, base=5): #arondi au multiple de 5 le plus proche
    #https://stackoverflow.com/questions/2272149/round-to-5-or-other-number-in-python
    base *= base_10(x)
    rnd = base * round(x/base)
    if rnd != 0:
        return rnd
    else:
        return 10**( int(floor(log10(abs(x)))))

def calculer_contrainte(list_force, section,unite=1000): #unite= multiple de 1N, ici 1000 veut dire des kN. Les sections sont considerer en mm²
    list_cnt=[]
    for f in list_force:
        list_cnt.append( (f*unite)/section )
    return list_cnt

def calculer_deformation(list_deplacement, L0):# en mm,  L0 est la longeur initiale
    list_def=[]
    for dep in list_deplacement:
        list_def.append( dep / L0 *100) # *100 pour mettre en %
    return list_def
        
        
def extraire_nom_du_csv(nom_csv):
    #print("nom_csv",nom_csv)
    nom_csv = nom_csv.split("/")
    nom_csv = nom_csv[-1]
    nom_csv = nom_csv.split(".")
    nom = nom_csv[0]
    
    return str(nom)

def moyenne_courbe(list_X, list_Y, degré_d_extrapolation=0): #list_X = [ [x11,x12,x13,...],[x21,x22,x23,...],[x31,x32,x33,...] ], degré_d_extrapolation (%) 
    #basé  sur: https://stackoverflow.com/questions/51933785/getting-a-mean-curve-of-several-curves-with-x-values-not-being-the-same
    xs = list_X
    ys = list_Y

    pas_x = float(xs[0][1] - xs[0][0])

    nb_mini_courbes_moyener = round( len(xs) - len(xs)*(degré_d_extrapolation/100))

    list_xmin = []
    list_xmax = []
    for i in range(len(xs)):
        list_xmin.append( np.min(xs[i]) )
        list_xmax.append( np.max(xs[i]) )
    list_xmin.sort()
    list_xmax.sort()
    
    min_xs = list_xmin[nb_mini_courbes_moyener-1]
    max_xs = list_xmax[-nb_mini_courbes_moyener]

    """ 
    if ne_pas_trop_extrapoler == True: #on fait la moyenne que là où on a toute les courbes qui se chevauche
        list_xmin = []
        list_xmax = []
        for i in range(len(xs)):
            list_xmin.append( np.min(xs[i]) )
            list_xmax.append( np.max(xs[i]) )
        min_xs = max(list_xmin)
        max_xs = min(list_xmax)
    elif ne_pas_trop_extrapoler == False:
        max_xs = xs[0][0]
        min_xs = xs[0][-1]
        for i in range(len(xs)):
            max_i = np.max(xs[i])
            min_i = np.min(xs[i])
            if max_i > max_xs:
                max_xs = float(max_i)
            if min_i < min_xs:
                min_xs = float(min_i)
    """
    #print("pas_x",pas_x)

    mini = math.floor(min_xs/pas_x)
    maxi = math.ceil(max_xs/pas_x)
    #print("mini",mini)
    #print("maxi",maxi)
    moy_x = list( np.array(range(mini, maxi, 1)) *pas_x )
    
    #mean_x_axis = [i for i in range(max(xs))]
    #ys_interp = [np.interp(mean_x_axis, xs[i], ys[i]) for i in range(len(xs))]
    ys_interp = [np.interp(moy_x, xs[i], ys[i]) for i in range(len(xs))]
    mean_y_axis = np.mean(ys_interp, axis=0)

    return(moy_x, mean_y_axis)

def plot_to_csv(list_x, list_y):
    arr = np.array([np.array(list_x),np.array(list_y)])
    arr = list(arr.T) #transposition de la matrice
    for i in range(len(arr)):
        arr[i] = list(arr[i])
    return arr

def dériver_brut(X,Y):
    dX = X[:-1]
    dY=[]
    for i in range(len(Y2)-1):
        dY.append((Y[i+1] - Y[i]) /(X[i+1] - X2i]))
    return(dX, dY)
    
        
def trouver_point_inflection(X,Y,force_moyenne=15):
    points_inflections = []
    pente_au_points_inflection = []

    dX, dY = dériver_brut(X,Y)#dériver 1ere
    dX_moy, dY_moy = moyenner_courbe(dX, dY, force=force_moyenne)

    ddX, ddY = dériver_brut(dX_moy, dY_moy)# dériver seconde
    ddX_moy, ddY_moy = moyenner_courbe(ddX, ddY, force=force_moyenne)
    
    val_prec = ddY_moy[0]
    for i in range(len(ddY_moy)-1): #On cherche la valeur en X des points d'inflections
        if ddy_moy[i] == 0:
            points_inflections.append([ddX_moy])
        elif math.copysign(1, ddY_moy[i]) != math.copysign(1, ddY_moy[i+1]): #test si les signes des deux valeurs sont différents
            A = (ddX_moy[i],ddY_moy[i]))
            B = (ddX_moy[i+1],ddY_moy[i+1]))
            ddX0=interpolation( A,B,0)
            points_inflections.append([ddX0])

    num_pnt_inflection = 0
    i=0
    while num_pnt_inflection < len(points_inflections): #On cherche la valeur en Y des points d'inflections
        x_infl = points_inflections[num_pnt_inflection][0]
        if X[i] == x_infl:
            points_inflections[num_pnt_inflection].append(Y[i])
            num_pnt_inflection += 1
            
        elif: (X[i] < x_infl) and (X[i+1] > x_infl):
            A = (Y[i],X[i]))
            B = (Y[i+1],X[i+1]))
            y_infl=interpolation( A,B,x_infl)
            points_inflections[num_pnt_inflection].append(y_infl)
            num_pnt_inflection += 1
        i+=1

    num_pnt_inflection = 0
    i=0
    while num_pnt_inflection < len(points_inflections): #On cherche la valeur de la pente(dy) au niveau des points d'inflections
        x_infl = points_inflections[num_pnt_inflection][0]
        if dX_moy[i] == x_infl:
            pente_au_points_inflection.append(dY_moy[i])
            num_pnt_inflection += 1
            
        elif: (dX_moy[i] < x_infl) and (dX_moy[i+1] > x_infl):
            A = (dY_moy[i],dX_moy[i]))
            B = (dY_moy[i+1],dX_moy[i+1]))
            y_infl=interpolation( A,B,x_infl)
            points_inflections.append(y_infl)
            num_pnt_inflection += 1
        i+=1
            
    return points_inflections #liste [[X,Y,pente], [],...]
        
def recallage_donner_par_point_inflection(X,Y):
    points_inflections = trouver_point_inflection(X,Y)

    premier_point_infl = points_inflections[0][:2] #on prend les deux première valeurs (X et Y)
    pente_premier_infl = points_inflections[0][2]
    
    
def depouiller_essais_traction_simple(Nom_csv, Parametres):
    """
    #création de la palette de couleurs utiliser dans les graphique
    palette1 = [ 'navy','mediumblue', 'dodgerblue', 'deepskyblue', 'royalblue','tab:blue','cornflowerblue','lightskyblue','lightsteelblue','lavender'] # couleurs utiliser sur le graph
    palette2 = [ 'lightcoral','brown','firebrick','maroon','tomato','orangered','salmon','coral','indianred','darkred','tab:red']
    Palette = [palette1, palette2]
    Palette = [['dodgerblue'], ['tomato'],['tab:green']]
    

    cross_Palette = [['crimson','red'],
                     ['b','tab:blue'],
                     ['gold','orange']]
    #for style in ['classic', 'seaborn-v0_8-bright', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white']:
    #for style in plt.style.available:
        #print("style:",style)
    """

    #print("Nom_csv",Nom_csv)
    window_size = ['', '', '', '']
    fig, ax = plt.subplots()
    if type(Nom_csv) == list:
        nb_essais = len(Nom_csv)
    else:
        nb_essais = 1

    for i in range(nb_essais):
        if nb_essais != 1:
            nom_csv = Nom_csv[i]
            parametres = Parametres[i]
        else:
            nom_csv = Nom_csv
            parametres = Parametres
        nom_csv = Nom_csv[i]
        parametres = Parametres[i]
        nom_echantillon = extraire_nom_du_csv(nom_csv)

        print("\n\n============---",nom_echantillon,"---============\n")
        donnee = lire_csv_sans_lib(nom_csv ) #ouverture du fichier .csv exporter depuis la machine de traction
        

        nom_para = parametres[::2] # séparation des description de parametres et de leur valeur
        val_para = parametres[1::2]
        montrer_N_eprouvette = val_para[0]
        centrer_deformation_à_0 = val_para[1]
        Rp = val_para[2]
        montrer_que_certaines_eprouvettes = val_para[3]
        Longueur_éprouvette  = val_para[4]
        debug = val_para[5]
        montrer_deriver = val_para[6]
        force_depla = val_para[7] #si vrai, affiche diagrame force/déplacement, si faux affiche diagrame contrante/déformation
        affich_cnt_max = val_para[8]
        affich_rp02 = val_para[9]
        mode_affichage_courbe = val_para[10]
        mettre_fleche = val_para[11]
        titre = str(val_para[12])
        bas_arondi = val_para[13]
        aff_moy_cnt_max = val_para[14]
        montrer_N_échantillon = val_para[15]
        montrer_description_échantillon = val_para[16]
        description_echantillon = val_para[17]
        afficher_courbe_moyenne = val_para[18]

        

        montrer_que_certaines_eprouvettes = montrer_que_certaines_eprouvettes[i]

        X_taille_fleche = 10#% de la taille de la fenètre
        Y_taille_fleche = 15#% de la taille de la fenètre

        """
        montrer_N_échantillon = False
        if montrer_N_eprouvette == True and mode_affichage_courbe in [2,4]:
            montrer_N_échantillon = True
            montrer_N_eprouvette = False
        """
        #création de la palette de couleurs utiliser dans les graphique
        Palette_moy=[]
        if mode_affichage_courbe == 1:
            palette1 = [ 'navy','mediumblue', 'dodgerblue', 'deepskyblue', 'royalblue','tab:blue','cornflowerblue','lightskyblue','lightsteelblue','lavender'] # couleurs utiliser sur le graph
            palette2 = [ 'lightcoral','brown','firebrick','maroon','tomato','orangered','salmon','coral','indianred','darkred','tab:red']
            Palette = [palette1, palette2]
            cross_Palette = [['crimson','red'],['b','tab:blue'],['gold','orange']]

            style = '-'
            largeur = 1
            
        elif mode_affichage_courbe == 2:
            Palette = [['b'], ['crimson'],['tab:green'],["tab:brown"]]
            a = 0.2 #alpha chanel, (transparence)
            Palette =     [[(0, 0, 1,a)], [(220/255, 20/255, 60/255,a)],[(44/255, 160/255, 44/255,a)],[(243/255, 59/255, 238/255,a)],[(0,0,0,a)],[(222/255, 255/255, 0/255,a)]] #RGB avec valeurs entre 0 et 1
            Palette_moy = [[(0, 0, 1,1)], [(220/255, 20/255, 60/255,1)],[(44/255, 160/255, 44/255,1)],[(243/255, 59/255, 238/255,1)],[(0,0,0,1)],[(222/255, 255/255, 0/255,1)]]
            cross_Palette = Palette_moy
            
            style = 'dotted'
            style='-'
            largeur = 0.7
            
        elif mode_affichage_courbe == 3:
            Palette = [['b'], ['crimson'],['tab:green'],["tab:brown"]]
            cross_Palette = Palette
            
            style = 'dotted'
            largeur = 0.01
            
        elif mode_affichage_courbe == 4:
            Palette =[['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']]
            cross_Palette = [['crimson','red'],['b','tab:blue'],['gold','orange']]
            style = '-'
            largeur = 1.2

        

        style_moy = '-.'
        largeur_moy = 1.5




        n_palette = i%len(Palette)
        palette = Palette[n_palette]
        if Palette_moy != []:
            palette_moy = Palette_moy[n_palette]

        n_cross_palette = i%len(cross_Palette)
        cross_palette = cross_Palette[n_cross_palette]

        #extraction de la donnée de la géométrie des eprouvettes
        j = 0
        while donnee[j] != ['']:#on cherche le premier blanc dans le fichier, qui sert de délimiteur
            j+=1

        if i> len(description_echantillon):
            description_echantillon = donnee[0][2]
        else:
            description_echantillon= description_echantillon[i]

        description_ech_pousser = False
        if description_echantillon[-1] == '¤':
            description_echantillon = description_echantillon[:-1]
            description_ech_pousser= True
            info_sup = []
       
            
        donnee_géométrie = donnee[1:j]
        donnee = donnee[j+1:]

        

        #determination de la taille de l'entette et du nombre d'eprouvettes
        taille_entete=3 # en comptant a partir de 0
        nb_eprouvette = 1

        while donnee[taille_entete+1][0].isnumeric() == True:
            #print("donnee[taille_entete+1][0] ",donnee[taille_entete+1][0])
            taille_entete+=1
            nb_eprouvette += 1

        taille_entete += 5

        if debug:
            print("\nnombre d'éprouvettes: ",nb_eprouvette)
        print("donnee_géométrie:",donnee_géométrie)
        #extraction de la géométrie de chaque éprouvette
        if donnee_géométrie[0][2]=='Rectangulaire':
            if len(donnee_géométrie[1]) == 3: # epaisseur constante sur tout l'échantillon
                epaisseur_ep = [float(donnee_géométrie[1][2])] * nb_eprouvette
                print("epaisseur constante sur tout l'échantillon")
            else:#                             epaisseur différente pour chaque eprouvettes
                epaisseur_ep = donnee_géométrie[1][2:]
                for j in range(len(epaisseur_ep)):
                    epaisseur_ep[j] = float(epaisseur_ep[j])
                if len(epaisseur_ep) != nb_eprouvette:
                    print("/!\\ LE NOMBRE DE VALEUR D'EPAISSEURS NE CORESPOND PAS AU NOMBRE D'EPROUVETTES")
                    return

            if len(donnee_géométrie[2]) == 3: # largeur constante sur tout l'échantillon
                largeur_ep = [float(donnee_géométrie[2][2])] * nb_eprouvette
            else:#                             largeurr différente pour chaque eprouvettes
                largeur_ep = donnee_géométrie[2][2:]
                for j in range(len(largeur_ep)):
                    largeur_ep[j] = float(largeur_ep[j])
                if len(largeur_ep) != nb_eprouvette:
                    print("/!\\ LE NOMBRE DE VALEUR DE LARGEUR NE CORESPOND PAS AU NOMBRE D'EPROUVETTES")
                    return

            if len(donnee_géométrie[3]) == 3: # longueur constante sur tout l'échantillon
                longueur_ep = [float(donnee_géométrie[3][2])] * nb_eprouvette
            else:#                             longueur différente pour chaque eprouvettes
                longueur_ep = donnee_géométrie[3][2:]
                for j in range(len(longueur_ep)):
                    longueur_ep[j] = float(longueur_ep[j])
                if len(longueur_ep) != nb_eprouvette:
                    print("/!\\ LE NOMBRE DE VALEUR DE LONGEUR NE CORESPOND PAS AU NOMBRE D'EPROUVETTES")
                    return
            section_ep=[]
            for j in range(nb_eprouvette):
                section_ep.append( round( epaisseur_ep[j] * largeur_ep[j] , 2))
        
        if donnee_géométrie[0][2]=="Circulaire":
            if len(donnee_géométrie[1]) == 3: # Diamètre constante sur tout l'échantillon
                diamètre_ep = [float(donnee_géométrie[1][2])] * nb_eprouvette
                print("epaisseur constante sur tout l'échantillon")
            else:#                             epaisseur différente pour chaque eprouvettes
                diamètre_ep = donnee_géométrie[1][2:]
                for j in range(len(diamètre_ep)):
                    diamètre_ep[j] = float(diamètre_ep[j])
                if len(diamètre_ep) != diamètre_ep:
                    print("/!\\ LE NOMBRE DE VALEUR DE DIAMETRES NE CORESPOND PAS AU NOMBRE D'EPROUVETTES")
                    return

            if len(donnee_géométrie[2]) == 3: # longueur constante sur tout l'échantillon
                longueur_ep = [float(donnee_géométrie[2][2])] * nb_eprouvette
            else:#                             longueur différente pour chaque eprouvettes
                longueur_ep = donnee_géométrie[2][2:]
                for j in range(len(longueur_ep)):
                    longueur_ep[j] = float(longueur_ep[j])
                if len(longueur_ep) != nb_eprouvette:
                    print("/!\\ LE NOMBRE DE VALEUR DE LONGEUR NE CORESPOND PAS AU NOMBRE D'EPROUVETTES")
                    return
            section_ep=[]
            for j in range(nb_eprouvette):
                section_ep.append( round( math.pi * (diamètre_ep[j]/2)**2  , 2))
                
            
        #print("Sections:",section_ep)
            

        #Extraction des donées pour chaque eprouvettes:
        liste_deplacement, liste_force, liste_contraintes, liste_deformation, nb_eprouvette_reel, liste_eprouvette_vide, liste_autre = extraire_donee_par_eprouvettes(donnee, nb_eprouvette, taille_entete)

        

        #création de la liste d'élémentexporter dan le rapport
        export_data=[["Longueur entre la sortie des mors (mm)",Longueur_éprouvette],
                     [""],
                     ["n°éprouvette","nom éprouvette","Vitesse d'essai","Contrainte max","Deformation à contrainte max","Module de Young machine","Module de Young 'adaptatif'","Module de Young ISO 527","Rp0,2"],
                     ["","",donnee[2][2],donnee[2][3],donnee[2][4],donnee[2][5],donnee[2][5],donnee[2][5],donnee[2][5]]]
        
        


        #affichage des courbes sur un graphique   
        
        #plt.style.use(style)

        list_point_cnt_max = []
        list_cnt_max = []
        list_def_cnt_max = []

        list_point_for_max = []
        list_for_max = []
        list_dep_for_max = []
        
        list_rp02 = []
        l_rp02 = []
        ep_rp02 = []
        list_E = [] #liste des Module d'Young en N/mm² (MPa)
        list_K = [] #liste des raideurs en N/mm
        
        
        """
        #centrer_deformation_à_0 = input("\nDeplacer les deformation en 0? [ENTER / N] ")#demande à l'utilisateur si il veut décaler l'axe des déformation pour compensser la mise en tension du filament au début de l'essais
        if centrer_deformation_à_0 == "n" or centrer_deformation_à_0 == "N":
            centrer_deformation_à_0 = False
        else:
            centrer_deformation_à_0 = True
        """

        List_X = []
        List_Y = []
        
        
        if debug:
            print("liste_eprouvette_vide",liste_eprouvette_vide)
            print("nb_eprouvette:",nb_eprouvette)
        nom_echantillon_déja_montrer = False
        ep_sauter = 0 #on saute des éprouvette si elle n'a aucune donnée (c'est déja arrivé)
        for ep in range(nb_eprouvette):
            if debug:
                print("\nep",ep)
            if not( (ep+1) in liste_eprouvette_vide ):#on vérifie que l'éprouvette à des données associées
                if (montrer_que_certaines_eprouvettes == []) or (ep+1 in montrer_que_certaines_eprouvettes): #on verifie le formatage et si on dois géré l'éprouvette actuelle
                    if debug:
                        print("n° eprouvette:",ep+1)

                    #style visuel de la courbe
                    k = ep%len(palette)
                    couleur = palette[k]
                    if Palette_moy != []:
                        couleur_moy = palette_moy[k]
                    else:
                        couleur_moy = palette[k]

                    offset_deplacement=0
                    if centrer_deformation_à_0: # "reprise de moue"
                        deplacement,offset_deplacement = commencer_à_0_N(liste_deplacement[ep-ep_sauter],liste_force[ep-ep_sauter])
                        #deformation = commencer_à_0_MPA(liste_deformation[ep-ep_sauter],liste_contraintes[ep-ep_sauter],longueur_initiale=Longueur_éprouvette) 
                    else:
                        deplacement = liste_deplacement[ep-ep_sauter]
                        #deformation = liste_deformation[ep-ep_sauter]
                    #print("longueur_ep",longueur_ep)
                    #print("ep",ep)
                    L0 = longueur_ep[ep] + offset_deplacement
                    
                    deformation = calculer_deformation(deplacement, L0)
                    #print("section_ep[ep-ep_sauter]",section_ep[ep-ep_sauter])
                    contrainte  = calculer_contrainte(liste_force[ep-ep_sauter], section_ep[ep-ep_sauter])# Contrainte = Force/Section (force kN, section en mm²)

                    if force_depla == False: # Contrainte/Déformation
                        X = np.array(deformation)
                        #Y = np.array(liste_contraintes[ep-ep_sauter])
                        Y = np.array(contrainte)
                    else: #Force/Déplacement
                        X = np.array(deplacement)
                        Y = np.array(liste_force[ep-ep_sauter])
                        print("\n",donnee[3+ep][1])
                        print("max Y:",max(Y))
                        print("max X:",X[ list(Y).index(max(Y)) ])

                    List_X.append(X)
                    List_Y.append(Y)

                    
                        
                    cnt_max=Point_Y_max(deformation,contrainte)
                    for_max=Point_Y_max(deplacement,liste_force[ep-ep_sauter])
                    
                    list_point_cnt_max.append(cnt_max)
                    list_cnt_max.append(cnt_max[1])
                    list_def_cnt_max.append(cnt_max[0])

                    list_point_for_max.append(for_max)
                    list_for_max.append(for_max[1])
                    list_dep_for_max.append(for_max[0])

                    ligne_entete = 3+ep
                    E_machine = donnee[ligne_entete][5]
                    E_ISO_527 = calculer_E_a_partir_de_la_ISO_527(deformation, contrainte,debug=debug)
                    
                    #Calcule du Rp0.2 et E
                    #print("len(deformation)",len(deformation))
                    #print("cnt_max[1]",cnt_max[1])
                    E_adaptatif = calcule_E_adaptatif(deformation, contrainte,debug=debug)
                    if debug:
                        print("len(liste_contraintes[ep-ep_sauter])",len(contrainte))
                        print("len(deformation)",len(deformation))
                        print("cnt_max[1]",cnt_max[1])
                        print("Rp",Rp)
                    eps02, rp02, E=calcule_Rp02_sans_E(contrainte, deformation, cnt_max[1], Rp=Rp)
                    #eps02, rp02 =  calcule_Rp02_avec_E(liste_contraintes[ep-ep_sauter], deformation, E_ISO_527)
                    
                    eps02, rp02 = calcule_Rp02_avec_E(contrainte, deformation, E_adaptatif)
                    eps02, rp02 = calcule_Rp02_adaptatif(contrainte, deformation, Rp=Rp,debug=debug)
                    
                    list_rp02.append([eps02, rp02])
                    l_rp02.append(rp02)
                    ep_rp02.append(eps02)
                    
                    list_E.append(E)
                    if debug:
                        print("nb de points de l'éprouvette",len(contrainte))
                        print("E (machine de traction)",E_machine)
                        print("E ISO 527",E_ISO_527)
                        print("E_adaptatif",E_adaptatif)
                        print("E (Rp0.2)",E)
                        print("for_max[1]",for_max[1])

                        
                    dep02, rpf02, K =calcule_Rp02_sans_E(deplacement, liste_force[ep-ep_sauter], for_max[1], Rp=Rp, calcule_de='K')
                    
                    list_K.append(K/100)
                    #determinasion de la taille de la fenetre d'affichage:
                    X = X.tolist() #conversion du numpy array en list
                    Y = Y.tolist()
                    if (window_size[0] == '') or (min(X) < window_size[0]):
                        window_size[0] = min(X)
                    if (window_size[2] == '') or (min(Y) < window_size[2]):
                        window_size[2] = min(Y)
                    if (window_size[3] == '') or (max(Y) > window_size[3]):
                        window_size[3] = max(Y)

                    if ((window_size[1] == '') or
                        (max(X) > window_size[1]) and Y[X.index(max(X))]>max(Y)*0.1):
                        window_size[1] = max(X)
                    elif X[Y.index(max(Y))] > window_size[1]:
                        window_size[1] = X[Y.index(max(Y))]

                    
                    #AFFICHAGE DE LA COURBE LIER A L'EPROUVETTE
                    if (nom_echantillon_déja_montrer==False) and mode_affichage_courbe != 3:
                        nom = []
                        if montrer_N_échantillon == 1:
                            nom.append( ''.join(['ech n°',nom_echantillon]) )
                        """
                        if montrer_description_échantillon:
                            nom.append( ''.join(['',description_echantillon]))
                        """
                        nom = ', '.join(nom)
                        
                        if montrer_N_échantillon :#or montrer_description_échantillon:
                            nom_echantillon_déja_montrer = True
                            plt.plot(X, Y, color=couleur, linestyle=style, linewidth=largeur ,label=nom)
                        
                        montrer_description_échantillon
                            
                    if montrer_N_eprouvette==1:
                        nom = ''.join([nom_echantillon,'-',str(ep+1)])
                        plt.plot(X, Y, color=couleur, linestyle=style, linewidth=largeur ,label=nom)
                    elif montrer_N_eprouvette==2:
                        ligne_entete = 3+ep
                        nom = donnee[ligne_entete][1]
                        plt.plot(X, Y, color=couleur, linestyle=style, linewidth=largeur ,label=nom)
                    else:
                        plt.plot(X, Y, color=couleur, linestyle=style, linewidth=largeur)

                    
                    #on met dans le rapport les élément de l'entete qui nous interesse
                    
                    export_data.append([ donnee[ligne_entete][0], donnee[ligne_entete][1], donnee[ligne_entete][2], cnt_max[1], cnt_max[0], E,E_adaptatif, E_ISO_527, rp02])

            else:
                ep_sauter +=1

        
        
        #plt.axis([-2, 20, -3, 65])
        moy_X,moy_Y = moyenne_courbe(List_X,List_Y)
        list_export_courbe_moyenne = plot_to_csv(moy_X,moy_Y)
        if afficher_courbe_moyenne == True:
            plt.plot(moy_X, moy_Y, color=couleur_moy, linestyle=style_moy, linewidth=largeur_moy) #affichage de la courbe moyenne
        

        l = ep%len(cross_palette)
        if force_depla == False: # Contrainte/Déformation
            if affich_cnt_max == True:
                #affichage des contraintes max pour chaque courbes
                nom_echantillon_déja_montrer = False
                for cnt_max in list_point_cnt_max:
                    if (mode_affichage_courbe == 3) and nom_echantillon_déja_montrer==False:
                        nom_echantillon_déja_montrer = True
                        nom = []
                        if montrer_N_échantillon :
                            nom.append( ''.join(['ech n°',nom_echantillon]) )
                        """if montrer_description_échantillon:
                            nom.append( ''.join(['',description_echantillon]))"""
                        nom = ', '.join(nom)
                        plt.plot(cnt_max[0],cnt_max[1], '+', color = cross_palette[l],label=nom)
                    else:
                        plt.plot(cnt_max[0],cnt_max[1], '+', color = cross_palette[l])

            #affichage des Rp0,2 pour chque courbes
            if affich_rp02 == True:
                for pnt in list_rp02:
                    plt.plot(pnt[0],pnt[1], '+', color = cross_palette[l])
        else:
            #affichage des Forces max pour chaque courbes
            for for_max in list_point_for_max:
                plt.plot(for_max[0],for_max[1], '+', color = cross_palette[l])

            
            



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
        export_data.append(['         -=-'])
        export_data.append(['eprouvette utiliser:',montrer_que_certaines_eprouvettes])
        export_data.append([''])
        export_data.append(["Moyenne de la contrainte max",moyenne_contrainte_max])
        export_data.append(["Ecart type",ecart_type_contrainte_max])
        export_data.append(['Dans   68% des cas Contrainte max €',round(moyenne_contrainte_max-1*ecart_type_contrainte_max, nb_ar) , round(moyenne_contrainte_max+1*ecart_type_contrainte_max, nb_ar) ,''.join(['+-',str(round(1*ecart_type_contrainte_max,nb_ar))]) ])
        export_data.append(['Dans   95% des cas Contrainte max €',round(moyenne_contrainte_max-2*ecart_type_contrainte_max, nb_ar) , round(moyenne_contrainte_max+2*ecart_type_contrainte_max, nb_ar) ,''.join(['+-',str(round(2*ecart_type_contrainte_max,nb_ar))]) ])
        export_data.append(['Dans 99.7% des cas Contrainte max €',round(moyenne_contrainte_max-3*ecart_type_contrainte_max, nb_ar) , round(moyenne_contrainte_max+3*ecart_type_contrainte_max, nb_ar) ,''.join(['+-',str(round(3*ecart_type_contrainte_max,nb_ar))])])
        export_data.append([''])
        export_data.append(['Mediane de la contrainte max',statistics.median(list_cnt_max)])

        if force_depla == False: # Contrainte/Déformation
            txt = ''.join([r"$\sigma_{max}$ ∈ [",str(round(moyenne_contrainte_max-2*ecart_type_contrainte_max, nb_ar))," ; ",str(round(moyenne_contrainte_max+2*ecart_type_contrainte_max, nb_ar)),'] à 95%'])
            if affich_cnt_max == True:
                #affichage moyenne contrainte maxaff_moy_cnt_max
                if mettre_fleche== True:
                    x_txt = moyenne_deformation_contrainte_max*1 + 6*ecart_type_deformation_contrainte_max
                    y_txt = moyenne_contrainte_max*1 + 4*ecart_type_contrainte_max
                    b=window_size[1]-window_size[0]
                    h=window_size[3]-window_size[2]
                    x_txt = moyenne_deformation_contrainte_max + b*X_taille_fleche/100
                    y_txt = moyenne_contrainte_max + h*Y_taille_fleche/100
                    
                    plt.text(x_txt, y_txt, txt)
                    p1 = patches.FancyArrowPatch( (moyenne_deformation_contrainte_max, moyenne_contrainte_max), (x_txt, y_txt),arrowstyle='<-', mutation_scale=20)
                    plt.gca().add_patch(p1)

                
                n=2
                xerr = [ecart_type_deformation_contrainte_max*n]# , ecart_type_deformation_contrainte_max*n]
                yerr = [ecart_type_contrainte_max*n]#, ecart_type_contrainte_max*n]
                if aff_moy_cnt_max ==True:
                    legende_moyenne = ''.join([r"$\overline{\sigma_{max}}$(",nom_echantillon,") à 95%"])
                    plt.errorbar([moyenne_deformation_contrainte_max], [moyenne_contrainte_max], xerr=xerr, yerr=yerr, capsize=3, fmt="o", ecolor = cross_palette[l], color = cross_palette[l],label = legende_moyenne)

                #prise en comptte des bare d'érreur dans le dimensionement de la fenettre
                x_max = moyenne_deformation_contrainte_max +xerr[0]
                x_min = moyenne_deformation_contrainte_max -xerr[0]
                y_max = moyenne_contrainte_max+yerr[0]
                y_min = moyenne_contrainte_max-yerr[0]

                if (window_size[0] == '') or (x_min < window_size[0]):
                    window_size[0] = x_min
                if (window_size[1] == '') or (x_max > window_size[1]):
                    window_size[1] = x_max
                if (window_size[2] == '') or (y_min < window_size[2]):
                    window_size[2] = y_min
                if (window_size[3] == '') or (y_max > window_size[3]):
                    window_size[3] = y_max

            if description_ech_pousser:
                info_sup.append(txt)
                
        # moyenne Rp0.2
        if len(l_rp02) >= 2:
            moyenne_rp02= statistics.mean(l_rp02)
            ecart_type_rp02 = statistics.stdev(l_rp02)

            moyenne_deformation_rp02= statistics.mean(ep_rp02)
            ecart_type_deformation_rp02 = statistics.stdev(ep_rp02)

            moyenne_E = statistics.mean(list_E)
            ecart_type_E = statistics.stdev(list_E)

            moyenne_K = statistics.mean(list_K)
            ecart_type_K = statistics.stdev(list_K)
        else:
            moyenne_rp02= l_rp02[0]
            ecart_type_rp02 = 0

            moyenne_deformation_rp02= ep_rp02[0]
            ecart_type_deformation_rp02 = 0

            moyenne_E = list_E[0]
            ecart_type_E = 0

            moyenne_K = list_K[0]
            ecart_type_K = 0

        nb_ar=2

        print("\n            ===\n")
        print("médiane des Rp0.2:",statistics.median(l_rp02))
        print("")
        print("moyenne Rp0.2:",moyenne_rp02)
        print("ecart_type_rp02:",ecart_type_rp02)
        print("")
        print("Dans   68% des cas Rp0.2 ∈",[round(moyenne_rp02-ecart_type_rp02, nb_ar) , round(moyenne_rp02+ecart_type_rp02, nb_ar)],"MPa")
        print("Dans   95% des cas Rp0.2 ∈",[round(moyenne_rp02-2*ecart_type_rp02, nb_ar) , round(moyenne_rp02+2*ecart_type_rp02, nb_ar)],"MPa")
        print("Dans 99.7% des cas Rp0.2 ∈",[round(moyenne_rp02-3*ecart_type_rp02, nb_ar) , round(moyenne_rp02+3*ecart_type_rp02, nb_ar)],"MPa")

        print("\n            ===\n")
        print("médiane des E:",statistics.median(list_E))
        print("")
        print("moyenne E:",moyenne_E)
        print("ecart_type_E:",ecart_type_E)
        print("")
        print("Dans   68% des cas E ∈",[round(moyenne_E-ecart_type_E, nb_ar) , round(moyenne_E+ecart_type_E, nb_ar)],"MPa")
        print("Dans   95% des cas E ∈",[round(moyenne_E-2*ecart_type_E, nb_ar) , round(moyenne_E+2*ecart_type_E, nb_ar)],"MPa")
        print("Dans 99.7% des cas E ∈",[round(moyenne_E-3*ecart_type_E, nb_ar) , round(moyenne_E+3*ecart_type_E, nb_ar)],"MPa")

        print("\n            ===\n")
        print("médiane des K:",statistics.median(list_K))
        print("")
        print("moyenne K:",moyenne_K)
        print("ecart_type_K:",ecart_type_K)
        print("")
        print("Dans   68% des cas K ∈",[round(moyenne_K-ecart_type_K, nb_ar) , round(moyenne_K+ecart_type_K, nb_ar)],"kN/mm")
        print("Dans   95% des cas K ∈",[round(moyenne_K-2*ecart_type_K, nb_ar) , round(moyenne_K+2*ecart_type_K, nb_ar)],"kN/mm")
        print("Dans 99.7% des cas K ∈",[round(moyenne_K-3*ecart_type_K, nb_ar) , round(moyenne_K+3*ecart_type_K, nb_ar)],"kN/mm")

        if description_ech_pousser and force_depla == False:
            info_sup.append( ''.join(["E ∈ [",str(round(moyenne_E-2*ecart_type_E, nb_ar))," ; ",str(round(moyenne_E+2*ecart_type_E, nb_ar)),'] à 95%']) )
        if description_ech_pousser and force_depla == True:
            info_sup.append( ''.join(["K ∈ [",str(round(moyenne_K-2*ecart_type_K, nb_ar))," ; ",str(round(moyenne_K+2*ecart_type_K, nb_ar)),'] à 95%']) )
            

        export_data.append([''])
        export_data.append(["Moyenne des Rp0.2",moyenne_rp02])
        export_data.append(["Ecart type des Rp0.2",ecart_type_rp02])
        export_data.append(['Dans   68% des cas Rp0.2 €',round(moyenne_rp02-1*ecart_type_rp02, nb_ar) , round(moyenne_rp02+1*ecart_type_rp02, nb_ar) ,''.join(['+-',str(round(1*ecart_type_contrainte_max,nb_ar))]) ])
        export_data.append(['Dans   95% des cas Rp0.2 €',round(moyenne_rp02-2*ecart_type_rp02, nb_ar) , round(moyenne_rp02+2*ecart_type_rp02, nb_ar) ,''.join(['+-',str(round(2*ecart_type_contrainte_max,nb_ar))]) ])
        export_data.append(['Dans 99.7% des cas Rp0.2 €',round(moyenne_rp02-3*ecart_type_rp02, nb_ar) , round(moyenne_rp02+3*ecart_type_rp02, nb_ar) ,''.join(['+-',str(round(3*ecart_type_contrainte_max,nb_ar))]) ])
        export_data.append([''])
        export_data.append(['Mediane du Rp0.2',statistics.median(l_rp02)])

        if force_depla == False: # Contrainte/Déformation
            txt = ''.join([" \n$Rp_{{{}}}$ ∈ [".format(Rp),str(round(moyenne_rp02-2*ecart_type_rp02, nb_ar))," ; ",str(round(moyenne_rp02+2*ecart_type_rp02, nb_ar)),'] à 95%'])#\n(E ∈ [',str(round(moyenne_E-2*ecart_type_E, nb_ar))," ; ",str(round(moyenne_E+2*ecart_type_E, nb_ar)),'] à 95%)'])
            txt2 = ''.join(["$Rp_{{{}}}$ ∈ [".format(Rp),str(round(moyenne_rp02-2*ecart_type_rp02, nb_ar))," ; ",str(round(moyenne_rp02+2*ecart_type_rp02, nb_ar)),'] à 95%'])#\n(E ∈ [',str(round(moyenne_E-2*ecart_type_E, nb_ar))," ; ",str(round(moyenne_E+2*ecart_type_E, nb_ar)),'] à 95%)'])
            if affich_rp02 == True:
                #affichage moyenne Rp0.2
                if mettre_fleche== True:
                    x_txt = moyenne_deformation_rp02*1 + 7*ecart_type_deformation_rp02
                    y_txt = moyenne_rp02*1 - 3*ecart_type_rp02
                    b=window_size[1]-window_size[0]
                    h=window_size[3]-window_size[2]
                    x_txt = moyenne_deformation_rp02 + b*X_taille_fleche/100
                    y_txt = moyenne_rp02 + h*Y_taille_fleche/100
                    
                    
                                   
                    plt.text(x_txt, y_txt, txt)
                    p1 = patches.FancyArrowPatch( (moyenne_deformation_rp02, moyenne_rp02), (x_txt, y_txt),arrowstyle='<-', mutation_scale=20)
                    plt.gca().add_patch(p1)

                n=2
                xerr = [ecart_type_deformation_rp02*n]# , ecart_type_deformation_contrainte_max*n]
                yerr = [ecart_type_rp02*n]#, ecart_type_contrainte_max*n]
                plt.errorbar([moyenne_deformation_rp02], [moyenne_rp02], xerr=xerr, yerr=yerr, capsize=3, fmt="o", ecolor = cross_palette[l], color = cross_palette[l],label = "moyenne des Rp0.2 avec\nun intervale de confiance à 95%")

            if description_ech_pousser:
                info_sup.append(txt2)
        

        
        
        # moyenne Fmax
        if len(list_for_max) >= 2:
            moyenne_fmax= statistics.mean(list_for_max)
            ecart_type_fmax = statistics.stdev(list_for_max)

            moyenne_deplacement_fmax = statistics.mean(list_dep_for_max)
            ecart_type_deplacement_fmax = statistics.stdev(list_dep_for_max)

        else:
            moyenne_fmax= list_for_max[0]
            ecart_type_fmax = 0

            moyenne_deplacement_fmax= list_dep_for_max[0]
            ecart_type_deplacement_fmax = 0


        nb_ar=2

        print("\n            ===\n")
        print("médiane des Fmax:",statistics.median(list_for_max))
        print("")
        print("moyenne Fmax:", moyenne_fmax)
        print(" ecart_type_fmax:", ecart_type_fmax)
        print("")
        print("Dans   68% des cas Fmax ∈",[round(moyenne_fmax-ecart_type_fmax, nb_ar) , round(moyenne_fmax+ecart_type_fmax, nb_ar)],"kN")
        print("Dans   95% des cas Fmax ∈",[round(moyenne_fmax-2*ecart_type_fmax, nb_ar) , round(moyenne_fmax+2*ecart_type_fmax, nb_ar)],"kN")
        print("Dans 99.7% des cas Fmax ∈",[round(moyenne_fmax-3*ecart_type_fmax, nb_ar) , round(moyenne_fmax+3*ecart_type_fmax, nb_ar)],"kN")


        export_data.append([''])
        export_data.append(["Moyenne des Fmax",moyenne_fmax])
        export_data.append(["Ecart type des Fmax",ecart_type_fmax])
        export_data.append(['Dans   68% des cas Fmax €',round(moyenne_fmax-1*ecart_type_fmax, nb_ar) , round(moyenne_fmax+1*ecart_type_fmax, nb_ar) ,''.join(['+-',str(round(1*ecart_type_contrainte_max,nb_ar))]) ])
        export_data.append(['Dans   95% des cas Fmax €',round(moyenne_fmax-2*ecart_type_fmax, nb_ar) , round(moyenne_fmax+2*ecart_type_fmax, nb_ar) ,''.join(['+-',str(round(2*ecart_type_contrainte_max,nb_ar))]) ])
        export_data.append(['Dans 99.7% des cas Fmax €',round(moyenne_fmax-3*ecart_type_fmax, nb_ar) , round(moyenne_fmax+3*ecart_type_fmax, nb_ar) ,''.join(['+-',str(round(3*ecart_type_contrainte_max,nb_ar))]) ])
        export_data.append([''])
        export_data.append(['Mediane du Fmax',statistics.median(list_for_max)])

        if description_ech_pousser and force_depla == True:
            info_sup.append( ''.join(["Fmax ∈ [",str(round(moyenne_fmax-2*ecart_type_fmax, nb_ar))," ; ",str(round(moyenne_fmax+2*ecart_type_fmax, nb_ar)),'] à 95%']) )


        if force_depla == True: # Contrainte/Déformation
            #affichage moyenne Rp0.2
            txt = ''.join([" \n$Rp_{{{}}}$ ∈ [".format(Rp),str(round(moyenne_rp02-2*ecart_type_rp02, nb_ar))," ; ",str(round(moyenne_rp02+2*ecart_type_rp02, nb_ar)),'] à 95%'])#\n(E ∈ [',str(round(moyenne_E-2*ecart_type_E, nb_ar))," ; ",str(round(moyenne_E+2*ecart_type_E, nb_ar)),'] à 95%)'])
            if affich_rp02 == True:
                if mettre_fleche== True:
                    x_txt = moyenne_deformation_rp02*1 + 7*ecart_type_deformation_rp02
                    y_txt = moyenne_rp02*1 - 3*ecart_type_rp02
                    
                                   
                    plt.text(x_txt, y_txt, txt)
                    p1 = patches.FancyArrowPatch( (moyenne_deformation_rp02, moyenne_rp02), (x_txt, y_txt),arrowstyle='<-', mutation_scale=20)
                    plt.gca().add_patch(p1)

                n=2
                xerr = [ecart_type_deformation_rp02*n]# , ecart_type_deformation_contrainte_max*n]
                yerr = [ecart_type_rp02*n]#, ecart_type_contrainte_max*n]
                plt.errorbar([moyenne_deformation_rp02], [moyenne_rp02], xerr=xerr, yerr=yerr, capsize=3, fmt="o", ecolor = cross_palette[l], color = cross_palette[l],label = "moyenne des Rp0.2 avec\nun intervale de confiance à 95%")


        if montrer_description_échantillon:
            if description_ech_pousser:
                description_echantillon = [description_echantillon] + info_sup
                description_echantillon = '\n'.join(description_echantillon)
            plt.plot(moy_X[0], moy_Y[0], color=couleur_moy, linestyle='-', linewidth=largeur_moy, label=description_echantillon)
                    
        export_data.append([''])
        export_data += list_export_courbe_moyenne #rajout de la courbe moyenne
        ecrire_liste_dans_csv(export_data,"test.csv")


        
            

    #rajout d'une marge sur la fenetre d'affichage
    marge = 5 #%
    dx = window_size[1] - window_size[0]
    window_size[0] = window_size[0]- dx * (marge/100)
    window_size[1] = window_size[1]+ dx * (marge/100)
    dy = window_size[3] - window_size[2]
    window_size[2] = window_size[2]- dy * (marge/100)
    window_size[3] = window_size[3]+ dy * (marge/100)
    #plt.axis([-2, 20, -3, 65])
    plt.axis(window_size)

    #cadrillage
    delta_X = abs(window_size[0]-window_size[1])
    delta_Y = abs(window_size[3]-window_size[2])
    nb_ligne = 10
    nb_ligne_mineur = 5

    x_espacement = round_to_base(delta_X/nb_ligne , base=bas_arondi)
    x_espacement_mineur = round_to_base(x_espacement/nb_ligne_mineur , base=bas_arondi)
    y_espacement = round_to_base(delta_Y/nb_ligne , base=bas_arondi)
    y_espacement_mineur = round_to_base(y_espacement/nb_ligne_mineur , base=bas_arondi)
    
    
    #https://stackoverflow.com/questions/24943991/change-grid-interval-and-specify-tick-labels
    maj_posy = ticker.MultipleLocator(y_espacement)   # major ticks for every 5 units
    min_posy = ticker.MultipleLocator(y_espacement_mineur)    # minor ticks for every 1 units
    maj_posx = ticker.MultipleLocator(x_espacement)   
    min_posx = ticker.MultipleLocator(x_espacement_mineur)    

    ax.xaxis.set(major_locator=maj_posx, minor_locator=min_posx)
    ax.yaxis.set(major_locator=maj_posy, minor_locator=min_posy)

    ax.tick_params(axis='both', which='minor', length=0)   # remove minor tick lines

    # different settings for major & minor gridlines
    ax.grid(which='major', alpha=0.5)
    ax.grid(which='minor', alpha=0.2, linestyle='--')

    
    if force_depla == False: # Contrainte/Déformation
        plt.ylabel('Contrainte (MPa)')
        plt.xlabel('Déformation de traction (%)')
    elif force_depla == True: # Force/Déplacement
        plt.ylabel('Force (kN)')
        plt.xlabel('Déplacement (mm)')

    plt.title(titre)#\n10/06/2024') 
    plt.legend()
    #plt.grid()
    
    #plt.show()
    
    
    if montrer_deriver:
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


        print("fig.show()")
        #fig.show()
        print("plt.show()")
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





def parametres_identiques(para,list_nom):
    if type(list_nom) is list:
        num = len(list_nom)
    else:
        return para
    Parametres=[]
    for i in range(num):
        Parametres.append(para)
    return Parametres


def b(R,A=100,a=10):
    return (A+math.pi*R**2)/a

a=[1,2,3,4,5,6,7,8,9]
b=[11,22,33,44,55,66,77,88,99]
c=np.array([np.array(a),np.array(b)])
c.T


nom_csv =["C:/Users/ecreach/Documents/PFE Caratérisation impression 3D/PETG-1-050724.csv",
          "C:/Users/ecreach/Documents/PFE Caratérisation impression 3D/PETG-2-050724.csv",
          "C:/Users/ecreach/Documents/PFE Caratérisation impression 3D/PETG-3-050724.csv"]#,
          #"C:/Users/ecreach/Documents/PFE Caratérisation impression 3D/PETG-5-080724.csv",
          #"C:/Users/ecreach/Documents/PFE Caratérisation impression 3D/PETG-6-080724.csv"]


nom_csv =["C:/Users/ecreach/Documents/PFE Caratérisation impression 3D/PETG-2-050724.csv",
          "C:/Users/ecreach/Documents/PFE Caratérisation impression 3D/PETG-5-080724.csv",
          "C:/Users/ecreach/Documents/PFE Caratérisation impression 3D/PETG-6-080724.csv",
          "C:/Users/ecreach/Documents/PFE Caratérisation impression 3D/PETG-7-080724.csv"]

nom_csv =["C:/Users/ecreach/Documents/PFE Caratérisation impression 3D/PETG-7-080724.csv",
          "C:/Users/ecreach/Documents/PFE Caratérisation impression 3D/PETG-8-090724.csv",
          "C:/Users/ecreach/Documents/PFE Caratérisation impression 3D/PETG-9-090724.csv"]

nom_csv=["C:/Users/ecreach/Documents/PFE Caratérisation impression 3D/essais 10-07-2024 pour laurent/EP1-100724.csv",
         "C:/Users/ecreach/Documents/PFE Caratérisation impression 3D/essais 10-07-2024 pour laurent/EP2-100724.csv",
         "C:/Users/ecreach/Documents/PFE Caratérisation impression 3D/essais 10-07-2024 pour laurent/EP3-100724.csv",
         "C:/Users/ecreach/Documents/PFE Caratérisation impression 3D/essais 10-07-2024 pour laurent/EP4-100724.csv",
         "C:/Users/ecreach/Documents/PFE Caratérisation impression 3D/essais 10-07-2024 pour laurent/EP5-100724.csv",
         "C:/Users/ecreach/Documents/PFE Caratérisation impression 3D/essais 10-07-2024 pour laurent/EP6-100724.csv"]

nom_csv = "C:/Users/ecreach/Documents/PFE Caratérisation impression 3D/essais 10-07-2024 pour laurent/EP5-100724.csv"
         
nom_csv=["C:/Users/ecreach/Documents/PFE Caratérisation impression 3D/PETG-11-220724.csv",
         "C:/Users/ecreach/Documents/PFE Caratérisation impression 3D/PETG-12-220724.csv"]

nom_csv=["C:/Users/ecreach/Documents/PFE Caratérisation impression 3D/PETG-13-230724.csv",
         "C:/Users/ecreach/Documents/PFE Caratérisation impression 3D/PETG-3-050724.csv",
         "C:/Users/ecreach/Documents/PFE Caratérisation impression 3D/PETG-4-050724.csv"]

nom_csv = ["C:/Users/ecreach/Documents/PFE Caratérisation impression 3D/essais 10-07-2024 pour laurent/EP5-100724.csv"]




parametres=['montrer n° eprouvette', 0,
            'Deplacer les deformation en 0', True,
            'Rp', 0.2,
            'montrer_que_certaines_eprouvettes', [[1],[],[],[],[],[],[]],
            'Longueur_éprouvette (obsolete)', 115,
            'debug',True,
            'montrer dériver',False,
            'Force/déplacement',True,
            'afficher contrainte max',True,
            'afficher Rp02',False,
            'mode affichage courbe [1,2,3,4]',2,
            'mettre flèches',False,
            'nom graphique','Essai de traction sur Filaments PETG',
            'base arondi',5,
            'afficher moyenne contrainte max',False,
            'montrer n° échantillon', False,
            'montrer description échantillon',True,
            'Description échantillon',['Eprouvette 60° imprimer à 260°¤',
                                       'Eprouvette 60° imprimer à 230°¤'],
            'afficher courbe moyenne',False,]

nom_para = "C:/Users/ecreach/Documents/PFE Caratérisation impression 3D/Essai de traction sur Eprouvette courbe PETG Vierge francofil (11,12,14).txt"
nom_para = "C:/Users/ecreach/Documents/PFE Caratérisation impression 3D/Essai de traction sur Filaments PETG.txt"
#nom_para = "C:/Users/ecreach/Documents/PFE Caratérisation impression 3D/Eprouvette ISO527 PETG Bleu.txt"
nom_para = "C:/Users/ecreach/Documents/PFE Caratérisation impression 3D/temp.txt"

nom_csv, parametres=lire_parametres(nom_para)
#print("nom_csv",nom_csv)
para = parametres_identiques(parametres,nom_csv)
depouiller_essais_traction_simple(nom_csv,para)

#combiner_plusieurs_échantillons(nom_csv,"C:/Users/ecreach/Desktop/test.csv")

























"""

parametres=['montrer n° eprouvette', 0,
            'Deplacer les deformation en 0', True,
            'Rp', 0.2,
            'montrer_que_certaines_eprouvettes', [[],[],[],[],[],[],[]],
            'Longueur_éprouvette (obsolete)', 115,
            'debug',False,
            'montrer dériver',False,
            'Force/déplacement',False,
            'afficher contrainte max',True,
            'afficher Rp02',False,
            'mode affichage courbe [1,2,3,4]',3,
            'mettre flèches',False,
            'nom graphique','Essai de traction sur Filaments PETG',
            'base arondi',5,
            'afficher moyenne contrainte max',True,
            'montrer n° échantillon', False,
            'montrer description échantillon',True,
            'Description échantillon',['PETG Vierge Froncofil',
                                       'PETG Bleu    Francofil',
                                       'PETG Vierge Daily sun'],
            'afficher courbe moyenne',True,]

"""

