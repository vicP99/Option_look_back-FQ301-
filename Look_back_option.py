import numpy as np                    # bibliotheque pour vecteurs et matrices
import matplotlib.pyplot as plt
from numpy import linalg       # pour le plot
from scipy.stats import norm          # distribution gaussienne
import scipy.stats as stats
import math as m

nom_methode="implicite" #Au choix: explicite, implicite, CN,.

test="convergence_spatiale_explicite"#Au choix: CFL_explixit, convergence_spatiale_explicite, 




eps=0.00001
# Parametres principaux
K = 1.0        # strike
T = 1.0        # echeance
r = 0.5       # taux de l'actif sans risque
sigma = 1   # volatilite du sous-jacent


# fonction du payoff
def phi_SM(S, M):
    return M - S

def phi_x(x):
    return 1-x


def N_(x):
    return stats.norm.cdf(x)

def v(t,S,M):
    d1 = (np.log((S+.000000001)/M)+(r+1/2*sigma*sigma*(T-t)))/(sigma*m.sqrt(T-t))
    sol1 = M*m.exp(-r*(T-t))*N_(-d1+sigma*m.sqrt(T-t))
    sol2 = S*N_(-d1)
    sol3 = S*m.exp(-r*(T-t))*m.pow(sigma,2)/(2*r)*(-((S+.000000001)/M)**(-2*r/sigma/sigma)*N_(d1-2*r/sigma*m.sqrt(T-t))+m.exp(r*(T-t))*N_(d1))
    return sol1-sol2+sol3

def mat_A(sigma, r, x):

    N = x.size 
    h = x[1] -x[0]

    A1 = - 1 / (2.*h*h) * np.diag(np.power(sigma*x, 2)).dot( 
        np.eye(N,k=-1) - 2. * np.eye(N) + np.eye(N,k=1) )
    A2 = - 1/2/h * np.diag(r*x).dot( -np.eye(N,k=-1) + np.eye(N, k=1) )
    A3 = r * np.eye(N)

    A4=np.zeros((N,N))

    #termes liés à la condition au bord
    A4[N-1,N-2]= x[N-1]*x[N-1]*sigma*sigma/2/h/h + r/h/2 * x[N-1]
    A4[N-1,N-1]= 2*h*x[N-1]*x[N-1]*sigma*sigma/2/h/h + 2*h*r*x[N-1]/2/h
    return (A1+A2+A3- A4)

def W_V(W,t,S,M):
    return M*W(t,S/M)
def erreur_explicite(J,N):
    x=np.linspace(0,1,N)
    h = x[1] -x[0]
    k = T / J
    W0=phi_x(x)
    Wj=np.copy(W0)
    A=mat_A(sigma,r,x)
    for j in range(J):
        Wj=(np.eye(N)-k*A).dot(Wj)
    W_exact=v(0,x,1)
    Norm=max(abs(np.array(Wj-W_exact)))
    return Norm
def erreur_implicite(J,N):
    x=np.linspace(0,1,N)
    h = x[1] -x[0]
    k = T / (1.0*J)
    W0=phi_x(x)
    Wj=np.copy(W0)
    A=mat_A(sigma,r,x)
    for j in range(J):
        Wj=np.linalg.solve(np.eye(N) +k*A,Wj)
    W_exact=v(0,x,1)
    Norm=max(abs(np.array(Wj-W_exact)))
 #   plt.plot(x,Wj-W_exact)
    return Norm
def erreur_CN(J,N):
    x=np.linspace(0,1,N)
    h = x[1] -x[0]
    k = T / (1.0*J)
    W0=phi_x(x)
    Wj=np.copy(W0)
    A=mat_A(sigma,r,x)
    for j in range(J):
        Wj=np.linalg.solve(np.eye(N)+1/2*k*A,(np.eye(N)-1/2*k*A).dot(Wj))
    W_exact=v(0,x,1)
    Norm=max(abs(np.array(Wj-W_exact)))
 #   plt.plot(x,Wj-W_exact)
    return Norm

#trouve la condition de CFL
def test_CFL(N):
    x=np.linspace(0,1,N)
    h = x[1] -x[0]
    J_cfl= T/(h*h/(sigma*sigma + r))
    print(J_cfl)
    pas=5*J_cfl/100
    J_liste=[]
    Norm_Liste=[]
    cassure=0
    for p in range (-10,11) : 
        J=int(pas*p + J_cfl)
        k = T / J
        W0=phi_x(x)
        Wj=np.copy(W0)
        A=mat_A(sigma,r,x)
        for j in range(J):
            Wj=(np.eye(N) -k*A).dot(Wj)
        W_exact=v(0,x,1)
        Norm=np.linalg.norm(Wj-W_exact)
        if(Norm<10):
            Norm_Liste.append(Norm)
            J_liste.append(J)   
        else:
            Norm_Liste.append(10)
            J_liste.append(J) 
            cassure=J
    return J_liste,Norm_Liste,cassure,J_cfl

#Affiche la condition CFL pour différente valeurs de pas
if(test=="CFL_explicite"):
    X=[]
    Y=[]
    Z=[]
    for n in range (10,80):
        J_liste,Norm_Liste,cassure,J_cfl=test_CFL(n)
        X.append(n)
        Y.append(cassure)
        Z.append(J_cfl)
    exacte,=plt.plot(X,Y,color="red")
    theorique,=plt.plot(X,Z,color="green")
    plt.legend([exacte, theorique],["Observé", "Théorique(CFL)"])
    plt.title("Comparaison entre la condition CFL et celle observée")
    plt.xlabel("Discrétisation spatiale (Nbre de points)")
    plt.ylabel("Discrétisation temporelle (Nbre de points)")
    plt.savefig("condtionCFL")
elif(test=="convergence_spatiale_explicite"):
    J=40000 #pas temporelle
    p=2
    h_cfl= np.sqrt(T/(J/(sigma*sigma + r)))
    N_cfl=int(1/h_cfl)
    print(N_cfl)
    nbre_point=15
    x=[]
    y=[]
    for i in range(nbre_point):
        x.append(np.log(1/(N_cfl-nbre_point*5+5*i)))
        y.append(np.log(erreur_explicite(J,N_cfl-nbre_point*5+5*i)))
    z=y[-1] +p*(np.array(x) -x[-1])
    exacte,=plt.plot(x,y,color="red")
    theorique,=plt.plot(x,z,color="green")
    plt.legend([exacte, theorique], ["Erreur observé", "Droite de pente "+ str(p)])
    plt.xlabel("Discrétisation spatiale: log(h)")
    plt.ylabel("Log erreur: log||u_exacte - u_théorique||")
    plt.title("Convergence spatiale explicite pour "+str(J)+" pas temporels")
    plt.savefig("convergence_spatiale_explicite")
elif(test=="convergence_temporelle_explicite"):
    N=150
    p=1
    J_cfl= int(N*N*T/(1/(sigma*sigma + r)))
    print(J_cfl)
    nbre_point=15
    x=[]
    y=[]
    for i in range(nbre_point):
        x.append(np.log(1/(J_cfl+i)))
        y.append(np.log(erreur_explicite(J_cfl+i,N)))
    z=y[-1] +p*(np.array(x) -x[-1])
    exacte,=plt.plot(x,y,color="red")
    theorique,=plt.plot(x,z,color="green")
    plt.legend([exacte, theorique], ["Erreur observé", "Droite de pente "+ str(p)])
    plt.xlabel("Discrétisation temporelle: log(k)")
    plt.ylabel("Log erreur: log||u_exacte - u_théorique||")
    plt.title("Convergence temporelle explicite pour "+str(N)+ " pas spatiaux")
    plt.savefig("convergence_temporelle_explicite")
elif(test=="convergence_spatiale_implicite"):
    J=20000
    p=2
    N_cfl=20
    nbre_point=15
    x=[]
    y=[]
    for i in range(nbre_point):
        x.append(np.log(1/(N_cfl+5*i)))
        y.append(np.log(erreur_implicite(J,(N_cfl+5*i))))
    z=y[-1]+p*(np.array(x) -x[-1])
    exacte,=plt.plot(x,y,color="red")
    theorique,=plt.plot(x,z,color="green")
    plt.legend([exacte, theorique], ["Erreur observé", "Droite de pente "+ str(p)])
    plt.xlabel("Discrétisation spatiale: log(h)")
    plt.ylabel("Log erreur: log||u_exacte - u_théorique||")
    plt.title("Convergence spatiale implicite pour "+str(J)+ " pas temporels")
    plt.savefig("convergence_spatiale_implicite")
elif(test=="convergence_temporelle_implicite"):
    p=1
    N=200
    J_cfl= 20
    nbre_point=15
    x=[]
    y=[]
    for i in range(nbre_point):
        x.append(np.log(1/(J_cfl+i*15)))
        y.append(np.log(erreur_implicite(J_cfl+i*15,N)))
    z=y[-1] +p*(np.array(x) -x[-1])
    exacte,=plt.plot(x,y,color="red")
    theorique,=plt.plot(x,z,color="green")
    plt.legend([exacte, theorique], ["Erreur observé", "Droite de pente "+ str(p)])
    plt.xlabel("Discrétisation temporelle: log(k)")
    plt.ylabel("Log erreur: log||u_exacte - u_théorique||")
    plt.title("Convergence temporelle implicite pour "+str(N)+ " pas spatiaux")
    plt.savefig("convergence_temporelle_implicite")
elif(test=="convergence_spatiale_CN"):
    p=2
    J=10000
    N_cfl=20
    nbre_point=15
    x=[]
    y=[]
    for i in range(nbre_point):
        x.append(np.log(1/(N_cfl+5*i)))
        y.append(np.log(erreur_CN(J,(N_cfl+5*i))))
    z=y[-1]+p*(np.array(x) -x[-1])
    exacte,=plt.plot(x,y,color="red")
    theorique,=plt.plot(x,z,color="green")
    plt.legend([exacte, theorique], ["Erreur observé", "Droite de pente "+ str(p)])
    plt.xlabel("Discrétisation spatiale: log(h)")
    plt.ylabel("Log erreur: log||u_exacte - u_théorique||")
    plt.title("Convergence spatiale Ckrank-Nikolson pour "+str(J)+ " pas temporels")
    plt.savefig("convergence_spatiale_CN")
elif(test=="convergence_temporelle_CN"):
    p=2
    N=300
    J_cfl= 100
    nbre_point=15
    x=[]
    y=[]
    for i in range(nbre_point):
        x.append(np.log(1/(J_cfl+i*10)))
        y.append(np.log(erreur_CN(J_cfl+i*10,N)))
    z=y[-1] +p*(np.array(x) -x[-1])
    exacte,=plt.plot(x,y,color="red")
    theorique,=plt.plot(x,z,color="green")
    plt.legend([exacte, theorique], ["Erreur observé", "Droite de pente "+ str(p)])
    plt.xlabel("Discrétisation temporelle: log(k)")
    plt.ylabel("Log erreur: log||u_exacte - u_théorique||")
    plt.title("Convergence temporelle Ckrank-Nikolson pour "+str(N)+ " pas spatiaux")
    plt.savefig("convergence_temporelle_CN")
# Parametres de discretisation
N = 500    # pas dans le maillage en space
J = 500     # pas dans le maillage en temps

x=np.linspace(eps,1,N)
# pas de discretisation en espace et en temps
h = x[1] -x[0]
k = T / J

W0=phi_x(x)
Wj=np.copy(W0)

A=mat_A(sigma,r,x)
plt.figure()
for j in range(J):
    if(nom_methode=="explicite"):
        Wj=(np.eye(N) -k*A).dot(Wj)
    elif(nom_methode=="implicite"):
        Wj=np.linalg.solve(np.eye(N) +k*A,Wj)
    elif(nom_methode=="CN"):
        Wj=np.linalg.solve(np.eye(N)+1/2*k*A,(np.eye(N)-1/2*k*A).dot(Wj))
    else:
        print("mauvais nom de méthode")
        quit()
    # visualisation de la solution approchee
    if (j + 1) % 10 == 0:
        plt.plot(x, Wj, color='blue', linestyle='dashed', linewidth=1)

W_exact=v(0,x,1)

valin,=plt.plot(x, W0,  color='orange', linestyle='solid', linewidth=2)
valexacte,  = plt.plot(x, W_exact,  color='green', linestyle='solid', linewidth=2)
valapp, = plt.plot(x, Wj,  color='red', linestyle='dashed', linewidth=1)
plt.xlabel("Actif sous-jacent")
plt.ylabel("Valeur de l'option")


plt.legend([valin, valapp, valexacte], ["Valeur à l'écheance", "Valeur (approx)", "Valeur (exacte)"])

plt.title(nom_methode)
plt.savefig("W(x,t): methode " + nom_methode)

