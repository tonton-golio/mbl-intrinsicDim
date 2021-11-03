import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt
from utils import*
import pandas as pd
import seaborn as sns

# st.background('white')
# Intro text
st.title('MBL intrinsic dimension')
st.write('Here is an interactive experience with perform dimension analysis of Hubbard Hamiltonian eigenstates.')

st.write('We wanna contribute to answering the question originally posed by P. W. Anderson in 1958: How disordered does a potential have be for the system to retain local memory.')


# Lets keep a table of results here at the top
'''WHY DOES IT RESET WHEN I CHANGE THE SLIDERS IN THE SIDEBARR!!'''
data = []
df = pd.DataFrame(data, columns='L, W, ID'.split(', '))
table = st.table(df)


st.write('Yo yo, the hamiltonian looks like this:')
'''$$\mathcal{H} = t\sum_i^{L-1}(\hat{c}_{i+1}^\dagger \hat{c}_i +h.c.) 
            + U\sum_{i}^{L-1} \hat{n}_i\hat{n}_{i+1}
            + \sum_{i}^L h_i\hat{n}_i.$$'''
st.write('Btw, this is the Hubbard Hamiltonian in the second quantization formalism. The parameters can be ajusted via the sliders in the sidebar!')

# Sidebar
L = st.sidebar.slider('System size, L',2,10,6,2)
W = st.sidebar.slider('Disorder strength, W',0.1,10.,3.65,.1)
U = st.sidebar.slider('Interaction strength, U',0.,2.,1.,.1)
t = st.sidebar.slider('Hopping strenght, t',0.,2.,1.,.1)

import_data = st.button('Import data (lots of data)!')


bot1 = st.button('run')
    #label='Make hamiltonian and show potential (also diag)') # starts False, toggle for True
if bot1 == True:
    H, V = constructHamiltonian(L = L, W = W, U = U, t = t, seed=42, return_potential=True)
    fig = plt.figure(figsize=(10,3))
    _, vecs = np.linalg.eigh(H)
    plt.title('Potential')
    plt.plot(np.arange(1,len(V)+1), V)
    plt.ylim(-W,W)
    st.pyplot(fig)
    st.write('\nNow that we have the potential, lets have a look at the intrinsic dimension')
    fig2 = plt.figure(figsize=(10,3))
    d, chi2, R1 = nn2(vecs, plot=True, eps=0, return_R1=True)
    st.pyplot(fig2)
    st.write('The intrinsic dimension of this sample is ',round(d,2))
    data.append([L,W,d])
    df = pd.DataFrame(data, columns='L, W, ID'.split(', '))
    table.table(df)


    st.write('\nNext up; lets have a look at the nearest neighbour distance distribution!')
    fig3 = plt.figure(figsize=(10,4))
    sns.distplot(R1)
    st.pyplot(fig3)

    '''Dunno what this should actually look like?'''


    st.write('\n\n\n\nWould there be any value in importing data (perhaps place a button for this at the top. This could be used ot populate the table :D) If yes: plot the many ID_imshow and perhaps even perform scaling collapse')

    st.write('\n\nEigencomponent dominance')
    st.image('figures/EigCDom-L14-seeds20-ws11.png')
    '''Well, this needs a background...'''

    st.write('\n\nMany ID')
    st.image('figures/2nn_many_L14.png')
    '''Well, this needs a background...'''

    st.write('\n\nScaling Collapse')
    '''$$L^{-\zeta/\nu}a_{ij}=\tilde{f}\left(L^{1/\nu}(\varrho_j-\varrho_c)\right)$$'''
    st.image('figures/collapsed_skip7.png')

    st.write('\nThis analysis yields a critical point, $W_c=3.65\pm0.2$')
    
    







#bot2 = st.button(label='Lets have a look at those eigencomponents')
#if bot2 == True:


    



