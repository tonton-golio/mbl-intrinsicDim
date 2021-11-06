import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils import*
import pandas as pd
import seaborn as sns

with open('data/app_text.txt') as f:
    text = f.read().split('\n\n')

st.title('MBL intrinsic dimension')
intro_text = [st.write(text[i]) for i in range(3)]

# Sidebar
_ = st.sidebar.header(text[3])
L = st.sidebar.slider('System size, L',2,14,10,2)
W = st.sidebar.slider('Disorder strength, W',0.1,10.,3.65,.1)
U = st.sidebar.slider('Interaction strength, U',0.,2.,1.,.1)
t = st.sidebar.slider('Hopping strenght, t',0.,2.,1.,.1)
seed = st.sidebar.slider('seed',0,100,42,1)

st.header('Functions')
col1, col2, col3, col4 = st.columns(4)
make_potential = col1.button('Potential')
make_hamiltonian = col2.button('Leading eigenstates')
make_2nn = col3.button('2 nearest neighbours')
make_r1 = col4.button('Nearest neighbour distribution')

fig = plt.figure(figsize=(12,3))

if make_potential == True:
    np.random.seed(seed)
    V = np.random.uniform(-1,1,size=L) * W
    plt.title('Potential')
    plt.plot(np.arange(len(V)), V)
    plt.scatter(np.arange(len(V)), V, c='r')
    plt.ylim(-W,W)
    st.pyplot(fig)
    st.write(text[4])
    

if make_hamiltonian == True:
    H = constructHamiltonian(L = L, W = W, U = U, t = t, seed=seed)
    _, vecs = np.linalg.eigh(H)
    plt.plot(np.arange(len(vecs)), vecs[:,:3])
    st.pyplot(fig)

    col1_inner, col2_inner = st.columns(2)
    _ = col1_inner.write(text[5])
    data_index_illu = ['000111', '001011', '001101', '001110']
    df_index_illu = pd.DataFrame(data_index_illu, columns=['Configuration'])

    _ = col2_inner.table(df_index_illu)

if make_2nn == True:
    H = constructHamiltonian(L = L, W = W, U = U, t = t, seed=seed)
    _, vecs = np.linalg.eigh(H)
    d, chi2, R1 = nn2(vecs, plot=True, eps=0, return_R1=True)
    st.pyplot(fig)
    st.write(text[6])

if make_r1 == True:
    H = constructHamiltonian(L = L, W = W, U = U, t = t, seed=seed)
    _, vecs = np.linalg.eigh(H)
    d, chi2, R1 = nn2(vecs, plot=False, eps=0, return_R1=True)
    sns.distplot(R1)
    st.pyplot(fig)



