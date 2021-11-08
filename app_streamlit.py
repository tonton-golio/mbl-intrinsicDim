import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import pandas as pd
import seaborn as sns
from plotting import *


st.set_page_config(layout='wide')


text_dict = {}
with open('data/app_text.txt') as f:
    text = f.read().split('#')[1:]
for sec in text:
    lines = sec.split('\n\n')
    text_dict[lines[0]] = dict(zip(range(len(lines)-2),lines[1:-1]))


st.title('MBL intrinsic dimension')
intro_text = [st.sidebar.write(text_dict['INTRO'][i]) for i in range(len(text_dict['INTRO'])-1)]
latex = st.sidebar.latex(r'''{}'''.format(text_dict['INTRO'][2]))

# Sidebar
_ = st.sidebar.header(text_dict['SIDEBAR'][0])
L = st.sidebar.slider('System size, L',2,14,10,2)
W = st.sidebar.slider('Disorder strength, W',0.1,10.,3.65,.1)
disorder_distribution = st.sidebar.selectbox('disorder_distribution', ['uniform', 'bimodal', 'normal', 'sinusoidal', 'trimodal'])
periodic_boundary_condition = st.sidebar.checkbox('periodic_boundary_condition')
fig_sidebar, ax_sidebar = plt.subplots(1,1,figsize=(3,1))
V = construct_potential(L = 1000, W = W, seed=42, disorder_distribution =disorder_distribution)
sns.histplot(V, ax=ax_sidebar, bins=40)
st.sidebar.pyplot(fig_sidebar)




U = st.sidebar.slider('Interaction strength, U',0.,2.,1.,.1)
t = st.sidebar.slider('Hopping strenght, t',0.,2.,1.,.1)
seed = st.sidebar.slider('seed',0,100,42,1)

#st.header('Functions')
col1, col2, col3, col4 = st.columns(4)
make_potential = col1.button('Potential')
make_hamiltonian = col2.button('Leading eigenstates')
make_2nn = col3.button('2 nearest neighbours')
make_r1 = col4.button('Nearest neighbour distribution')

fig = plt.figure(figsize=(12,3))


if make_potential == True:
    V = construct_potential(L = L, W = W, seed=seed, disorder_distribution =disorder_distribution)
    plot_potential_st(V, W, periodic_boundary_condition=periodic_boundary_condition)
    st.pyplot(fig)
    st.write(text_dict['POTENTIAL'][0]+disorder_distribution+text_dict['POTENTIAL'][1])
    st.code(text_dict['CODE_'+disorder_distribution][0])


if make_hamiltonian == True:
    H = constructHamiltonian(L = L, W = W, U = U, t = t, seed=seed, disorder_distribution=disorder_distribution, periodic_boundary_condition=periodic_boundary_condition)
    _, vecs = np.linalg.eigh(H)
    plt.plot(np.arange(len(vecs)), vecs[:,:1])
    st.pyplot(fig)

    col1_inner, col2_inner = st.columns(2)
    _ = col1_inner.write(text_dict['EIGENSTATES'][0])
    data_index_illu = ['000111', '001011', '001101', '001110']
    df_index_illu = pd.DataFrame(data_index_illu, columns=['Configuration'])

    _ = col2_inner.table(df_index_illu)
    s2i, i2s = basisStates(L)
    most_probable_config = i2s[sorted(list(i2s.keys()))[np.argmax(abs(vecs[:,0]))]]
    st.write(text_dict['EIGENSTATES'][1]+ f' **{most_probable_config}** '+text_dict['EIGENSTATES'][2])

    fig2 = plt.figure(figsize=(12,3))
    V = construct_potential(L = L, W = W, seed=seed, disorder_distribution =disorder_distribution)
    plot_potential_st(V, W, periodic_boundary_condition=periodic_boundary_condition)

    show = 1
    for state in range(show):
        probable_config = i2s[sorted(list(i2s.keys()))[np.argmax(abs(vecs[:,state]))]]
        #st.write(probable_config)
        for index, i in enumerate(probable_config):
            if i == '1':
                plt.scatter(index, V[index], c='r', s=142, label=state, alpha=1)
    #plt.legend()
    st.pyplot(fig2)


if make_2nn == True:
    H = constructHamiltonian(L = L, W = W, U = U, t = t, seed=seed, disorder_distribution=disorder_distribution, periodic_boundary_condition=periodic_boundary_condition)
    _, vecs = np.linalg.eigh(H)
    d, chi2, x,y = nn2(vecs, return_xy=True)
    plot_2nn(x,y,d)
    st.pyplot(fig)
    st.write(text_dict['2NN'][0])

if make_r1 == True:
    H = constructHamiltonian(L = L, W = W, U = U, t = t, seed=seed, disorder_distribution=disorder_distribution, periodic_boundary_condition=periodic_boundary_condition)
    _, vecs = np.linalg.eigh(H)
    d, chi2, R1 = nn2(vecs, return_R1=True)
    sns.distplot(R1)
    st.pyplot(fig)

