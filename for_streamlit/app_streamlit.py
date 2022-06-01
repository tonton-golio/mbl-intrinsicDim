import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
from plotting import *


import os
import sys

pwd = os.getcwd() 
a = pwd.split('/')
path_add = '/'.join(a[:-1])

sys.path.insert(0, path_add)
from utils import *


st.set_page_config(layout='wide')

text_dict = {}
with open('for_streamlit/app_text.txt') as f:
    text = f.read().split('#')[1:]
for sec in text:
    lines = sec.split('\n\n')
    text_dict[lines[0]] = dict(zip(range(len(lines)-2),lines[1:-1]))

# Sidebar
st.sidebar.title('MBL intrinsic dimension')
st.sidebar.write("Main website [anton.gg](https://anton.gg)")


#_ = st.sidebar.header(text_dict['SIDEBAR'][0])
mode = 'Real time'  # st.radio('Pick one', ['Real time', 'Pre-loaded'])

if mode == 'Real time':


    col1s1, col2s1 = st.sidebar.columns(2)
    L = col1s1.slider('System size, L',2,14,10,2)
    seed = col2s1.slider('Seed',0,100,42,1)

    col1s0, col2s0 = st.sidebar.columns(2)
    W = col1s0.slider('Disorder strength, W',0.1,10.,3.65,.1)
    disorder_distribution = col2s0.selectbox('disorder_distribution', ['uniform','normal','bimodal','trimodal'])

    fig_sidebar, ax_sidebar = plt.subplots(1,1,figsize=(3,1))
    ax_sidebar.patch.set_facecolor('orange')
    ax_sidebar.patch.set_alpha(0.19)

    fig_sidebar.patch.set_facecolor('blue')
    fig_sidebar.patch.set_alpha(0.0)

    V = construct_potential(L = L, W = W, seed=seed, disorder_distribution =disorder_distribution)
    sns.kdeplot(V, ax=ax_sidebar)
    st.sidebar.pyplot(fig_sidebar)

    periodic_boundary_condition = st.sidebar.checkbox('periodic boundaries')

    col1s2, col2s2 = st.sidebar.columns(2)
    U = col1s2.slider('Interaction, U',0.,2.,1.,.1)
    t = col2s2.slider('Hopping, t',0.,2.,1.,.1)


    # Add rows to a dataframe after
    # showing it.
    df1 = pd.DataFrame(data = np.array([[8,4.7, 'trimodal', True,1.0, 1.0, 42, 17.374, 0.6]]),
                columns='L,W,disorder_distribution,periodic_boundary_condition,U,t,seed,id,chi2'.split(','))
    element = st.sidebar.dataframe(df1)


    # Add rows to a chart after
    # showing it.


    #st.header('Functions')
    col0, col1, col2, col3 = st.columns(4)
    make_potential = col1.button('Potential')
    make_hamiltonian = col2.button('Leading eigenstates')
    make_2nn = col3.button('2 nearest neighbours')
    background = col0.button('Background', True)

    buttons = [make_potential, make_hamiltonian, make_2nn]

    if sum(buttons) == 0:
        background = True

    make_r1 = False# col4.button('Nearest neighbour distribution')

    fig = plt.figure(figsize=(12,3))
    sns.set_style("whitegrid")
    if background == True:
        st.title('Many-body Localization via Intrinsic Dimension')

        intro_text = [st.write(text_dict['INTRO'][i]) for i in range(len(text_dict['INTRO'])-1)]
        st.latex(r'''{}'''.format(text_dict['INTRO'][2]))


    if make_potential == True:
        V = construct_potential(L = L, W = W, seed=seed, disorder_distribution =disorder_distribution)
        plot_potential_st(V, W, periodic_boundary_condition=periodic_boundary_condition)
        st.pyplot(fig)
        st.write(text_dict['POTENTIAL'][0]+disorder_distribution+text_dict['POTENTIAL'][1])
        st.code(text_dict['CODE_'+disorder_distribution][0])


    if make_hamiltonian == True:
        H = constructHamiltonian(L = L, W = W, U = U, t = t, seed=seed, disorder_distribution=disorder_distribution, periodic_boundary_condition=periodic_boundary_condition)
        _, vecs = np.linalg.eigh(H)
        plt.plot(np.arange(len(vecs)), vecs[:,:1]**2, c='purple', lw=2)
        plt.xlabel('Configuration index', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        st.pyplot(fig)

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
                    plt.scatter(index, V[index], c='purple', s=142, label=state, alpha=.57)
        #plt.legend()
        st.pyplot(fig2)

        col1_inner, col2_inner = st.columns(2)
        _ = col1_inner.write(text_dict['EIGENSTATES'][0])
        data_index_illu = ['000111', '001011', '001101', '001110']
        df_index_illu = pd.DataFrame(data_index_illu, columns=['Configuration'])
        _ = col2_inner.table(df_index_illu)


    if make_2nn == True:
        H = constructHamiltonian(L = L, W = W, U = U, t = t, seed=seed, disorder_distribution=disorder_distribution, periodic_boundary_condition=periodic_boundary_condition)
        _, vecs = np.linalg.eigh(H)
        d, chi2, x,y = nn2(vecs, return_xy=True)
        df2 =  pd.DataFrame(data=np.array([[L,W,disorder_distribution,periodic_boundary_condition,U,t,seed,round(d,3),round(chi2,3)]]),
                    columns='L,W,disorder_distribution,periodic_boundary_condition,U,t,seed,id,chi2'.split(','))
        #st.dataframe(df2)
        element.add_rows(df2)

        plot_2nn(x,y,d)
        st.pyplot(fig)
        st.write(text_dict['2NN'][0])

    if make_r1 == True:
        H = constructHamiltonian(L = L, W = W, U = U, t = t, seed=seed, disorder_distribution=disorder_distribution, periodic_boundary_condition=periodic_boundary_condition)
        _, vecs = np.linalg.eigh(H)
        d, chi2, R1 = nn2(vecs, return_R1=True)
        sns.distplot(R1)
        st.pyplot(fig)

if mode == 'Pre-loaded':
    seed in st.sidebar.selectbox('Pick one', [7, 42, 69])
    W in st.sidebar.selectbox('disorder strength', [1, 3.65, 6.3])
    st.write('L=14; Boundaries are periodic, U and t are fixed at 1.')
    L = 14
    disorder_dist = st.sidebar.selectbox('disorder_dist', 
                        'uniform, normal, bimodal')

    
    