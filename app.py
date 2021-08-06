import streamlit as st
from bmosantos import lib
from bmosantos import bmorawlib
from bmosantos import wavelib

from dotenv import load_dotenv
import os

load_dotenv()

df = lib.get_data(os.getenv('REMOBS_TOKEN'))

raw_df = bmorawlib.get_data(os.getenv('REMOBS_TOKEN'))


# wave_df = wavelib.get_data(os.getenv('REMOBS_TOKEN'))

df = lib.calculate_distance(df)

st.write("# DADOS BMO-BR BACIA DE SANTOS")
st.write(f"### {(df['date_time'].min())} até {(df['date_time'].max())}")
st.write(f"### Última posição: LAT {(df['lat'].iloc[-1])}, LON {(df['lon'].iloc[-1])}")

st.write("## MAPA")

lib.plot_map(df)

st.write("## GRÁFICOS")

st.write("## DADOS BRUTOS DO VENTO")
st.write("### Velocidade do vento")
bmorawlib.plot_graphs(raw_df, ['wspd1', 'wspd2'])
st.write("### Direção do vento")
bmorawlib.plot_graphs(raw_df, ['wdir1', 'wdir2'])

st.write("## MELHOR VENTO")
st.write("### Velocidade do vento")
lib.plot_graphs(df, ['wspd'])
st.write("### Direção do vento")
lib.plot_graphs(df, ['wdir'])


st.write("### Altura de ondas")
lib.plot_graphs(df, ['swvht1', 'swvht2'])
st.write("### Direção de ondas")
lib.plot_graphs(df, ['wvdir1', 'wvdir2'])
st.write("### Período das ondas")
lib.plot_graphs(df, ['tp1', 'tp2'])

st.write("### Intensidade das Correntes")
lib.plot_graphs(df, ['cspd1', 'cspd2', 'cspd3'])
st.write("### Direção das correntes")
lib.plot_graphs(df, ['cdir1', 'cdir2', 'cdir3'])


# st.write("### Espectro Direcional de ondas")
# wavelib.plot_pleds(wave_df)
