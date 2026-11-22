import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Dashboard Analítico ADS-B (1090 MHz)", layout="wide")

if not os.path.exists('telemetria_adsb.csv'):
    st.error("'telemetria_adsb.csv' não encontrado, execute `python3 gerar_dados.py` primeiro.")
    st.stop()

@st.cache_data
def load_data():
    return pd.read_csv('telemetria_adsb.csv')

df = load_data()

def calcular_moda(serie, is_continua=True):
    s = serie.dropna()
    if len(s) == 0:
        return np.nan, "Sem dados"
    if not is_continua or len(s) < 5:
        val_moda = s.mode().iloc[0] if not s.mode().empty else np.nan
        return val_moda, "Moda Discreta Exata"
    
    k = max(5, int(np.ceil(1 + 3.322 * np.log10(len(s)))))
    contagens, limites = np.histogram(s, bins=k)
    idx = int(np.argmax(contagens))
    l_inf = limites[idx]
    l_sup = limites[idx + 1]
    h = l_sup - l_inf
    f_m = contagens[idx]
    f_ant = contagens[idx - 1] if idx > 0 else 0
    f_pos = contagens[idx + 1] if idx < len(contagens) - 1 else 0
    d1 = f_m - f_ant
    d2 = f_m - f_pos
    moda_czuber = l_inf + (d1 / (d1 + d2)) * h if (d1 + d2) > 0 else l_inf + h / 2
    return moda_czuber, f"Czuber [{l_inf:.0f}; {l_sup:.0f})"

st.sidebar.title("Filtros de Telemetria")

cat_selecionadas = st.sidebar.multiselect(
    "Categoria da Aeronave (Nominal):",
    options=df['categoria'].unique(),
    default=df['categoria'].unique()
)

turb_selecionadas = st.sidebar.multiselect(
    "Nível de Turbulência (Ordinal):",
    options=sorted(df['nivel_turbulencia'].unique()),
    default=sorted(df['nivel_turbulencia'].unique())
)

alt_min, alt_max = int(df['altitude_pes'].min()), int(df['altitude_pes'].max())
faixa_altitude = st.sidebar.slider(
    "Faixa de Altitude Barométrica (pés):",
    min_value=alt_min, max_value=alt_max, value=(alt_min, alt_max)
)

df_filtrado = df[
    (df['categoria'].isin(cat_selecionadas)) &
    (df['nivel_turbulencia'].isin(turb_selecionadas)) &
    (df['altitude_pes'].between(faixa_altitude[0], faixa_altitude[1]))
]

st.sidebar.divider()
st.sidebar.markdown(f"**Registros Filtrados:** `{len(df_filtrado)}` / `{len(df)}`")

if os.path.exists('telemetria_adsb_analise.xlsx'):
    with open('telemetria_adsb_analise.xlsx', 'rb') as f:
        st.sidebar.download_button(
            label="Baixar Planilha Completa (.xlsx)",
            data=f,
            file_name="Entregavel_Telemetria_ADSB.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if len(df_filtrado) < 10:
    st.warning("Selecione um intervalo maior nos filtros laterais (mínimo de 10 registros).")
    st.stop()

st.title("Dashboard de Telemetria Radar ADS-B (1090 MHz)")
st.markdown(
    "**Engenharia de Confiabilidade e Tráfego Aéreo:** Aplicação interativa de Estatística Descritiva, "
    "Modelagem Probabilística e Inferência Estatística sobre telemetria aeronáutica."
)

tab_desc, tab_prob, tab_inf, tab_dados = st.tabs([
    "A. Estatística Descritiva",
    "B. Probabilidades e Distribuições",
    "C. Inferência e Testes de Hipótese",
    "Base de Dados e Excel"
])

with tab_desc:
    st.subheader("A. Estatística Descritiva e Medidas Resumo")
    
    mapa_variaveis = {
        'altitude_pes': ('Altitude Barométrica (pés) [Contínua]', True),
        'velocidade_nos': ('Velocidade de Solo (nós) [Contínua]', True),
        'rssi_dbm': ('Potência do Sinal RSSI (dBm) [Contínua]', True),
        'mensagens_seg': ('Taxa de Pacotes ADS-B (msg/s) [Discreta]', False),
        'satelites_visiveis': ('Satélites GNSS Visíveis [Discreta]', False)
    }
    
    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        var_escolhida = st.selectbox(
            "Selecione a variável quantitativa para análise:",
            options=list(mapa_variaveis.keys()),
            format_func=lambda x: mapa_variaveis[x][0]
        )
    with col_sel2:
        agrupador_box = st.selectbox(
            "Agrupar Boxplot por (Qualitativa):",
            options=['categoria', 'nivel_turbulencia', 'alerta_desvio']
        )

    serie = df_filtrado[var_escolhida].dropna()
    is_cont = mapa_variaveis[var_escolhida][1]
    
    media = serie.mean()
    mediana = serie.median()
    moda_val, detalhe_moda = calcular_moda(serie, is_continua=is_cont)
    variancia = serie.var(ddof=1)
    desvio_padrao = serie.std(ddof=1)
    cv = (desvio_padrao / abs(media)) * 100 if media != 0 else 0

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Média (x̄)", f"{media:.2f}")
    m2.metric("Mediana (Md)", f"{mediana:.2f}")
    m3.metric("Moda (Mo)", f"{moda_val:.2f}", detalhe_moda, delta_color="off")
    m4.metric("Variância (s²)", f"{variancia:.2f}")
    m5.metric("Desvio Padrão (s)", f"{desvio_padrao:.2f}")
    m6.metric("Coef. de Variação (CV)", f"{cv:.2f}%")

    if is_cont:
        st.caption(
            "*Para variáveis contínuas, a moda é estimada por interpolação na classe modal  "
        )

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=serie,
            histnorm='probability density',
            name='Densidade Empírica',
            marker_color='#3b82f6',
            opacity=0.65
        ))
        x_grid = np.linspace(serie.min(), serie.max(), 200)
        pdf_normal = stats.norm.pdf(x_grid, loc=media, scale=desvio_padrao)
        fig_hist.add_trace(go.Scatter(
            x=x_grid, y=pdf_normal,
            mode='lines',
            name='Curva de Densidade (Normal)',
            line=dict(color="#9f0ccc", width=3)
        ))
        fig_hist.update_layout(
            title=f"Histograma com Curva de Densidade: {var_escolhida}",
            xaxis_title=var_escolhida,
            yaxis_title="Densidade",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_g2:
        fig_box = px.box(
            df_filtrado,
            x=agrupador_box,
            y=var_escolhida,
            color=agrupador_box,
            points="outliers",
            title=f"Boxplot Interativo de {var_escolhida} por {agrupador_box}",
            template="plotly_white"
        )
        st.plotly_chart(fig_box, use_container_width=True)

with tab_prob:
    st.subheader("B.1 Probabilidade Simples, Condicional e Verificação de Independência")
    st.markdown(f"**Espaço Amostral ($\Omega$):** Conjunto dos `{len(df_filtrado)}` registros de telemetria ativos no filtro.")
    
    c_ev1, c_ev2 = st.columns(2)
    with c_ev1:
        limiar_alt = st.number_input("Definir Evento A: Altitude Barométrica maior que (pés):", value=30000, step=1000)
    with c_ev2:
        cat_evento_b = st.selectbox("Definir Evento B: Aeronave pertence à Categoria:", options=df['categoria'].unique())

    n_total = len(df_filtrado)
    evento_A = df_filtrado['altitude_pes'] > limiar_alt
    evento_B = df_filtrado['categoria'] == cat_evento_b
    
    p_A = evento_A.sum() / n_total
    p_B = evento_B.sum() / n_total
    p_A_inter_B = (evento_A & evento_B).sum() / n_total
    p_A_dado_B = p_A_inter_B / p_B if p_B > 0 else 0.0
    produto_pA_pB = p_A * p_B
    diferenca_indep = abs(p_A_inter_B - produto_pA_pB)

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("P(A) [Simples]", f"{p_A:.2%}")
    p2.metric("P(B) [Simples]", f"{p_B:.2%}")
    p3.metric("P(A ∩ B) [Conjunta]", f"{p_A_inter_B:.2%}")
    p4.metric("P(A | B) [Condicional]", f"{p_A_dado_B:.2%}")

    st.latex(
        rf"P(A \mid B) = \frac{{P(A \cap B)}}{{P(B)}} = \frac{{{p_A_inter_B:.4f}}}{{{p_B:.4f}}} = {p_A_dado_B:.4f}"
    )

    st.markdown("#### Verificação Empírica de Independência Estatística")
    st.markdown(
        f"Dois eventos são estatisticamente independentes se $P(A \\cap B) = P(A) \\cdot P(B)$. "
        f"Nos dados filtrados: $P(A \\cap B) = {p_A_inter_B:.4f}$ vs $P(A)\\cdot P(B) = {produto_pA_pB:.4f}$ "
        f"(Divergência $|\Delta| = {diferenca_indep:.4f}$)."
    )
    if diferenca_indep < 0.015:
        st.success("**Evidência Empírica de Independência:** A diferença entre P(A ∩ B) e P(A)·P(B) é desprezível.")
    else:
        st.warning("**Eventos Estatisticamente Dependentes:** A categoria da aeronave altera significativamente a probabilidade de operar acima dessa altitude ($P(A|B) \\neq P(A)$).")

    st.markdown("#### Tabela de Contingência (Dupla Entrada)")
    col_tab1, col_tab2 = st.columns(2)
    with col_tab1:
        st.caption("Frequências Absolutas Observadas")
        st.dataframe(pd.crosstab(df_filtrado['categoria'], df_filtrado['nivel_turbulencia'], margins=True), use_container_width=True)
    with col_tab2:
        st.caption("Probabilidades Conjuntas Relativas (%)")
        st.dataframe((pd.crosstab(df_filtrado['categoria'], df_filtrado['nivel_turbulencia'], normalize='all', margins=True) * 100).round(2), use_container_width=True)

    st.divider()
    st.subheader("B.2 Ajuste de Distribuições Teóricas (Discreta e Contínua)")
    
    tipo_dist = st.radio(
        "Escolha o Modelo Teórico para comparação com os dados empíricos:",
        options=[
            "Distribuição de Poisson (Discreta: Taxa de Mensagens ADS-B/s)",
            "Distribuição Binomial (Discreta: Satélites GNSS Visíveis em n=12 canais)",
            "Distribuição Normal / Gaussiana (Contínua: Velocidade de Cruzeiro Comercial)"
        ],
        horizontal=True
    )

    if "Poisson" in tipo_dist:
        dados_pois = df_filtrado['mensagens_seg']
        lam_hat = dados_pois.mean()
        st.latex(rf"P(X = k) = \frac{{\lambda^k e^{{-\lambda}}}}{{k!}}, \quad \hat{{\lambda}}_{{\text{{MV}}}} = {lam_hat:.3f} \text{{ pacotes/s}}")
        
        k_vals = np.arange(0, int(dados_pois.max()) + 2)
        freq_emp = [np.mean(dados_pois == k) for k in k_vals]
        prob_teo = stats.poisson.pmf(k_vals, mu=lam_hat)
        
        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(x=k_vals, y=freq_emp, name='Frequência Empírica', marker_color='#3b82f6'))
        fig_p.add_trace(go.Scatter(x=k_vals, y=prob_teo, mode='lines+markers', name='PMF Teórica de Poisson', line=dict(color='#ef4444', width=3)))
        fig_p.update_layout(title="Aderência à Distribuição de Poisson (Pacotes/s)", xaxis_title="k (mensagens/s)", yaxis_title="Probabilidade", template="plotly_white")
        st.plotly_chart(fig_p, use_container_width=True)

    elif "Binomial" in tipo_dist:
        dados_bin = df_filtrado['satelites_visiveis']
        n_canais = 12
        p_hat = dados_bin.mean() / n_canais
        st.latex(rf"P(X = k) = \binom{{n}}{{k}} p^k (1-p)^{{n-k}}, \quad n = {n_canais}, \quad \hat{{p}} = {p_hat:.4f}")
        
        k_vals = np.arange(0, n_canais + 1)
        freq_emp = [np.mean(dados_bin == k) for k in k_vals]
        prob_teo = stats.binom.pmf(k_vals, n=n_canais, p=p_hat)
        
        fig_b = go.Figure()
        fig_b.add_trace(go.Bar(x=k_vals, y=freq_emp, name='Frequência Empírica', marker_color='#8b5cf6'))
        fig_b.add_trace(go.Scatter(x=k_vals, y=prob_teo, mode='lines+markers', name='PMF Teórica Binomial', line=dict(color='#f59e0b', width=3)))
        fig_b.update_layout(title="Aderência à Distribuição Binomial (Satélites GNSS Visíveis)", xaxis_title="k (satélites)", yaxis_title="Probabilidade", template="plotly_white")
        st.plotly_chart(fig_b, use_container_width=True)

    else:
        dados_norm = df_filtrado[df_filtrado['categoria'] == 'Comercial']['velocidade_nos']
        if len(dados_norm) < 5:
            dados_norm = df_filtrado['velocidade_nos']
        mu_hat, sigma_hat = dados_norm.mean(), dados_norm.std(ddof=1)
        st.latex(rf"f(x) = \frac{{1}}{{\sigma\sqrt{{2\pi}}}} \exp\left(-\frac{{1}}{{2}}\left(\frac{{x-\mu}}{{\sigma}}\right)^2\right), \quad \hat{{\mu}} = {mu_hat:.2f}, \quad \hat{{\sigma}} = {sigma_hat:.2f}")
        
        x_vals = np.linspace(dados_norm.min(), dados_norm.max(), 200)
        fig_n = go.Figure()
        fig_n.add_trace(go.Histogram(x=dados_norm, histnorm='probability density', name='Dados Empíricos', marker_color='#10b981', opacity=0.6))
        fig_n.add_trace(go.Scatter(x=x_vals, y=stats.norm.pdf(x_vals, mu_hat, sigma_hat), mode='lines', name='PDF Normal Teórica', line=dict(color='#1d4ed8', width=3)))
        fig_n.update_layout(title="Aderência à Distribuição Normal: Velocidade de Solo (nós)", xaxis_title="Velocidade (nós)", yaxis_title="Densidade f(x)", template="plotly_white")
        st.plotly_chart(fig_n, use_container_width=True)

with tab_inf:
    st.subheader("C.1 Intervalos de Confiança (Média e Proporção)")
    
    c_ic1, c_ic2 = st.columns(2)
    with c_ic1:
        var_ic = st.selectbox(
            "Métrica Crítica para Estimativa Intervalar:",
            options=['velocidade_nos', 'altitude_pes', 'rssi_dbm', 'proporcao_alerta_tcas'],
            format_func=lambda x: "Proporção de Alertas TCAS (p̂) [IC Z de Proporção]" if x == 'proporcao_alerta_tcas' else f"Média de {x} (μ) [IC t de Student]"
        )
    with c_ic2:
        conf_level = st.slider("Nível de Confiança (1 - α):", min_value=0.90, max_value=0.99, value=0.95, step=0.01)

    if var_ic == 'proporcao_alerta_tcas':
        n_ic = len(df_filtrado)
        p_hat_ic = (df_filtrado['alerta_desvio'] == 'Sim').mean()
        z_crit_ic = stats.norm.ppf((1 + conf_level) / 2)
        ep_ic = np.sqrt((p_hat_ic * (1 - p_hat_ic)) / n_ic)
        margem_erro = z_crit_ic * ep_ic
        ic_inf, ic_sup = max(0.0, p_hat_ic - margem_erro), min(1.0, p_hat_ic + margem_erro)
        
        ic1, ic2, ic3, ic4 = st.columns(4)
        ic1.metric("Proporção Amostral (p̂)", f"{p_hat_ic:.2%}")
        ic2.metric("Margem de Erro (±E)", f"± {margem_erro:.2%}")
        ic3.metric(f"Limite Inferior ({conf_level:.0%})", f"{ic_inf:.2%}")
        ic4.metric(f"Limite Superior ({conf_level:.0%})", f"{ic_sup:.2%}")
        st.latex(rf"IC_{{{conf_level:.0%}}}(p) = \hat{{p}} \pm Z_{{\alpha/2}} \sqrt{{\frac{{\hat{{p}}(1-\hat{{p}})}}{{n}}}} = {p_hat_ic:.4f} \pm {margem_erro:.4f} = [{ic_inf:.4f}; \, {ic_sup:.4f}]")
    else:
        amostra_ic = df_filtrado[var_ic].dropna()
        n_ic = len(amostra_ic)
        media_ic = amostra_ic.mean()
        erro_padrao = stats.sem(amostra_ic)
        t_crit_ic = stats.t.ppf((1 + conf_level) / 2, df=n_ic - 1)
        margem_erro = t_crit_ic * erro_padrao
        ic_inf, ic_sup = media_ic - margem_erro, media_ic + margem_erro

        ic1, ic2, ic3, ic4 = st.columns(4)
        ic1.metric("Média Amostral (x̄)", f"{media_ic:.2f}")
        ic2.metric("Margem de Erro (±E)", f"± {margem_erro:.2f}")
        ic3.metric(f"Limite Inferior ({conf_level:.0%})", f"{ic_inf:.2f}")
        ic4.metric(f"Limite Superior ({conf_level:.0%})", f"{ic_sup:.2f}")
        st.latex(rf"IC_{{{conf_level:.0%}}}(\mu) = \bar{{x}} \pm t_{{\alpha/2, \, n-1}} \cdot \frac{{s}}{{\sqrt{{n}}}} = {media_ic:.2f} \pm {margem_erro:.2f} = [{ic_inf:.2f}; \, {ic_sup:.2f}]")

    st.divider()
    st.subheader("C.2 Execução de Testes de Hipótese e Tomada de Decisão")

    col_th1, col_th2 = st.columns([2, 1])
    with col_th1:
        teste_escolhido = st.selectbox(
            "Selecione o Teste de Hipótese para Executar:",
            options=[
                "1. Teste t de Welch (Duas Amostras Independentes: Velocidade Comercial vs Aviação Leve)",
                "2. Teste Qui-Quadrado de Independência (Nível de Turbulência vs Alerta de Desvio TCAS)",
                "3. Teste t de Student (Uma Amostra: Auditar se RSSI médio é igual a -70 dBm)",
                "4. Teste Z de Proporções (Uma Amostra: Verificar se taxa de Alertas TCAS difere de 10%)"
            ]
        )
    with col_th2:
        alpha = st.selectbox("Nível de Significância (α):", options=[0.01, 0.05, 0.10], index=1)

    if teste_escolhido.startswith("1."):
        st.markdown("**Contexto de Sistema:** Comparar a velocidade média entre aeronaves Comerciais e Aviação Leve (usando a aproximação de **Welch-Satterthwaite** para variâncias populacionais distintas).")
        st.latex(r"H_0: \mu_{\text{comercial}} = \mu_{\text{leve}} \quad \text{vs} \quad H_1: \mu_{\text{comercial}} \neq \mu_{\text{leve}}")
        
        g1 = df_filtrado[df_filtrado['categoria'] == 'Comercial']['velocidade_nos'].dropna()
        g2 = df_filtrado[df_filtrado['categoria'] == 'Aviação Leve']['velocidade_nos'].dropna()
        if len(g1) < 2 or len(g2) < 2:
            st.warning("Para o Teste 1, mantenha as categorias 'Comercial' e 'Aviação Leve' ativas no filtro lateral. Usando a base completa para o cálculo.")
            g1 = df[df['categoria'] == 'Comercial']['velocidade_nos'].dropna()
            g2 = df[df['categoria'] == 'Aviação Leve']['velocidade_nos'].dropna()
        
        stat_calc, p_val = stats.ttest_ind(g1, g2, equal_var=False)
        v1, v2 = g1.var(ddof=1), g2.var(ddof=1)
        n1, n2 = len(g1), len(g2)
        gl = ((v1 / n1 + v2 / n2) ** 2) / (((v1 / n1) ** 2) / (n1 - 1) + ((v2 / n2) ** 2) / (n2 - 1))
        val_critico = stats.t.ppf(1 - alpha / 2, df=gl)
        st.latex(rf"\nu_{{\text{{Welch}}}} = \frac{{\left(\frac{{s_1^2}}{{n_1}} + \frac{{s_2^2}}{{n_2}}\right)^2}}{{\frac{{(s_1^2/n_1)^2}}{{n_1-1}} + \frac{{(s_2^2/n_2)^2}}{{n_2-1}}}} = {gl:.2f} \text{{ graus de liberdade}}")
        nome_stat = f"Estatística t (GL={gl:.1f})"
        str_critico = f"± {val_critico:.4f}"

    elif teste_escolhido.startswith("2."):
        st.markdown("**Contexto de Sistema:** Verificar se a ocorrência de alertas de desvio (TCAS) depende estatisticamente do nível de turbulência enfrentado pela aeronave.")
        st.latex(r"H_0: \text{Turbulência e Alerta são independentes} \quad \text{vs} \quad H_1: \text{Existe dependência estatística entre as variáveis}")
        
        tabela_contingencia = pd.crosstab(df_filtrado['nivel_turbulencia'], df_filtrado['alerta_desvio'])
        stat_calc, p_val, gl, _ = stats.chi2_contingency(tabela_contingencia)
        val_critico = stats.chi2.ppf(1 - alpha, df=gl)
        nome_stat = f"Estatística χ² (GL={gl})"
        str_critico = f"{val_critico:.4f}"

    elif teste_escolhido.startswith("3."):
        st.markdown("**Contexto de Sistema:** Auditar se a potência média de recepção (RSSI) da antena ADS-B atende ao limiar nominal de projeto ($\mu_0 = -70\text{ dBm}$).")
        st.latex(r"H_0: \mu_{\text{RSSI}} = -70\text{ dBm} \quad \text{vs} \quad H_1: \mu_{\text{RSSI}} \neq -70\text{ dBm}")
        
        amostra_rssi = df_filtrado['rssi_dbm'].dropna()
        stat_calc, p_val = stats.ttest_1samp(amostra_rssi, popmean=-70.0)
        gl = len(amostra_rssi) - 1
        val_critico = stats.t.ppf(1 - alpha / 2, df=gl)
        nome_stat = f"Estatística t (GL={gl})"
        str_critico = f"± {val_critico:.4f}"

    else:
        st.markdown("**Contexto de Sistema:** Verificar pelo Teste $Z$ de Proporções se a taxa de acionamento de alertas TCAS difere do limite operacional histórico ($p_0 = 10\%$).")
        st.latex(r"H_0: p = 0,10 \quad \text{vs} \quad H_1: p \neq 0,10")
        
        n_prop = len(df_filtrado)
        p_hat = (df_filtrado['alerta_desvio'] == 'Sim').mean()
        p_0 = 0.10
        stat_calc = (p_hat - p_0) / np.sqrt((p_0 * (1 - p_0)) / n_prop)
        p_val = 2 * (1 - stats.norm.cdf(abs(stat_calc)))
        val_critico = stats.norm.ppf(1 - alpha / 2)
        nome_stat = "Estatística Z Calculada"
        str_critico = f"± {val_critico:.4f}"

    r1, r2, r3, r4 = st.columns(4)
    r1.metric(nome_stat, f"{stat_calc:.4f}")
    r2.metric("Valor Crítico", str_critico)
    r3.metric("P-Valor (p-value)", f"{p_val:.4e}")
    r4.metric("Significância (α)", f"{alpha:.2f}")

    if p_val < alpha:
        st.error(
            f"**Decisão Formal (Rejeita-se $H_0$):** Como o p-valor (`{p_val:.4e}`) é menor que o nível de significância $\\alpha = {alpha}$ "
            f"(e a estatística calculada `{abs(stat_calc):.4f}` ultrapassa o valor crítico `{val_critico:.4f}`), **rejeitamos a Hipótese Nula ($H_0$)**. "
            f"Há evidência estatística significativa para sustentar $H_1$."
        )
    else:
        st.success(
            f"**Decisão Formal (Não se rejeita $H_0$):** Como o p-valor (`{p_val:.4e}`) é maior ou igual a $\\alpha = {alpha}$, "
            f"**não há evidências suficientes para rejeitar a Hipótese Nula ($H_0$)**."
        )
with tab_dados:
    st.subheader("Dicionário de Variáveis e Estrutura da Planilha Excel")
    dicionario = pd.DataFrame([
        {"Variável": "icao24_id", "Tipo Estatístico": "Qualitativa Nominal", "Descrição Técnica": "Identificador hexadecimal único do transponder Mode-S (24 bits)"},
        {"Variável": "categoria", "Tipo Estatístico": "Qualitativa Nominal", "Descrição Técnica": "Categoria operacional da aeronave (Comercial, Aviação Leve, Militar)"},
        {"Variável": "nivel_turbulencia", "Tipo Estatístico": "Qualitativa Ordinal", "Descrição Técnica": "Severidade da instabilidade atmosférica reportada (1-Leve < 2-Moderada < 3-Severa)"},
        {"Variável": "alerta_desvio", "Tipo Estatístico": "Qualitativa Nominal (Binária)", "Descrição Técnica": "Acionamento de alerta de desvio vertical / TCAS (Sim / Não)"},
        {"Variável": "altitude_pes", "Tipo Estatístico": "Quantitativa Contínua", "Descrição Técnica": "Altitude barométrica padrão em pés (ft)"},
        {"Variável": "velocidade_nos", "Tipo Estatístico": "Quantitativa Contínua", "Descrição Técnica": "Velocidade de solo (Ground Speed) em nós (kt)"},
        {"Variável": "rssi_dbm", "Tipo Estatístico": "Quantitativa Contínua", "Descrição Técnica": "Potência do sinal de radiofrequência recebido em 1090 MHz (dBm)"},
        {"Variável": "mensagens_seg", "Tipo Estatístico": "Quantitativa Discreta", "Descrição Técnica": "Taxa de pacotes ADS-B decodificados por segundo (Processo de Poisson)"},
        {"Variável": "satelites_visiveis", "Tipo Estatístico": "Quantitativa Discreta", "Descrição Técnica": "Quantidade de satélites GNSS utilizados na solução de navegação"}
    ])
    st.table(dicionario)
    st.subheader("Amostra Interativa da Base de Dados Filtrada")
    st.dataframe(df_filtrado, use_container_width=True)