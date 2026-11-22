import pandas as pd
import numpy as np
from scipy import stats

np.random.seed(42)
n = 1000

categorias = np.random.choice(['Comercial', 'Aviação Leve', 'Militar'], size=n, p=[0.65, 0.25, 0.10])

ordem_turbulencia = ['1 - Leve', '2 - Moderada', '3 - Severa']
turbulencia = np.random.choice(ordem_turbulencia, size=n, p=[0.70, 0.22, 0.08])

altitudes, velocidades, rssi, alerta_tcas = [], [], [], []

for i in range(n):
    cat = categorias[i]
    turb = turbulencia[i]
    
    if cat == 'Comercial':
        alt = np.random.normal(35000, 2500)
        vel = np.random.normal(460, 28)
        sinal = np.random.normal(-68, 6)
    elif cat == 'Aviação Leve':
        alt = np.random.normal(9500, 1400)
        vel = np.random.normal(145, 18)
        sinal = np.random.normal(-78, 8)
    else:
        alt = np.random.normal(41000, 4200)
        vel = np.random.normal(590, 55)
        sinal = np.random.normal(-64, 7)
        
    altitudes.append(max(1000, round(alt, 2)))
    velocidades.append(max(70, round(vel, 2)))
    rssi.append(round(sinal, 2))
    
    prob_alerta = 0.04 if turb == '1 - Leve' else (0.15 if turb == '2 - Moderada' else 0.35)
    alerta_tcas.append(np.random.choice(['Sim', 'Não'], p=[prob_alerta, 1 - prob_alerta]))

mensagens_por_seg = np.random.poisson(lam=4.2, size=n)
satelites_gnss = np.random.binomial(n=12, p=0.75, size=n)

df = pd.DataFrame({
    'icao24_id': [f'E4{i:04X}' for i in range(n)],
    'categoria': categorias,
    'nivel_turbulencia': turbulencia,
    'alerta_desvio': alerta_tcas,
    'altitude_pes': altitudes,
    'velocidade_nos': velocidades,
    'rssi_dbm': rssi,
    'mensagens_seg': mensagens_por_seg,
    'satelites_visiveis': satelites_gnss
})

df.to_csv('telemetria_adsb.csv', index=False)

def calcular_moda_rigorosa(serie, continua=True):
    s = serie.dropna()
    if not continua:
        return round(float(s.mode().iloc[0]), 4)
    k = int(np.ceil(1 + 3.322 * np.log10(len(s))))
    contagens, limites = np.histogram(s, bins=k)
    idx_modal = int(np.argmax(contagens))
    l_inf = limites[idx_modal]
    h = limites[1] - limites[0]
    f_m = contagens[idx_modal]
    f_ant = contagens[idx_modal - 1] if idx_modal > 0 else 0
    f_pos = contagens[idx_modal + 1] if idx_modal < len(contagens) - 1 else 0
    d1 = f_m - f_ant
    d2 = f_m - f_pos
    moda_czuber = l_inf + (d1 / (d1 + d2)) * h if (d1 + d2) > 0 else l_inf + h / 2
    return round(float(moda_czuber), 4)

vars_quant = [
    ('altitude_pes', True), ('velocidade_nos', True), ('rssi_dbm', True),
    ('mensagens_seg', False), ('satelites_visiveis', False)
]
linhas_desc = []
for col, is_cont in vars_quant:
    s = df[col]
    media = s.mean()
    std = s.std(ddof=1)
    linhas_desc.append({
        'Variável': col,
        'Natureza': 'Contínua' if is_cont else 'Discreta',
        'N': len(s),
        'Média': round(media, 4),
        'Mediana': round(s.median(), 4),
        'Moda (Bruta)': round(float(s.mode().iloc[0]), 4),
        'Moda (Czuber / Classe Modal)': calcular_moda_rigorosa(s, continua=is_cont),
        'Variância Amostral (s²)': round(s.var(ddof=1), 4),
        'Desvio Padrão (s)': round(std, 4),
        'Coef. Variação CV (%)': round((std / abs(media)) * 100, 4),
        'Mínimo': round(s.min(), 4),
        'Máximo': round(s.max(), 4)
    })
df_descritiva = pd.DataFrame(linhas_desc)
df_contingencia_abs = pd.crosstab(df['categoria'], df['nivel_turbulencia'], margins=True)
df_contingencia_rel = (pd.crosstab(df['categoria'], df['nivel_turbulencia'], normalize='all', margins=True) * 100).round(2)

linhas_ic = []
for col, _ in vars_quant[:3]:
    s = df[col]
    m = s.mean()
    ep = stats.sem(s)
    t_crit = stats.t.ppf(0.975, df=len(s) - 1)
    me = t_crit * ep
    linhas_ic.append({
        'Métrica': col,
        'Nível de Confiança': '95%',
        'Média Amostral': round(m, 4),
        'Erro Padrão': round(ep, 4),
        'Valor Crítico (t)': round(t_crit, 4),
        'Margem de Erro (±E)': round(me, 4),
        'IC Limite Inferior': round(m - me, 4),
        'IC Limite Superior': round(m + me, 4)
    })
df_ic = pd.DataFrame(linhas_ic)
g1 = df[df['categoria'] == 'Comercial']['velocidade_nos']
g2 = df[df['categoria'] == 'Aviação Leve']['velocidade_nos']
t_welch, p_welch = stats.ttest_ind(g1, g2, equal_var=False)
v1, v2 = g1.var(ddof=1), g2.var(ddof=1)
n1, n2 = len(g1), len(g2)
gl_welch = ((v1/n1 + v2/n2)**2) / (((v1/n1)**2)/(n1 - 1) + ((v2/n2)**2)/(n2 - 1))
t_crit_welch = stats.t.ppf(1 - 0.05/2, df=gl_welch)

tab_chi = pd.crosstab(df['nivel_turbulencia'], df['alerta_desvio'])
chi2_stat, p_chi2, gl_chi2, _ = stats.chi2_contingency(tab_chi)
chi2_crit = stats.chi2.ppf(1 - 0.05, df=gl_chi2)

t_1s, p_1s = stats.ttest_1samp(df['rssi_dbm'], popmean=-70.0)
gl_1s = len(df['rssi_dbm']) - 1
t_crit_1s = stats.t.ppf(1 - 0.05/2, df=gl_1s)

df_testes = pd.DataFrame([
    {
        'Teste de Hipótese': 'Teste t de Welch (2 Amostras Independentes)',
        'Hipótese Nula (H0)': 'μ_comercial = μ_leve',
        'Hipótese Alternativa (H1)': 'μ_comercial ≠ μ_leve',
        'Estatística Calculada': round(float(t_welch), 4),
        'Graus de Liberdade (GL)': round(float(gl_welch), 2),
        'Valor Crítico (α=0.05)': f"± {t_crit_welch:.4f}",
        'P-Valor': f"{p_welch:.4e}",
        'Decisão Formal': 'Rejeita H0' if p_welch < 0.05 else 'Não Rejeita H0'
    },
    {
        'Teste de Hipótese': 'Qui-Quadrado de Independência (χ²)',
        'Hipótese Nula (H0)': 'Turbulência e Alerta TCAS são independentes',
        'Hipótese Alternativa (H1)': 'Há dependência entre Turbulência e Alerta TCAS',
        'Estatística Calculada': round(float(chi2_stat), 4),
        'Graus de Liberdade (GL)': int(gl_chi2),
        'Valor Crítico (α=0.05)': f"{chi2_crit:.4f}",
        'P-Valor': f"{p_chi2:.4e}",
        'Decisão Formal': 'Rejeita H0' if p_chi2 < 0.05 else 'Não Rejeita H0'
    },
    {
        'Teste de Hipótese': 'Teste t de Student (1 Amostra - RSSI)',
        'Hipótese Nula (H0)': 'μ_RSSI = -70 dBm',
        'Hipótese Alternativa (H1)': 'μ_RSSI ≠ -70 dBm',
        'Estatística Calculada': round(float(t_1s), 4),
        'Graus de Liberdade (GL)': int(gl_1s),
        'Valor Crítico (α=0.05)': f"± {t_crit_1s:.4f}",
        'P-Valor': f"{p_1s:.4e}",
        'Decisão Formal': 'Rejeita H0' if p_1s < 0.05 else 'Não Rejeita H0'
    }
])
with pd.ExcelWriter('telemetria_adsb_analise.xlsx', engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='1_Base_Bruta_ADSB', index=False)
    df_descritiva.to_excel(writer, sheet_name='2_Estatistica_Descritiva', index=False)
    df_contingencia_abs.to_excel(writer, sheet_name='3_Contingencia_Prob', startrow=1)
    df_contingencia_rel.to_excel(writer, sheet_name='3_Contingencia_Prob', startrow=9)
    df_ic.to_excel(writer, sheet_name='4_Intervalos_Confianca', index=False)
    df_testes.to_excel(writer, sheet_name='5_Testes_Hipoteses', index=False)