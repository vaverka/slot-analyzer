# ==============================================================================
#  app.py - ANALISADOR UNIVERSAL DE SLOTS V7.7 (com seleção de arquivo do repositório)
# ==============================================================================
import json
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os  # Para trabalhar com sistema de arquivos

# --- Configuração ---
CONFIGS_FOLDER = "."  # Pasta com configurações predefinidas

# --- Função auxiliar para obter lista de arquivos da pasta no repositório ---
@st.cache_data
def get_local_config_files(folder_path):
    """
    Obtém uma lista de arquivos JSON da pasta local especificada.
    """
    try:
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
            return sorted(files)
        else:
            # Se a pasta não existir, retorna lista vazia
            # Isso não é um erro, apenas não há arquivos predefinidos
            return []
    except Exception as e:
        # Em caso de outros erros do OS, registra e retorna lista vazia
        # st.write pode ser muito cedo, usa print para registro no servidor
        print(f"Erro ao obter lista de arquivos de {folder_path}: {e}")
        return []

# --- Classe calculadora com inicialização confiável ---
sns.set_theme(style="whitegrid", palette="viridis")

class SlotProbabilityCalculator:
    def __init__(self, config):
        self.config = config
        self.win_probabilities = None
        self.min_bet, self.max_bet, self.max_win_at_min_bet, self.avg_win = None, None, None, None
        self.min_bankroll_formula, self.min_bankroll_calculation = "", ""
        self.volatility = self.config.get('game_config', {}).get('volatility', 'medium')
        self.calculate_all()

    def calculate_all(self):
        """Executa todos os cálculos básicos na ordem correta."""
        self.calculate_min_bankroll()
        self.calculate_win_probabilities()

    def calculate_win_probabilities(self):
        symbols = {s['id']: s for s in self.config.get('symbols', [])}
        if not symbols: self.win_probabilities = {}; return
        probabilities = self.config.get('probabilities', {})
        wild_power = float(probabilities.get('wild_substitution_power', 0.8))
        wild_symbol = next((s for s in symbols.values() if s.get('type') == 'wild'), None)
        wild_prob = float(wild_symbol.get('base_frequency', 0)) if wild_symbol else 0
        win_probs = {}
        for symbol_id, data in symbols.items():
            pure_prob = float(data.get('base_frequency', 0))
            combo_prob = pure_prob + (wild_prob * wild_power if data.get('type') != 'wild' else 0)
            spins_for_99 = math.log(0.01) / math.log(1 - combo_prob) if 0 < combo_prob < 1 else float('inf')
            win_probs[symbol_id] = {'name': data['name'], 'pure_probability': pure_prob, 'combo_probability': combo_prob, 'spins_for_99_prob': spins_for_99}
        self.win_probabilities = {'base': win_probs}

    def calculate_min_bankroll(self):
        bet_range = self.config.get('game_config', {}).get('bet_range', [0.10, 100.00])
        if not isinstance(bet_range, list) or len(bet_range) < 2: bet_range = [0.10, 100.00]
        self.min_bet, self.max_bet = float(bet_range[0]), float(bet_range[1])
        max_win_multiplier = float(self.config.get('probabilities', {}).get('max_win_multiplier', 2000))
        self.max_win_at_min_bet = max_win_multiplier * self.min_bet
        self.avg_win = 0.4 * self.max_win_at_min_bet
        min_bankroll = 0
        if self.volatility == 'high':
            part1, part2 = 100 * self.min_bet, 0.05 * self.avg_win
            self.min_bankroll_formula = "max(100 * Aposta Mín, 5% * Ganho Médio)"
            self.min_bankroll_calculation, min_bankroll = f"max(\${part1:.2f}, \${part2:.2f})", max(part1, part2)
        elif self.volatility == 'medium':
            part1, part2 = 75 * self.min_bet, 0.03 * self.avg_win
            self.min_bankroll_formula = "max(75 * Aposta Mín, 3% * Ganho Médio)"
            self.min_bankroll_calculation, min_bankroll = f"max(\${part1:.2f}, \${part2:.2f})", max(part1, part2)
        else:  # low
            part1, part2 = 50 * self.min_bet, 0.01 * self.avg_win
            self.min_bankroll_formula = "max(50 * Aposta Mín, 1% * Ganho Médio)"
            self.min_bankroll_calculation, min_bankroll = f"max(\${part1:.2f}, \${part2:.2f})", max(part1, part2)
        return round(min_bankroll, 2)

    def generate_bankroll_strategy(self, personal_bankroll, risk_level='medium'):
        min_bankroll = self.calculate_min_bankroll()
        min_bank_advice = []
        if personal_bankroll < min_bankroll:
            pb_formatted = f"{personal_bankroll:,.2f}"
            mb_formatted = f"{min_bankroll:,.2f}"
            min_bank_advice.append(f"🚨 **RISCO CRÍTICO**: Seu bankroll (\${pb_formatted}) está **SIGNIFICATIVAMENTE ABAIXO** do mínimo (\${mb_formatted})!")
            min_bank_advice.append("A probabilidade de perder todo o bankroll antes de uma vitória significativa **excede 95%**. **NÃO RECOMENDAMOS** jogar com este bankroll.")
        else:
            pb_formatted = f"{personal_bankroll:,.2f}"
            mb_formatted = f"{min_bankroll:,.2f}"
            min_bank_advice.append(f"✅ Seu bankroll (\${pb_formatted}) é suficiente para este slot (mínimo: \${mb_formatted}).")
        
        risk_multiplier_map = {'low': 1, 'medium': 2, 'high': 5}
        risk_multiplier = risk_multiplier_map.get(risk_level, 2)
        bankroll_power_base = 50
        bankroll_multiplier = max(1, 1 + math.log10(personal_bankroll / bankroll_power_base)) if personal_bankroll > bankroll_power_base else 1
        theoretical_bet = self.min_bet * risk_multiplier * bankroll_multiplier
        bet_step = self.min_bet
        snapped_bet = math.floor(theoretical_bet / bet_step) * bet_step
        safe_max_bet = min(self.max_bet, personal_bankroll / 20)
        bet_per_spin = max(self.min_bet, min(snapped_bet, safe_max_bet))
        
        tb_formatted = f"{theoretical_bet:.2f}"
        bps_formatted = f"{bet_per_spin:.2f}"
        mbet_formatted = f"{self.min_bet:.2f}"
        
        adjustment_note = ""
        if abs(bet_per_spin - theoretical_bet) > 0.01:
            if bet_per_spin == self.min_bet:
                adjustment_note = f" (Nota: aposta teórica \${tb_formatted} foi **ajustada** para o mínimo possível neste slot)."
            elif bet_per_spin < theoretical_bet:
                 adjustment_note = f" (Nota: aposta teórica \${tb_formatted} foi **reduzida e arredondada** de acordo com o passo de aposta)."
        
        base_win_prob, rtp = float(self.config.get('probabilities', {}).get('base_win_probability', 0.25)), self.config.get('game_config', {}).get('rtp', 0.96)
        bwp_pct = base_win_prob * 100
        losing_spins_count = 10 - int(base_win_prob * 10)
        rtp_pct = rtp * 100
        house_edge_val = 1000 * (1 - rtp)
        hev_formatted = f"{house_edge_val:.2f}"
        
        truth1 = f"Probabilidade de qualquer vitória por giro: **{bwp_pct:.1f}%**. Isso significa que em média **~{losing_spins_count} a cada 10 giros serão perdedores**."
        truth2 = f"**RTP {rtp_pct:.1f}%** significa que para cada \$1.000 apostado, o cassino mantém em média **\${hev_formatted}**."

        harsh_truths = [truth1, truth2]
        
        stop_loss_profile = {'low': 0.25, 'medium': 0.4, 'high': 0.5}
        win_goal_profile = {'low': 0.4, 'medium': 0.6, 'high': 1.0}
        
        sll_val = personal_bankroll * (1-stop_loss_profile[risk_level])
        sll_loss = personal_bankroll * stop_loss_profile[risk_level]
        wgl_val = personal_bankroll * (1+win_goal_profile[risk_level])
        wgl_profit = personal_bankroll * win_goal_profile[risk_level]
        
        sll_val_f = f"{sll_val:.2f}"
        sll_loss_f = f"{sll_loss:.2f}"
        wgl_val_f = f"{wgl_val:.2f}"
        wgl_profit_f = f"{wgl_profit:.2f}"
        
        strategy1 = f"**Aposta recomendada**: Para seu bankroll e nível de risco, a aposta real é **\${bps_formatted}**.{adjustment_note}"
        strategy2 = f"**Gerenciamento de apostas**: Comece com a aposta mínima **\${mbet_formatted}**. Se o jogo for bem, aumente gradualmente a aposta, mas não exceda o recomendado."
        strategy3 = f"**Stop-loss (regra de ferro)**: Pare imediatamente de jogar se seu bankroll cair para **\${sll_val_f}** (perda de \${sll_loss_f})."
        strategy4 = f"**Objetivo de vitória**: Fixe o lucro e pare de jogar se seu bankroll atingir **\${wgl_val_f}** (lucro de \${wgl_profit_f})."
        strategy5 = "**Psicologia**: **NUNCA** tente 'recuperar perdas'. Cada giro é independente."

        optimal_strategy = [strategy1, strategy2, strategy3, strategy4, strategy5]
        
        return {'min_bank_advice': min_bank_advice, 'harsh_truths': harsh_truths, 'optimal_strategy': optimal_strategy, 'bet_per_spin': bet_per_spin}

    def estimate_goal_chance(self, personal_bankroll, desired_win):
        rtp = self.config.get('game_config', {}).get('rtp', 0.96)
        if desired_win <= 0: return {"probability": 1.0}
        if personal_bankroll <= 0: return {"probability": 0.0}
        target_amount = personal_bankroll + desired_win
        effective_bankroll = personal_bankroll * rtp
        probability = effective_bankroll / target_amount
        return {"probability": probability}

    def visualize_win_probabilities(self, level='base'):
        if not self.win_probabilities: return None
        level_data = self.win_probabilities.get(level)
        if not level_data: return None
        df = pd.DataFrame.from_dict(level_data, orient='index').sort_values('combo_probability', ascending=False)
        if df.empty: return None
        df['combo_probability_pct'] = df['combo_probability'] * 100
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x='combo_probability_pct', y='name', data=df, palette='viridis_r', orient='h', hue='name', legend=False, ax=ax)
        ax.set_title(f'Probabilidade de combinação vencedora com símbolo (Nível: {level})', fontsize=16, pad=20)
        ax.set_xlabel('Probabilidade por giro (considerando Wild), %', fontsize=12); ax.set_ylabel('Símbolo', fontsize=12)
        for p in ax.patches:
            width = p.get_width()
            ax.text(width + 0.05, p.get_y() + p.get_height() / 2., f'{width:.3f}%', va='center', fontsize=10)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.tight_layout()
        return fig

    def get_results_table(self, level='base'):
        if not self.win_probabilities: return pd.DataFrame()
        level_data = self.win_probabilities.get(level)
        if not level_data: return pd.DataFrame()
        df = pd.DataFrame.from_dict(level_data, orient='index')
        if df.empty: return df
        df_sorted = df.sort_values(by='combo_probability', ascending=False)
        df_display = pd.DataFrame({
            'Símbolo': df_sorted['name'],
            'Probabilidade Pura (%)': df_sorted['pure_probability'].apply(lambda x: f"{x*100:.2f}%"),
            'Probabilidade Combinação (com Wild, %)': df_sorted['combo_probability'].apply(lambda x: f"{x*100:.2f}%"),
            'Giros para 99% de chance': df_sorted['spins_for_99_prob'].apply(lambda x: f"{int(x)}" if x != float('inf') else "∞")
        })
        return df_display

# --- Bloco principal do aplicativo web Streamlit ---
def main():
    st.set_page_config(page_title="Analisador de Slots", layout="wide", initial_sidebar_state="expanded")
    
    # --- Obtém lista de arquivos locais ---
    local_config_files = get_local_config_files(CONFIGS_FOLDER)
    
    with st.sidebar:
        st.title("🎰 Parâmetros de Análise")
        
        # --- Novo bloco de seleção de fonte de arquivo ---
        file_source = st.radio(
            "Selecione a fonte de configuração:",
            ('Carregar arquivo do computador', 'Selecionar das predefinições'),
            index=0  # Padrão "Carregar arquivo"
        )
        
        config_file = None
        
        if file_source == 'Carregar arquivo do computador':
            config_file = st.file_uploader("1a. Carregue configuração JSON do slot", type="json")
        elif file_source == 'Selecionar das predefinições' and local_config_files:
            selected_filename = st.selectbox(
                "1b. Selecione configuração do slot",
                options=local_config_files,
                format_func=lambda x: x  # Mostra nome do arquivo como está
            )
            if selected_filename:
                # Tenta abrir arquivo da pasta local
                try:
                    full_path = os.path.join(CONFIGS_FOLDER, selected_filename)
                    # Abre arquivo em modo binário e cria objeto BytesIO,
                    # que imita arquivo carregado para st.file_uploader
                    with open(full_path, 'rb') as f:
                        config_file = f.read()
                    # st.file_uploader espera objeto com atributo 'name'
                    # Encapsula bytes em objeto compatível com UploaderFile
                    from io import BytesIO
                    config_file = BytesIO(config_file)
                    config_file.name = selected_filename  # Adiciona nome do arquivo
                except Exception as e:
                     st.error(f"Erro ao carregar arquivo {selected_filename}: {e}")
                     config_file = None
        elif file_source == 'Selecionar das predefinições' and not local_config_files:
             st.info(f"Pasta '{CONFIGS_FOLDER}' não encontrada ou vazia.")
        
        personal_bankroll, desired_win, risk_level, analyze_button = 0, 0, 'medium', False
        if config_file is not None:
            personal_bankroll = st.number_input("2. Seu bankroll inicial ($)", min_value=0.01, value=200.0, step=10.0, format="%.2f")
            desired_win = st.number_input("3. Seu ganho líquido desejado ($)", min_value=1.0, value=500.0, step=10.0, format="%.2f")
            risk_level = st.selectbox("4. Seu nível de risco", options=['low', 'medium', 'high'], index=1).lower()
            analyze_button = st.button("🚀 Executar análise completa", type="primary", use_container_width=True)
    
    st.title("Analisador Universal de Probabilidades de Slots")
    st.markdown("Esta ferramenta ajuda você a entender as chances reais e desenvolver uma estratégia para qualquer slot baseado em seus parâmetros matemáticos.")
    
    if analyze_button and config_file is not None:
        try:
            # Para BytesIO precisa reposicionar ponteiro para início
            if hasattr(config_file, 'seek'):
                config_file.seek(0)
            config = json.load(config_file)
            calculator = SlotProbabilityCalculator(config)
            if personal_bankroll < calculator.min_bet:
                pb_formatted_error = f"{personal_bankroll:.2f}"
                mb_formatted_error = f"{calculator.min_bet:.2f}"
                st.error(f"**Seu bankroll (\${pb_formatted_error}) é menor que a aposta mínima deste slot (\${mb_formatted_error}).**")
                st.warning("Infelizmente, a análise é impossível. Por favor, aumente seu bankroll.")
                st.stop()
            game_config = config.get('game_config', {})
            
            gn_formatted = game_config.get('game_name', 'N/A')
            pb_formatted = f"{personal_bankroll:,.2f}"
            dw_formatted = f"{desired_win:,.2f}"
            rl_formatted = risk_level.capitalize()

            st.header(f"🎰 Análise Completa do Slot: {gn_formatted}", divider="rainbow")
            st.markdown(f"### Seus Parâmetros: Bankroll: \${pb_formatted} | Ganho Desejado: +\${dw_formatted} | Risco: **{rl_formatted}**")
            
            goal_result = calculator.estimate_goal_chance(personal_bankroll, desired_win)
            strategy = calculator.generate_bankroll_strategy(personal_bankroll, risk_level)
            bet_per_spin = strategy.get('bet_per_spin')
            guaranteed_spins = int(personal_bankroll / bet_per_spin) if bet_per_spin > 0 else float('inf')
            
            st.subheader("🎯 Análise do Seu Objetivo", divider="blue")
            col1, col2 = st.columns(2)
            with col1:
                dw_label_formatted = f"{desired_win:,.2f}"
                st.metric(label=f"Chance estimada de ganhar \${dw_label_formatted}", value=f"{goal_result['probability']*100:.4f}%")
            with col2:
                spins_str = f"{guaranteed_spins}" if guaranteed_spins != float('inf') else "∞"
                st.metric(label="Número garantido de giros (com aposta rec.)", value=spins_str)
            
            with st.expander("Como entender esses números? 🤔"):
                st.markdown(f"""
                #### Chance de vitória
                Esta é sua probabilidade matemática de alcançar o objetivo **antes que a vantagem do cassino (RTP < 100%) esgote seu bankroll**.
                #### Número garantido de giros
                Este é o **número real de giros** que você pode fazer com seu bankroll jogando com a **Aposta Recomendada** (\${bet_per_spin:.2f}).
                - **Como a aposta é determinada?** Multiplicamos a aposta mínima do slot (**\${calculator.min_bet:.2f}**) pelo coeficiente de risco (x1-x5) e por um coeficiente não linear do seu bankroll. Então o resultado é **arredondado e ajustado** para caber nos limites reais do slot.
                - **Esta é sua 'margem de segurança' real**: Quanto maior, mais tempo de jogo você tem para alcançar o objetivo.
                """)
            
            st.subheader("📊 Análise Visual de Probabilidades", divider="blue")
            fig = calculator.visualize_win_probabilities()
            if fig: st.pyplot(fig)
            st.dataframe(calculator.get_results_table(), use_container_width=True)
            
            st.header("♟️ Estratégia Pessoal de Jogo", divider="rainbow")
            with st.container(border=True):
                st.subheader("1. Veredito sobre Seu Bankroll")
                for advice in strategy['min_bank_advice']: 
                    st.markdown(f"➡️ {advice}")
            with st.container(border=True):
                st.subheader("2. Fundamentação e Cálculo do Bankroll Mínimo")
                st.markdown("Para a estratégia fazer sentido, seu bankroll deve suportar séries de perdas características desta volatilidade.")
                st.markdown("\n**Dados de origem para cálculo:**")
                st.markdown(f" • **Aposta mínima**: \${calculator.min_bet:.2f}")
                st.markdown(f" • **Ganho máximo com aposta mínima**: \${calculator.max_win_at_min_bet:,.2f}")
                st.markdown(f" • **Ganho médio significativo (com aposta mínima)**: \${calculator.avg_win:,.2f}")
                st.markdown(f" • **Volatilidade**: {calculator.volatility.capitalize()}")
                st.markdown("\n**Processo de cálculo:**")
                st.markdown(f"1. **Fórmula** (para volatilidade {calculator.volatility.capitalize()}): `{calculator.min_bankroll_formula}`")
                st.markdown(f"2. **Substitui valores**: `{calculator.min_bankroll_calculation}`")
                min_bankroll_final_str = ''.join(filter(lambda char: char.isdigit() or char in '.,', strategy['min_bank_advice'][0].split('$')[-1]))
                st.success(f"**Resultado**: O mínimo recomendado final é **\${min_bankroll_final_str}**")
            with st.container(border=True):
                st.subheader("3. Verdade Crua sobre as Chances (sem adornos)")
                for truth in strategy['harsh_truths']: 
                    st.markdown(f"➡️ {truth}")
            with st.container(border=True):
                st.subheader("4. Estratégia Otimizada Passo a Passo")
                for i, step in enumerate(strategy['optimal_strategy'], 1): 
                    st.markdown(f"**Passo {i}**: {step}")
                    
        except json.JSONDecodeError:
            st.error("Erro: O arquivo selecionado não é um JSON válido.")
        except Exception as e:
            st.error(f"Ocorreu um erro ao analisar o arquivo. Certifique-se de que o arquivo JSON tem estrutura correta. Erro: {e}")
    elif analyze_button and config_file is None:
        st.warning("Por favor, carregue um arquivo de configuração JSON do slot ou selecione da lista para iniciar a análise.")

if __name__ == "__main__":
    main()
