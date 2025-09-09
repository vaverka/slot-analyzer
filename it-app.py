# ==============================================================================
#  app.py - ANALIZZATORE UNIVERSALE DI SLOT V7.7 (con selezione file da repository)
# ==============================================================================
import json
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os  # Per lavorare con il file system

# --- Configurazione ---
CONFIGS_FOLDER = "."  # Cartella con configurazioni predefinite

# --- Funzione di supporto per ottenere l'elenco dei file dalla cartella nel repository ---
@st.cache_data
def get_local_config_files(folder_path):
    """
    Ottiene un elenco di file JSON dalla cartella locale specificata.
    """
    try:
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
            return sorted(files)
        else:
            # Se la cartella non esiste, restituisce un elenco vuoto
            # Questo non è un errore, semplicemente non ci sono file predefiniti
            return []
    except Exception as e:
        # In caso di altri errori del sistema, registra e restituisce un elenco vuoto
        # st.write potrebbe essere troppo presto, usa print per la registrazione lato server
        print(f"Errore ottenendo l'elenco dei file da {folder_path}: {e}")
        return []

# --- Classe calcolatrice con inizializzazione affidabile ---
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
        """Esegue tutti i calcoli di base nell'ordine corretto."""
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
            self.min_bankroll_formula = "max(100 * Puntata Min, 5% * Vincita Media)"
            self.min_bankroll_calculation, min_bankroll = f"max(\${part1:.2f}, \${part2:.2f})", max(part1, part2)
        elif self.volatility == 'medium':
            part1, part2 = 75 * self.min_bet, 0.03 * self.avg_win
            self.min_bankroll_formula = "max(75 * Puntata Min, 3% * Vincita Media)"
            self.min_bankroll_calculation, min_bankroll = f"max(\${part1:.2f}, \${part2:.2f})", max(part1, part2)
        else:  # low
            part1, part2 = 50 * self.min_bet, 0.01 * self.avg_win
            self.min_bankroll_formula = "max(50 * Puntata Min, 1% * Vincita Media)"
            self.min_bankroll_calculation, min_bankroll = f"max(\${part1:.2f}, \${part2:.2f})", max(part1, part2)
        return round(min_bankroll, 2)

    def generate_bankroll_strategy(self, personal_bankroll, risk_level='medium'):
        min_bankroll = self.calculate_min_bankroll()
        min_bank_advice = []
        if personal_bankroll < min_bankroll:
            pb_formatted = f"{personal_bankroll:,.2f}"
            mb_formatted = f"{min_bankroll:,.2f}"
            min_bank_advice.append(f"🚨 **RISCHIO CRITICO**: Il tuo bankroll (\${pb_formatted}) è **SIGNIFICATIVAMENTE INFERIORE** al minimo (\${mb_formatted})!")
            min_bank_advice.append("La probabilità di perdere l'intero bankroll prima di una vincita significativa **supera il 95%**. **SCONSIGLIAMO** di giocare con questo bankroll.")
        else:
            pb_formatted = f"{personal_bankroll:,.2f}"
            mb_formatted = f"{min_bankroll:,.2f}"
            min_bank_advice.append(f"✅ Il tuo bankroll (\${pb_formatted}) è sufficiente per questa slot (minimo: \${mb_formatted}).")
        
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
                adjustment_note = f" (Nota: la puntata teorica \${tb_formatted} è stata **adattata** al minimo possibile in questa slot)."
            elif bet_per_spin < theoretical_bet:
                 adjustment_note = f" (Nota: la puntata teorica \${tb_formatted} è stata **ridotta e arrotondata** secondo il passo di puntata)."
        
        base_win_prob, rtp = float(self.config.get('probabilities', {}).get('base_win_probability', 0.25)), self.config.get('game_config', {}).get('rtp', 0.96)
        bwp_pct = base_win_prob * 100
        losing_spins_count = 10 - int(base_win_prob * 10)
        rtp_pct = rtp * 100
        house_edge_val = 1000 * (1 - rtp)
        hev_formatted = f"{house_edge_val:.2f}"
        
        truth1 = f"Probabilità di qualsiasi vincita per spin: **{bwp_pct:.1f}%**. Ciò significa che in media **~{losing_spins_count} spin su 10 saranno perdenti**."
        truth2 = f"**RTP {rtp_pct:.1f}%** significa che per ogni \$1,000 scommesso, il casino trattiene in media **\${hev_formatted}**."

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
        
        strategy1 = f"**Puntata raccomandata**: Per il tuo bankroll e livello di rischio, la puntata reale è **\${bps_formatted}**.{adjustment_note}"
        strategy2 = f"**Gestione delle puntate**: Inizia con la puntata minima **\${mbet_formatted}**. Se il gioco va bene, aumenta gradualmente la puntata ma non superare la raccomandata."
        strategy3 = f"**Stop-loss (regola ferrea)**: Interrompi immediatamente il gioco se il tuo bankroll scende a **\${sll_val_f}** (perdita di \${sll_loss_f})."
        strategy4 = f"**Obiettivo vincita**: Blocca i profitti e interrompi il gioco se il tuo bankroll raggiunge **\${wgl_val_f}** (profitto di \${wgl_profit_f})."
        strategy5 = "**Psicologia**: **NON CERCARE MAI** di 'recuperare le perdite'. Ogni spin è indipendente."

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
        ax.set_title(f'Probabilità di combinazione vincente con simbolo (Livello: {level})', fontsize=16, pad=20)
        ax.set_xlabel('Probabilità per spin (con Wild), %', fontsize=12); ax.set_ylabel('Simbolo', fontsize=12)
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
            'Simbolo': df_sorted['name'],
            'Probabilità Pura (%)': df_sorted['pure_probability'].apply(lambda x: f"{x*100:.2f}%"),
            'Probabilità Combinazione (con Wild, %)': df_sorted['combo_probability'].apply(lambda x: f"{x*100:.2f}%"),
            'Spin per 99% di probabilità': df_sorted['spins_for_99_prob'].apply(lambda x: f"{int(x)}" if x != float('inf') else "∞")
        })
        return df_display

# --- Blocco principale dell'applicazione web Streamlit ---
def main():
    st.set_page_config(page_title="Analizzatore di Slot", layout="wide", initial_sidebar_state="expanded")
    
    # --- Ottieni l'elenco dei file locali ---
    local_config_files = get_local_config_files(CONFIGS_FOLDER)
    
    with st.sidebar:
        st.title("🎰 Parametri di Analisi")
        
        # --- Nuovo blocco per la selezione della fonte del file ---
        file_source = st.radio(
            "Seleziona la fonte di configurazione:",
            ('Carica file dal computer', 'Seleziona dai predefiniti'),
            index=0  # Predefinito "Carica file"
        )
        
        config_file = None
        
        if file_source == 'Carica file dal computer':
            config_file = st.file_uploader("1a. Carica configurazione JSON della slot", type="json")
        elif file_source == 'Seleziona dai predefiniti' and local_config_files:
            selected_filename = st.selectbox(
                "1b. Seleziona configurazione slot",
                options=local_config_files,
                format_func=lambda x: x  # Mostra il nome del file così com'è
            )
            if selected_filename:
                # Prova ad aprire il file dalla cartella locale
                try:
                    full_path = os.path.join(CONFIGS_FOLDER, selected_filename)
                    # Apri il file in modalità binaria e crea un oggetto BytesIO,
                    # che simula un file caricato per st.file_uploader
                    with open(full_path, 'rb') as f:
                        config_file = f.read()
                    # st.file_uploader si aspetta un oggetto con attributo 'name'
                    # Incapsula i byte in un oggetto compatibile con UploaderFile
                    from io import BytesIO
                    config_file = BytesIO(config_file)
                    config_file.name = selected_filename  # Aggiungi il nome del file
                except Exception as e:
                     st.error(f"Errore nel caricamento del file {selected_filename}: {e}")
                     config_file = None
        elif file_source == 'Seleziona dai predefiniti' and not local_config_files:
             st.info(f"Cartella '{CONFIGS_FOLDER}' non trovata o vuota.")
        
        personal_bankroll, desired_win, risk_level, analyze_button = 0, 0, 'medium', False
        if config_file is not None:
            personal_bankroll = st.number_input("2. Il tuo bankroll iniziale ($)", min_value=0.01, value=200.0, step=10.0, format="%.2f")
            desired_win = st.number_input("3. La tua vincita netta desiderata ($)", min_value=1.0, value=500.0, step=10.0, format="%.2f")
            risk_level = st.selectbox("4. Il tuo livello di rischio", options=['low', 'medium', 'high'], index=1).lower()
            analyze_button = st.button("🚀 Esegui analisi completa", type="primary", use_container_width=True)
    
    st.title("Analizzatore Universale delle Probabilità delle Slot")
    st.markdown("Questo strumento ti aiuta a comprendere le reali probabilità e sviluppare una strategia per qualsiasi slot basata sui suoi parametri matematici.")
    
    if analyze_button and config_file is not None:
        try:
            # Per BytesIO è necessario riposizionare il puntatore all'inizio
            if hasattr(config_file, 'seek'):
                config_file.seek(0)
            config = json.load(config_file)
            calculator = SlotProbabilityCalculator(config)
            if personal_bankroll < calculator.min_bet:
                pb_formatted_error = f"{personal_bankroll:.2f}"
                mb_formatted_error = f"{calculator.min_bet:.2f}"
                st.error(f"**Il tuo bankroll (\${pb_formatted_error}) è inferiore alla puntata minima in questa slot (\${mb_formatted_error}).**")
                st.warning("Sfortunatamente, l'analisi è impossibile. Per favore aumenta il tuo bankroll.")
                st.stop()
            game_config = config.get('game_config', {})
            
            gn_formatted = game_config.get('game_name', 'N/A')
            pb_formatted = f"{personal_bankroll:,.2f}"
            dw_formatted = f"{desired_win:,.2f}"
            rl_formatted = risk_level.capitalize()

            st.header(f"🎰 Analisi Completa della Slot: {gn_formatted}", divider="rainbow")
            st.markdown(f"### I tuoi Parametri: Bankroll: \${pb_formatted} | Vincita Desiderata: +\${dw_formatted} | Rischio: **{rl_formatted}**")
            
            goal_result = calculator.estimate_goal_chance(personal_bankroll, desired_win)
            strategy = calculator.generate_bankroll_strategy(personal_bankroll, risk_level)
            bet_per_spin = strategy.get('bet_per_spin')
            guaranteed_spins = int(personal_bankroll / bet_per_spin) if bet_per_spin > 0 else float('inf')
            
            st.subheader("🎯 Analisi del Tuo Obiettivo", divider="blue")
            col1, col2 = st.columns(2)
            with col1:
                dw_label_formatted = f"{desired_win:,.2f}"
                st.metric(label=f"Probabilità stimata di vincere \${dw_label_formatted}", value=f"{goal_result['probability']*100:.4f}%")
            with col2:
                spins_str = f"{guaranteed_spins}" if guaranteed_spins != float('inf') else "∞"
                st.metric(label="Numero garantito di spin (con puntata cons.)", value=spins_str)
            
            with st.expander("Come interpretare questi numeri? 🤔"):
                st.markdown(f"""
                #### Probabilità di vincita
                Questa è la tua probabilità matematica di raggiungere l'obiettivo **prima che il vantaggio del casino (RTP < 100%) esaurisca il tuo bankroll**.
                #### Numero garantito di spin
                Questo è il **numero reale di spin** che puoi fare con il tuo bankroll giocando con la **Puntata Raccomandata** (\${bet_per_spin:.2f}).
                - **Come viene determinata la puntata?** Moltiplichiamo la puntata minima della slot (**\${calculator.min_bet:.2f}**) per il coefficiente di rischio (x1-x5) e per un coefficiente non lineare del tuo bankroll. Poi il risultato viene **arrotondato e adattato** per adattarsi ai limiti reali della slot.
                - **Questo è il tuo reale 'margine di sicurezza'**: Più è grande, più tempo di gioco hai per raggiungere l'obiettivo.
                """)
            
            st.subheader("📊 Analisi Visuale delle Probabilità", divider="blue")
            fig = calculator.visualize_win_probabilities()
            if fig: st.pyplot(fig)
            st.dataframe(calculator.get_results_table(), use_container_width=True)
            
            st.header("♟️ Strategia di Gioco Personale", divider="rainbow")
            with st.container(border=True):
                st.subheader("1. Verdetto sul Tuo Bankroll")
                for advice in strategy['min_bank_advice']: 
                    st.markdown(f"➡️ {advice}")
            with st.container(border=True):
                st.subheader("2. Motivazione e Calcolo del Bankroll Minimo")
                st.markdown("Affinché la strategia abbia senso, il tuo bankroll deve poter resistere a serie di perdite caratteristiche di questa volatilità.")
                st.markdown("\n**Dati di origine per il calcolo:**")
                st.markdown(f" • **Puntata minima**: \${calculator.min_bet:.2f}")
                st.markdown(f" • **Vincita massima con puntata minima**: \${calculator.max_win_at_min_bet:,.2f}")
                st.markdown(f" • **Vincita media significativa (con puntata minima)**: \${calculator.avg_win:,.2f}")
                st.markdown(f" • **Volatilità**: {calculator.volatility.capitalize()}")
                st.markdown("\n**Processo di calcolo:**")
                st.markdown(f"1. **Formula** (per volatilità {calculator.volatility.capitalize()}): `{calculator.min_bankroll_formula}`")
                st.markdown(f"2. **Sostituisci valori**: `{calculator.min_bankroll_calculation}`")
                min_bankroll_final_str = ''.join(filter(lambda char: char.isdigit() or char in '.,', strategy['min_bank_advice'][0].split('$')[-1]))
                st.success(f"**Risultato**: Il minimo raccomandato finale è **\${min_bankroll_final_str}**")
            with st.container(border=True):
                st.subheader("3. Dura verità sulle probabilità (senza veli)")
                for truth in strategy['harsh_truths']: 
                    st.markdown(f"➡️ {truth}")
            with st.container(border=True):
                st.subheader("4. Strategia Ottimale Passo-Passo")
                for i, step in enumerate(strategy['optimal_strategy'], 1): 
                    st.markdown(f"**Passo {i}**: {step}")
                    
        except json.JSONDecodeError:
            st.error("Errore: Il file selezionato non è un JSON valido.")
        except Exception as e:
            st.error(f"Si è verificato un errore durante l'analisi del file. Assicurati che il file JSON abbia la struttura corretta. Errore: {e}")
    elif analyze_button and config_file is None:
        st.warning("Per favore carica un file di configurazione JSON della slot o seleziona dalla lista per iniziare l'analisi.")

if __name__ == "__main__":
    main()
