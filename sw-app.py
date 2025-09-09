# ==============================================================================
#  app.py - UNIVERSELL SLOTTMASKINSANALYSATOR V7.7 (med filval från repository)
# ==============================================================================
import json
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os  # För att arbeta med filsystemet

# --- Konfiguration ---
CONFIGS_FOLDER = "."  # Mapp med förinställda konfigurationer

# --- Hjälpfunktion för att hämta fillista från mapp i repository ---
@st.cache_data
def get_local_config_files(folder_path):
    """
    Hämtar en lista med JSON-filer från den angivna lokala mappen.
    """
    try:
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
            return sorted(files)
        else:
            # Om mappen inte finns, returnera tom lista
            # Detta är inte ett fel, det finns bara inga förinställda filer
            return []
    except Exception as e:
        # Vid andra OS-fel, logga och returnera tom lista
        # st.write kan vara för tidigt, använd print för serverloggning
        print(f"Fel vid hämtning av fillista från {folder_path}: {e}")
        return []

# --- Kalkylatorklass med pålitlig initiering ---
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
        """Kör alla grundläggande beräkningar i rätt ordning."""
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
            self.min_bankroll_formula = "max(100 * Min insats, 5% * Genomsnittlig vinst)"
            self.min_bankroll_calculation, min_bankroll = f"max(\${part1:.2f}, \${part2:.2f})", max(part1, part2)
        elif self.volatility == 'medium':
            part1, part2 = 75 * self.min_bet, 0.03 * self.avg_win
            self.min_bankroll_formula = "max(75 * Min insats, 3% * Genomsnittlig vinst)"
            self.min_bankroll_calculation, min_bankroll = f"max(\${part1:.2f}, \${part2:.2f})", max(part1, part2)
        else:  # low
            part1, part2 = 50 * self.min_bet, 0.01 * self.avg_win
            self.min_bankroll_formula = "max(50 * Min insats, 1% * Genomsnittlig vinst)"
            self.min_bankroll_calculation, min_bankroll = f"max(\${part1:.2f}, \${part2:.2f})", max(part1, part2)
        return round(min_bankroll, 2)

    def generate_bankroll_strategy(self, personal_bankroll, risk_level='medium'):
        min_bankroll = self.calculate_min_bankroll()
        min_bank_advice = []
        if personal_bankroll < min_bankroll:
            pb_formatted = f"{personal_bankroll:,.2f}"
            mb_formatted = f"{min_bankroll:,.2f}"
            min_bank_advice.append(f"🚨 **KRITISK RISK**: Ditt bankrulle (\${pb_formatted}) är **BETYDLIGT UNDER** minimum (\${mb_formatted})!")
            min_bank_advice.append("Sannolikheten att förlora hela bankrullen innan en betydande vinst **överstiger 95%**. Vi **RECKOMMENDERAR INTE** att spela med detta bankrulle.")
        else:
            pb_formatted = f"{personal_bankroll:,.2f}"
            mb_formatted = f"{min_bankroll:,.2f}"
            min_bank_advice.append(f"✅ Ditt bankrulle (\${pb_formatted}) är tillräckligt för denna spelautomat (minimum: \${mb_formatted}).")
        
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
                adjustment_note = f" (Notera: Teoretisk insats \${tb_formatted} har **justerats** till lägsta möjliga på denna spelautomat)."
            elif bet_per_spin < theoretical_bet:
                 adjustment_note = f" (Notera: Teoretisk insats \${tb_formatted} har **reducerats och avrundats** enligt insatssteget)."
        
        base_win_prob, rtp = float(self.config.get('probabilities', {}).get('base_win_probability', 0.25)), self.config.get('game_config', {}).get('rtp', 0.96)
        bwp_pct = base_win_prob * 100
        losing_spins_count = 10 - int(base_win_prob * 10)
        rtp_pct = rtp * 100
        house_edge_val = 1000 * (1 - rtp)
        hev_formatted = f"{house_edge_val:.2f}"
        
        truth1 = f"Sannolikhet för någon vinst per spinn: **{bwp_pct:.1f}%**. Det betyder att i genomsnitt **~{losing_spins_count} av 10 spinn kommer att vara förlorande**."
        truth2 = f"**RTP {rtp_pct:.1f}%** betyder att för varje \$1,000 satsat, behåller casinot i genomsnitt **\${hev_formatted}**."

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
        
        strategy1 = f"**Rekommenderad insats**: För ditt bankrulle och risikoinivå är den reala insatsen **\${bps_formatted}**.{adjustment_note}"
        strategy2 = f"**Insatshantering**: Börja med minimiinsatsen **\${mbet_formatted}**. Om spelet går bra, kan du gradvis öka insatsen men överskrid inte rekommendationen."
        strategy3 = f"**Stop-loss (järnregel)**: Sluta omedelbart spela om ditt bankrulle sjunker till **\${sll_val_f}** (förlust av \${sll_loss_f})."
        strategy4 = f"**Vinstmål**: Säkra vinster och sluta spela om ditt bankrulle når **\${wgl_val_f}** (vinst på \${wgl_profit_f})."
        strategy5 = "**Psykologi**: **FÖRSÖK ALDRIG** att 'vinna tillbaka'. Varje spinn är oberoende."

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
        ax.set_title(f'Sannolikhet för vinstkombination med symbol (Nivå: {level})', fontsize=16, pad=20)
        ax.set_xlabel('Sannolikhet per spinn (med Wild), %', fontsize=12); ax.set_ylabel('Symbol', fontsize=12)
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
            'Symbol': df_sorted['name'],
            'Ren sannolikhet (%)': df_sorted['pure_probability'].apply(lambda x: f"{x*100:.2f}%"),
            'Kombinationssannolikhet (med Wild, %)': df_sorted['combo_probability'].apply(lambda x: f"{x*100:.2f}%"),
            'Spinn för 99% chans': df_sorted['spins_for_99_prob'].apply(lambda x: f"{int(x)}" if x != float('inf') else "∞")
        })
        return df_display

# --- Huvudblock för Streamlit-webbapplikation ---
def main():
    st.set_page_config(page_title="Spelautomatsanalysator", layout="wide", initial_sidebar_state="expanded")
    
    # --- Hämta lista med lokala filer ---
    local_config_files = get_local_config_files(CONFIGS_FOLDER)
    
    with st.sidebar:
        st.title("🎰 Analysparametrar")
        
        # --- Nytt block för filkällval ---
        file_source = st.radio(
            "Välj konfigurationskälla:",
            ('Ladda upp fil från dator', 'Välj från förinställda'),
            index=0  # Standard "Ladda upp fil"
        )
        
        config_file = None
        
        if file_source == 'Ladda upp fil från dator':
            config_file = st.file_uploader("1a. Ladda upp spelautomats-JSON-konfiguration", type="json")
        elif file_source == 'Välj från förinställda' and local_config_files:
            selected_filename = st.selectbox(
                "1b. Välj spelautomatskonfiguration",
                options=local_config_files,
                format_func=lambda x: x  # Visa filnamn som det är
            )
            if selected_filename:
                # Försök öppna fil från lokal mapp
                try:
                    full_path = os.path.join(CONFIGS_FOLDER, selected_filename)
                    # Öppna fil i binärt läge och skapa BytesIO-objekt,
                    # som imiterar uppladdad fil för st.file_uploader
                    with open(full_path, 'rb') as f:
                        config_file = f.read()
                    # st.file_uploader förväntar sig objekt med 'name'-attribut
                    # Radbytes i ett med UploaderFile kompatibelt objekt
                    from io import BytesIO
                    config_file = BytesIO(config_file)
                    config_file.name = selected_filename  # Lägg till filnamn
                except Exception as e:
                     st.error(f"Fel vid laddning av fil {selected_filename}: {e}")
                     config_file = None
        elif file_source == 'Välj från förinställda' and not local_config_files:
             st.info(f"Mappen '{CONFIGS_FOLDER}' hittades inte eller är tom.")
        
        personal_bankroll, desired_win, risk_level, analyze_button = 0, 0, 'medium', False
        if config_file is not None:
            personal_bankroll = st.number_input("2. Din startbankrulle ($)", min_value=0.01, value=200.0, step=10.0, format="%.2f")
            desired_win = st.number_input("3. Din önskade nettovinst ($)", min_value=1.0, value=500.0, step=10.0, format="%.2f")
            risk_level = st.selectbox("4. Din risikonivå", options=['low', 'medium', 'high'], index=1).lower()
            analyze_button = st.button("🚀 Genomför fullständig analys", type="primary", use_container_width=True)
    
    st.title("Universell Spelautomats Sannolikhetsanalysator")
    st.markdown("Detta verktyg hjälper dig att förstå de verkliga oddsen och utveckla en strategi för vilken spelautomat som helst baserad på dess matematiska parametrar.")
    
    if analyze_button and config_file is not None:
        try:
            # För BytesIO behöver pekaren flyttas till början
            if hasattr(config_file, 'seek'):
                config_file.seek(0)
            config = json.load(config_file)
            calculator = SlotProbabilityCalculator(config)
            if personal_bankroll < calculator.min_bet:
                pb_formatted_error = f"{personal_bankroll:.2f}"
                mb_formatted_error = f"{calculator.min_bet:.2f}"
                st.error(f"**Din bankrulle (\${pb_formatted_error}) är mindre än minimiinsatsen på denna spelautomat (\${mb_formatted_error}).**")
                st.warning("Tyvärr är analys omöjlig. Vänligen öka din bankrulle.")
                st.stop()
            game_config = config.get('game_config', {})
            
            gn_formatted = game_config.get('game_name', 'N/A')
            pb_formatted = f"{personal_bankroll:,.2f}"
            dw_formatted = f"{desired_win:,.2f}"
            rl_formatted = risk_level.capitalize()

            st.header(f"🎰 Fullständig Spelautomatsanalys: {gn_formatted}", divider="rainbow")
            st.markdown(f"### Dina Parametrar: Bankrulle: \${pb_formatted} | Önskad vinst: +\${dw_formatted} | Risk: **{rl_formatted}**")
            
            goal_result = calculator.estimate_goal_chance(personal_bankroll, desired_win)
            strategy = calculator.generate_bankroll_strategy(personal_bankroll, risk_level)
            bet_per_spin = strategy.get('bet_per_spin')
            guaranteed_spins = int(personal_bankroll / bet_per_spin) if bet_per_spin > 0 else float('inf')
            
            st.subheader("🎯 Analys av Ditt Mål", divider="blue")
            col1, col2 = st.columns(2)
            with col1:
                dw_label_formatted = f"{desired_win:,.2f}"
                st.metric(label=f"Uppskattad chans att vinna \${dw_label_formatted}", value=f"{goal_result['probability']*100:.4f}%")
            with col2:
                spins_str = f"{guaranteed_spins}" if guaranteed_spins != float('inf') else "∞"
                st.metric(label="Garanterat antal spinn (med rek. insats)", value=spins_str)
            
            with st.expander("Hur förstår man dessa siffror? 🤔"):
                st.markdown(f"""
                #### Vinstchans
                Detta är din matematiska sannolikhet att uppnå målet **innan casinots fördel (RTP < 100%) tar slut på din bankrulle**.
                #### Garanterat antal spinn
                Detta är det **verkliga antalet spinn** du kan göra med din bankrulle när du spelar med **Rekommenderad insats** (\${bet_per_spin:.2f}).
                - **Hur bestäms insatsen?** Vi multiplicerar spelautomatsens minimiinsats (**\${calculator.min_bet:.2f}**) med riskfaktorn (x1-x5) och med en icke-linjär bankrullefaktor. Sedan **rundas resultatet av och justeras** för att passa spelautomatsens verkliga gränser.
                - **Detta är din verkliga 'säkerhetsmarginal'**: Ju större den är, desto mer speltid har du för att uppnå målet.
                """)
            
            st.subheader("📊 Visuell Sannolikhetsanalys", divider="blue")
            fig = calculator.visualize_win_probabilities()
            if fig: st.pyplot(fig)
            st.dataframe(calculator.get_results_table(), use_container_width=True)
            
            st.header("♟️ Personlig Spellstrategi", divider="rainbow")
            with st.container(border=True):
                st.subheader("1. Utlåtande om Din Bankrulle")
                for advice in strategy['min_bank_advice']: 
                    st.markdown(f"➡️ {advice}")
            with st.container(border=True):
                st.subheader("2. Motivering och Beräkning av Minimibankrulle")
                st.markdown("För att strategin ska vara meningsfull måste din bankrulle kunna överleva förlustserier som är karakteristiska för denna volatilitet.")
                st.markdown("\n**Källdata för beräkning:**")
                st.markdown(f" • **Minimiinsats**: \${calculator.min_bet:.2f}")
                st.markdown(f" • **Maxvinst vid minimiinsats**: \${calculator.max_win_at_min_bet:,.2f}")
                st.markdown(f" • **Genomsnittlig signifikant vinst (vid minimiinsats)**: \${calculator.avg_win:,.2f}")
                st.markdown(f" • **Volatilitet**: {calculator.volatility.capitalize()}")
                st.markdown("\n**Beräkningsprocess:**")
                st.markdown(f"1. **Formel** (för {calculator.volatility.capitalize()} volatilitet): `{calculator.min_bankroll_formula}`")
                st.markdown(f"2. **Sätt in värden**: `{calculator.min_bankroll_calculation}`")
                min_bankroll_final_str = ''.join(filter(lambda char: char.isdigit() or char in '.,', strategy['min_bank_advice'][0].split('$')[-1]))
                st.success(f"**Resultat**: Slutliga rekommenderade minimum är **\${min_bankroll_final_str}**")
            with st.container(border=True):
                st.subheader("3. Hård sanning om oddsen (osanerad)")
                for truth in strategy['harsh_truths']: 
                    st.markdown(f"➡️ {truth}")
            with st.container(border=True):
                st.subheader("4. Optimal Steg-för-Steg Strategi")
                for i, step in enumerate(strategy['optimal_strategy'], 1): 
                    st.markdown(f"**Steg {i}**: {step}")
                    
        except json.JSONDecodeError:
            st.error("Fel: Den valda filen är inte en giltig JSON.")
        except Exception as e:
            st.error(f"Ett fel inträffade vid analys av filen. Se till att JSON-filen har korrekt struktur. Fel: {e}")
    elif analyze_button and config_file is None:
        st.warning("Vänligen ladda upp en spelautomats-JSON-konfigurationsfil eller välj från listan för att börja analysen.")

if __name__ == "__main__":
    main()
