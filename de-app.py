# ==============================================================================
#  app.py - UNIVERSALER SLOT-ANALYSATOR V7.7 (mit Dateiauswahl aus Repository)
# ==============================================================================
import json
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os  # Für die Arbeit mit dem Dateisystem

# --- Konfiguration ---
CONFIGS_FOLDER = "."  # Ordner mit vordefinierten Konfigurationen

# --- Hilfsfunktion zum Abrufen der Dateiliste aus dem Repository-Ordner ---
@st.cache_data
def get_local_config_files(folder_path):
    """
    Ruft eine Liste von JSON-Dateien aus dem angegebenen lokalen Ordner ab.
    """
    try:
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
            return sorted(files)
        else:
            # Wenn der Ordner nicht existiert, wird eine leere Liste zurückgegeben
            # Dies ist kein Fehler, es gibt einfach keine vordefinierten Dateien
            return []
    except Exception as e:
        # Bei anderen OS-Fehlern wird protokolliert und eine leere Liste zurückgegeben
        # st.write könnte zu früh sein, daher print für Server-Protokollierung
        print(f"Fehler beim Abrufen der Dateiliste aus {folder_path}: {e}")
        return []

# --- Rechnerklasse mit zuverlässiger Initialisierung ---
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
        """Führt alle grundlegenden Berechnungen in der richtigen Reihenfolge durch."""
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
            self.min_bankroll_formula = "max(100 * Mindestwette, 5% * Durchschnittsgewinn)"
            self.min_bankroll_calculation, min_bankroll = f"max(\${part1:.2f}, \${part2:.2f})", max(part1, part2)
        elif self.volatility == 'medium':
            part1, part2 = 75 * self.min_bet, 0.03 * self.avg_win
            self.min_bankroll_formula = "max(75 * Mindestwette, 3% * Durchschnittsgewinn)"
            self.min_bankroll_calculation, min_bankroll = f"max(\${part1:.2f}, \${part2:.2f})", max(part1, part2)
        else:  # low
            part1, part2 = 50 * self.min_bet, 0.01 * self.avg_win
            self.min_bankroll_formula = "max(50 * Mindestwette, 1% * Durchschnittsgewinn)"
            self.min_bankroll_calculation, min_bankroll = f"max(\${part1:.2f}, \${part2:.2f})", max(part1, part2)
        return round(min_bankroll, 2)

    def generate_bankroll_strategy(self, personal_bankroll, risk_level='medium'):
        min_bankroll = self.calculate_min_bankroll()
        min_bank_advice = []
        if personal_bankroll < min_bankroll:
            pb_formatted = f"{personal_bankroll:,.2f}"
            mb_formatted = f"{min_bankroll:,.2f}"
            min_bank_advice.append(f"🚨 **KRITISCHES RISIKO**: Dein Bankroll (\${pb_formatted}) ist **DEUTLICH UNTER** dem Minimum (\${mb_formatted})!")
            min_bank_advice.append("Die Wahrscheinlichkeit, den gesamten Bankroll vor einem signifikanten Gewinn zu verlieren, **übersteigt 95%**. Wir **EMPFEHLEN NICHT**, mit diesem Bankroll zu spielen.")
        else:
            pb_formatted = f"{personal_bankroll:,.2f}"
            mb_formatted = f"{min_bankroll:,.2f}"
            min_bank_advice.append(f"✅ Dein Bankroll (\${pb_formatted}) ist ausreichend für diesen Slot (Minimum: \${mb_formatted}).")
        
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
                adjustment_note = f" (Hinweis: Die theoretische Wette \${tb_formatted} wurde auf das in diesem Slot mögliche Minimum **angepasst**)."
            elif bet_per_spin < theoretical_bet:
                 adjustment_note = f" (Hinweis: Die theoretische Wette \${tb_formatted} wurde **reduziert und gerundet** gemäß dem Wettschritt)."
        
        base_win_prob, rtp = float(self.config.get('probabilities', {}).get('base_win_probability', 0.25)), self.config.get('game_config', {}).get('rtp', 0.96)
        bwp_pct = base_win_prob * 100
        losing_spins_count = 10 - int(base_win_prob * 10)
        rtp_pct = rtp * 100
        house_edge_val = 1000 * (1 - rtp)
        hev_formatted = f"{house_edge_val:.2f}"
        
        truth1 = f"Wahrscheinlichkeit für einen Gewinn pro Spin: **{bwp_pct:.1f}%**. Das bedeutet, dass im Durchschnitt **~{losing_spins_count} von 10 Spins verloren gehen**."
        truth2 = f"**RTP {rtp_pct:.1f}%** bedeutet, dass das Casino von jedem \$1,000 Einsatz im Durchschnitt **\${hev_formatted} behält**."

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
        
        strategy1 = f"**Empfohlener Einsatz**: Für deinen Bankroll und Risikolevel beträgt der reale Einsatz **\${bps_formatted}**.{adjustment_note}"
        strategy2 = f"**Einsatzmanagement**: Beginne mit dem Mindesteinsatz **\${mbet_formatted}**. Wenn das Spiel gut läuft, kannst du den Einsatz schrittweise erhöhen, aber nicht über die Empfehlung hinaus."
        strategy3 = f"**Stop-Loss (eiserne Regel)**: Beende das Spiel sofort, wenn dein Bankroll auf **\${sll_val_f}** sinkt (Verlust von \${sll_loss_f})."
        strategy4 = f"**Gewinnziel**: Sichere Gewinne und beende das Spiel, wenn dein Bankroll **\${wgl_val_f}** erreicht (Gewinn von \${wgl_profit_f})."
        strategy5 = "**Psychologie**: Versuche **NIEMALS**, Verluste 'zurückzugewinnen'. Jeder Spin ist unabhängig."

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
        ax.set_title(f'Wahrscheinlichkeit einer Gewinnkombination mit Symbol (Level: {level})', fontsize=16, pad=20)
        ax.set_xlabel('Wahrscheinlichkeit pro Spin (mit Wild), %', fontsize=12); ax.set_ylabel('Symbol', fontsize=12)
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
            'Reine Wahrscheinlichkeit (%)': df_sorted['pure_probability'].apply(lambda x: f"{x*100:.2f}%"),
            'Kombinationswahrscheinlichkeit (mit Wild, %)': df_sorted['combo_probability'].apply(lambda x: f"{x*100:.2f}%"),
            'Spins für 99% Chance': df_sorted['spins_for_99_prob'].apply(lambda x: f"{int(x)}" if x != float('inf') else "∞")
        })
        return df_display

# --- Hauptblock der Streamlit Webanwendung ---
def main():
    st.set_page_config(page_title="Slot-Analysator", layout="wide", initial_sidebar_state="expanded")
    
    # --- Liste der lokalen Dateien abrufen ---
    local_config_files = get_local_config_files(CONFIGS_FOLDER)
    
    with st.sidebar:
        st.title("🎰 Analyseparameter")
        
        # --- Neuer Block zur Auswahl der Dateiquelle ---
        file_source = st.radio(
            "Wähle die Konfigurationsquelle:",
            ('Datei vom Computer hochladen', 'Aus vordefinierten auswählen'),
            index=0  # Standardmäßig "Datei hochladen"
        )
        
        config_file = None
        
        if file_source == 'Datei vom Computer hochladen':
            config_file = st.file_uploader("1a. Lade Slot-JSON-Konfiguration hoch", type="json")
        elif file_source == 'Aus vordefinierten auswählen' and local_config_files:
            selected_filename = st.selectbox(
                "1b. Wähle Slot-Konfiguration",
                options=local_config_files,
                format_func=lambda x: x  # Dateiname unverändert anzeigen
            )
            if selected_filename:
                # Versuche, Datei aus lokalem Ordner zu öffnen
                try:
                    full_path = os.path.join(CONFIGS_FOLDER, selected_filename)
                    # Datei im Binärmodus öffnen und BytesIO-Objekt erstellen,
                    # das eine hochgeladene Datei für st.file_uploader simuliert
                    with open(full_path, 'rb') as f:
                        config_file = f.read()
                    # st.file_uploader erwartet ein Objekt mit 'name'-Attribut
                    # Bytes in ein mit UploaderFile kompatibles Objekt einpacken
                    from io import BytesIO
                    config_file = BytesIO(config_file)
                    config_file.name = selected_filename  # Dateinamen hinzufügen
                except Exception as e:
                     st.error(f"Fehler beim Laden der Datei {selected_filename}: {e}")
                     config_file = None
        elif file_source == 'Aus vordefinierten auswählen' and not local_config_files:
             st.info(f"Ordner '{CONFIGS_FOLDER}' nicht gefunden oder leer.")
        
        personal_bankroll, desired_win, risk_level, analyze_button = 0, 0, 'medium', False
        if config_file is not None:
            personal_bankroll = st.number_input("2. Dein Startbankroll ($)", min_value=0.01, value=200.0, step=10.0, format="%.2f")
            desired_win = st.number_input("3. Dein gewünschter Nettogewinn ($)", min_value=1.0, value=500.0, step=10.0, format="%.2f")
            risk_level = st.selectbox("4. Dein Risikolevel", options=['low', 'medium', 'high'], index=1).lower()
            analyze_button = st.button("🚀 Vollständige Analyse durchführen", type="primary", use_container_width=True)
    
    st.title("Universal Slot Wahrscheinlichkeitsanalysator")
    st.markdown("Dieses Tool hilft dir, die realen Chancen zu verstehen und eine Strategie für jeden Slot basierend auf seinen mathematischen Parametern zu entwickeln.")
    
    if analyze_button and config_file is not None:
        try:
            # Für BytesIO muss der Zeiger an den Anfang gesetzt werden
            if hasattr(config_file, 'seek'):
                config_file.seek(0)
            config = json.load(config_file)
            calculator = SlotProbabilityCalculator(config)
            if personal_bankroll < calculator.min_bet:
                pb_formatted_error = f"{personal_bankroll:.2f}"
                mb_formatted_error = f"{calculator.min_bet:.2f}"
                st.error(f"**Dein Bankroll (\${pb_formatted_error}) ist geringer als der Mindesteinsatz in diesem Slot (\${mb_formatted_error}).**")
                st.warning("Leider ist die Analyse unmöglich. Bitte erhöhe dein Bankroll.")
                st.stop()
            game_config = config.get('game_config', {})
            
            gn_formatted = game_config.get('game_name', 'N/A')
            pb_formatted = f"{personal_bankroll:,.2f}"
            dw_formatted = f"{desired_win:,.2f}"
            rl_formatted = risk_level.capitalize()

            st.header(f"🎰 Vollständige Slot-Analyse: {gn_formatted}", divider="rainbow")
            st.markdown(f"### Deine Parameter: Bankroll: \${pb_formatted} | Gewünschter Gewinn: +\${dw_formatted} | Risiko: **{rl_formatted}**")
            
            goal_result = calculator.estimate_goal_chance(personal_bankroll, desired_win)
            strategy = calculator.generate_bankroll_strategy(personal_bankroll, risk_level)
            bet_per_spin = strategy.get('bet_per_spin')
            guaranteed_spins = int(personal_bankroll / bet_per_spin) if bet_per_spin > 0 else float('inf')
            
            st.subheader("🎯 Analyse deines Ziels", divider="blue")
            col1, col2 = st.columns(2)
            with col1:
                dw_label_formatted = f"{desired_win:,.2f}"
                st.metric(label=f"Geschätzte Chance, \${dw_label_formatted} zu gewinnen", value=f"{goal_result['probability']*100:.4f}%")
            with col2:
                spins_str = f"{guaranteed_spins}" if guaranteed_spins != float('inf') else "∞"
                st.metric(label="Garantierte Anzahl von Spins (bei empf. Einsatz)", value=spins_str)
            
            with st.expander("Wie sind diese Zahlen zu verstehen? 🤔"):
                st.markdown(f"""
                #### Gewinnchance
                Dies ist deine mathematische Wahrscheinlichkeit, das Ziel **zu erreichen, bevor der Casino-Vorteil (RTP < 100%) dein Bankroll aufbraucht**.
                #### Garantierte Anzahl von Spins
                Dies ist die **tatsächliche Anzahl von Spins**, die du mit deinem Bankroll bei der **Empfohlenen Wette** (\${bet_per_spin:.2f}) machen kannst.
                - **Wie wird der Einsatz bestimmt?** Wir multiplizieren den Mindesteinsatz des Slots (**\${calculator.min_bet:.2f}**) mit dem Risikofaktor (x1-x5) und einem nichtlinearen Bankroll-Faktor. Dann wird das Ergebnis **gerundet und angepasst**, um es an die realen Limits des Slots anzupassen.
                - **Dies ist deine tatsächliche 'Sicherheitsmarge'**: Je größer sie ist, desto mehr Spielzeit hast du, um das Ziel zu erreichen.
                """)
            
            st.subheader("📊 Visuelle Wahrscheinlichkeitsanalyse", divider="blue")
            fig = calculator.visualize_win_probabilities()
            if fig: st.pyplot(fig)
            st.dataframe(calculator.get_results_table(), use_container_width=True)
            
            st.header("♟️ Persönliche Spielstrategie", divider="rainbow")
            with st.container(border=True):
                st.subheader("1. Urteil über dein Bankroll")
                for advice in strategy['min_bank_advice']: 
                    st.markdown(f"➡️ {advice}")
            with st.container(border=True):
                st.subheader("2. Begründung und Berechnung des Mindestbankrolls")
                st.markdown("Damit die Strategie Sinn macht, muss dein Bankroll Verlustserien standhalten, die für diese Volatilität charakteristisch sind.")
                st.markdown("\n**Quelldaten für die Berechnung:**")
                st.markdown(f" • **Mindesteinsatz**: \${calculator.min_bet:.2f}")
                st.markdown(f" • **Maximaler Gewinn bei Mindesteinsatz**: \${calculator.max_win_at_min_bet:,.2f}")
                st.markdown(f" • **Durchschnittlicher signifikanter Gewinn (bei Mindesteinsatz)**: \${calculator.avg_win:,.2f}")
                st.markdown(f" • **Volatilität**: {calculator.volatility.capitalize()}")
                st.markdown("\n**Berechnungsprozess:**")
                st.markdown(f"1. **Formel** (für {calculator.volatility.capitalize()}-Volatilität): `{calculator.min_bankroll_formula}`")
                st.markdown(f"2. **Werte einsetzen**: `{calculator.min_bankroll_calculation}`")
                min_bankroll_final_str = ''.join(filter(lambda char: char.isdigit() or char in '.,', strategy['min_bank_advice'][0].split('$')[-1]))
                st.success(f"**Ergebnis**: Die endgültige empfohlene Mindestsumme beträgt **\${min_bankroll_final_str}**")
            with st.container(border=True):
                st.subheader("3. Harte Wahrheit über die Chancen (ungeschminkt)")
                for truth in strategy['harsh_truths']: 
                    st.markdown(f"➡️ {truth}")
            with st.container(border=True):
                st.subheader("4. Optimale Schritt-für-Schritt-Strategie")
                for i, step in enumerate(strategy['optimal_strategy'], 1): 
                    st.markdown(f"**Schritt {i}**: {step}")
                    
        except json.JSONDecodeError:
            st.error("Fehler: Die ausgewählte Datei ist kein gültiges JSON.")
        except Exception as e:
            st.error(f"Beim Analysieren der Datei ist ein Fehler aufgetreten. Stelle sicher, dass die JSON-Datei die korrekte Struktur hat. Fehler: {e}")
    elif analyze_button and config_file is None:
        st.warning("Bitte lade eine Slot-JSON-Konfigurationsdatei hoch oder wähle aus der Liste aus, um die Analyse zu starten.")

if __name__ == "__main__":
    main()
