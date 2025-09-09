# ==============================================================================
#  app.py - YLEISKÄYTTÖINEN KOLIKKOPELIANALYSAATTORI V7.7 (tiedostonvalinta repositoriosta)
# ==============================================================================
import json
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os  # Työskentelyä varten tiedostojärjestelmän kanssa

# --- Asetukset ---
CONFIGS_FOLDER = "."  # Kansio ennalta määritetyillä asetuksilla

# --- Apufunktio tiedostolistan hakemiseen repositoriokansiosta ---
@st.cache_data
def get_local_config_files(folder_path):
    """
    Hakee JSON-tiedostolistin määritetystä paikallisesta kansiosta.
    """
    try:
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
            return sorted(files)
        else:
            # Jos kansiota ei ole olemassa, palauta tyhjä lista
            # Tämä ei ole virhe, vain ei ole ennalta määritettyjä tiedostoja
            return []
    except Exception as e:
        # Muiden OS-virheiden sattuessa, kirjaa ja palauta tyhjä lista
        # st.write voi olla liian aikaista, käytä printiä palvelinpuolen lokitukseen
        print(f"Virhe haettaessa tiedostolistaa kohteesta {folder_path}: {e}")
        return []

# --- Laskinluokka luotettavalla alustuksella ---
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
        """Suorittaa kaikki peruslaskennat oikeassa järjestyksessä."""
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
            self.min_bankroll_formula = "max(100 * Min panos, 5% * Keskimääräinen voitto)"
            self.min_bankroll_calculation, min_bankroll = f"max(\${part1:.2f}, \${part2:.2f})", max(part1, part2)
        elif self.volatility == 'medium':
            part1, part2 = 75 * self.min_bet, 0.03 * self.avg_win
            self.min_bankroll_formula = "max(75 * Min panos, 3% * Keskimääräinen voitto)"
            self.min_bankroll_calculation, min_bankroll = f"max(\${part1:.2f}, \${part2:.2f})", max(part1, part2)
        else:  # low
            part1, part2 = 50 * self.min_bet, 0.01 * self.avg_win
            self.min_bankroll_formula = "max(50 * Min panos, 1% * Keskimääräinen voitto)"
            self.min_bankroll_calculation, min_bankroll = f"max(\${part1:.2f}, \${part2:.2f})", max(part1, part2)
        return round(min_bankroll, 2)

    def generate_bankroll_strategy(self, personal_bankroll, risk_level='medium'):
        min_bankroll = self.calculate_min_bankroll()
        min_bank_advice = []
        if personal_bankroll < min_bankroll:
            pb_formatted = f"{personal_bankroll:,.2f}"
            mb_formatted = f"{min_bankroll:,.2f}"
            min_bank_advice.append(f"🚨 **KRITTIINEN RISKI**: Pelirahastosi (\${pb_formatted}) on **HUOMATTAVASTI ALHAIMMAN RAJAN ALAPUOLELLA** (\${mb_formatted})!")
            min_bank_advice.append("Todennäköisyys hävitä koko pelirahasto ennen merkittävää voittoa **ylittää 95%**. **EMME SUOSITTLE** pelaamaan tällä pelirahastolla.")
        else:
            pb_formatted = f"{personal_bankroll:,.2f}"
            mb_formatted = f"{min_bankroll:,.2f}"
            min_bank_advice.append(f"✅ Pelirahastosi (\${pb_formatted}) on riittävä tähän kolikkopeliin (minimi: \${mb_formatted}).")
        
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
                adjustment_note = f" (Huom: Teoreettinen panos \${tb_formatted} on **säädetty** tämän pelin alimpaan mahdolliseen)."
            elif bet_per_spin < theoretical_bet:
                 adjustment_note = f" (Huom: Teoreettinen panos \${tb_formatted} on **vähennetty ja pyöristetty** panosaskeleen mukaan)."
        
        base_win_prob, rtp = float(self.config.get('probabilities', {}).get('base_win_probability', 0.25)), self.config.get('game_config', {}).get('rtp', 0.96)
        bwp_pct = base_win_prob * 100
        losing_spins_count = 10 - int(base_win_prob * 10)
        rtp_pct = rtp * 100
        house_edge_val = 1000 * (1 - rtp)
        hev_formatted = f"{house_edge_val:.2f}"
        
        truth1 = f"Voiton todennäköisyys per kierros: **{bwp_pct:.1f}%**. Tämä tarkoittaa, että keskimäärin **~{losing_spins_count} 10:stä kierroksesta on häviöllisiä**."
        truth2 = f"**RTP {rtp_pct:.1f}%** tarkoittaa, että jokaista \$1,000 panosta kohden, kasino pitää keskimäärin **\${hev_formatted}**."

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
        
        strategy1 = f"**Suositeltu panos**: Pelirahastollesi ja riskitasollesi todellinen panos on **\${bps_formatted}**.{adjustment_note}"
        strategy2 = f"**Panosten hallinta**: Aloita minimipanoksella **\${mbet_formatted}**. Jos peli sujuu hyvin, voit asteittain kasvattaa panosta, mutta älä ylitä suositeltua."
        strategy3 = f"**Stop-loss (rautainen sääntö)**: Lopeta peli välittömästi, jos pelirahastosi putoaa **\${sll_val_f}**:iin (tappio \${sll_loss_f})."
        strategy4 = f"**Voittotavoite**: Lukitse voitot ja lopeta peli, jos pelirahastosi saavuttaa **\${wgl_val_f}**:n (voitto \${wgl_profit_f})."
        strategy5 = "**Psykologia**: **ÄLÄ KOSKAAN** yritä 'voittaa takaisin'. Jokainen kierros on riippumaton."

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
        ax.set_title(f'Voittoyhdistelmän todennäköisyys symbolilla (Taso: {level})', fontsize=16, pad=20)
        ax.set_xlabel('Todennäköisyys per kierros (mukaan lukien Wild), %', fontsize=12); ax.set_ylabel('Symboli', fontsize=12)
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
            'Symboli': df_sorted['name'],
            'Puhdas todennäköisyys (%)': df_sorted['pure_probability'].apply(lambda x: f"{x*100:.2f}%"),
            'Yhdistelmätodennäköisyys (Wildillä, %)': df_sorted['combo_probability'].apply(lambda x: f"{x*100:.2f}%"),
            'Kierroksia 99% todennäköisyydelle': df_sorted['spins_for_99_prob'].apply(lambda x: f"{int(x)}" if x != float('inf') else "∞")
        })
        return df_display

# --- Pääohjelma Streamlit-verkkosovellukselle ---
def main():
    st.set_page_config(page_title="Kolikkopelianalyysaattori", layout="wide", initial_sidebar_state="expanded")
    
    # --- Hae lista paikallisista tiedostoista ---
    local_config_files = get_local_config_files(CONFIGS_FOLDER)
    
    with st.sidebar:
        st.title("🎰 Analyysiparametrit")
        
        # --- Uusi lohko tiedostolähteen valintaan ---
        file_source = st.radio(
            "Valitse konfiguraation lähde:",
            ('Lataa tiedosto tietokoneelta', 'Valitse esimääritetyistä'),
            index=0  # Oletus "Lataa tiedosto"
        )
        
        config_file = None
        
        if file_source == 'Lataa tiedosto tietokoneelta':
            config_file = st.file_uploader("1a. Lataa kolikkopelin JSON-konfiguraatio", type="json")
        elif file_source == 'Valitse esimääritetyistä' and local_config_files:
            selected_filename = st.selectbox(
                "1b. Valitse kolikkopelin konfiguraatio",
                options=local_config_files,
                format_func=lambda x: x  # Näytä tiedostonimi sellaisenaan
            )
            if selected_filename:
                # Yritä avata tiedosto paikallisesta kansiosta
                try:
                    full_path = os.path.join(CONFIGS_FOLDER, selected_filename)
                    # Avaa tiedosto binääritilassa ja luo BytesIO-objekti,
                    # joka matkii st.file_uploaderille ladattua tiedostoa
                    with open(full_path, 'rb') as f:
                        config_file = f.read()
                    # st.file_uploader odottaa objektia, jolla on 'name'-attribuutti
                    # Kääri tavut UploaderFile-yhteensopivaksi objektiksi
                    from io import BytesIO
                    config_file = BytesIO(config_file)
                    config_file.name = selected_filename  # Lisää tiedostonimi
                except Exception as e:
                     st.error(f"Virhe ladatessa tiedostoa {selected_filename}: {e}")
                     config_file = None
        elif file_source == 'Valitse esimääritetyistä' and not local_config_files:
             st.info(f"Kansiota '{CONFIGS_FOLDER}' ei löydy tai se on tyhjä.")
        
        personal_bankroll, desired_win, risk_level, analyze_button = 0, 0, 'medium', False
        if config_file is not None:
            personal_bankroll = st.number_input("2. Alkupelirahastosi ($)", min_value=0.01, value=200.0, step=10.0, format="%.2f")
            desired_win = st.number_input("3. Toivottu nettotuotto ($)", min_value=1.0, value=500.0, step=10.0, format="%.2f")
            risk_level = st.selectbox("4. Riskitasosi", options=['low', 'medium', 'high'], index=1).lower()
            analyze_button = st.button("🚀 Suorita täysi analyysi", type="primary", use_container_width=True)
    
    st.title("Yleiskäyttöinen Kolikkopelien Todennäköisyyksien Analyysaattori")
    st.markdown("Tämä työkalu auttaa sinua ymmärtämään todelliset todennäköisyydet ja kehittämään strategian mille tahansa kolikkopelille sen matemaattisten parametrien perusteella.")
    
    if analyze_button and config_file is not None:
        try:
            # BytesIO:lle täytyy siirtää osoitin alkuun
            if hasattr(config_file, 'seek'):
                config_file.seek(0)
            config = json.load(config_file)
            calculator = SlotProbabilityCalculator(config)
            if personal_bankroll < calculator.min_bet:
                pb_formatted_error = f"{personal_bankroll:.2f}"
                mb_formatted_error = f"{calculator.min_bet:.2f}"
                st.error(f"**Pelirahastosi (\${pb_formatted_error}) on pienempi kuin tämän pelin minimipanos (\${mb_formatted_error}).**")
                st.warning("Valitettavasti analyysi on mahdoton. Ole hyvä ja lisää pelirahastoasi.")
                st.stop()
            game_config = config.get('game_config', {})
            
            gn_formatted = game_config.get('game_name', 'N/A')
            pb_formatted = f"{personal_bankroll:,.2f}"
            dw_formatted = f"{desired_win:,.2f}"
            rl_formatted = risk_level.capitalize()

            st.header(f"🎰 Täydellinen Kolikkopelianalyysi: {gn_formatted}", divider="rainbow")
            st.markdown(f"### Parametrit: Pelirahasto: \${pb_formatted} | Toivottu voitto: +\${dw_formatted} | Riskitaso: **{rl_formatted}**")
            
            goal_result = calculator.estimate_goal_chance(personal_bankroll, desired_win)
            strategy = calculator.generate_bankroll_strategy(personal_bankroll, risk_level)
            bet_per_spin = strategy.get('bet_per_spin')
            guaranteed_spins = int(personal_bankroll / bet_per_spin) if bet_per_spin > 0 else float('inf')
            
            st.subheader("🎯 Tavoitteesi analyysi", divider="blue")
            col1, col2 = st.columns(2)
            with col1:
                dw_label_formatted = f"{desired_win:,.2f}"
                st.metric(label=f"Arvioitu todennäköisyys voittaa \${dw_label_formatted}", value=f"{goal_result['probability']*100:.4f}%")
            with col2:
                spins_str = f"{guaranteed_spins}" if guaranteed_spins != float('inf') else "∞"
                st.metric(label="Taattujen kierrosten määrä (suos. panoksella)", value=spins_str)
            
            with st.expander("Miten ymmärtää näitä lukuja? 🤔"):
                st.markdown(f"""
                #### Voiton todennäköisyys
                Tämä on matemaattinen todennäköisyys saavuttaa tavoite **ennen kuin kasinon etu (RTP < 100%) kuluttaa pelirahastosi**.
                #### Taattujen kierrosten määrä
                Tämä on **todellinen kierrosmäärä**, jonka voit pelata pelirahastollasi **Suositellulla panoksella** (\${bet_per_spin:.2f}).
                - **Miten panos määritetään?** Kerromme pelin minimipanoksen (**\${calculator.min_bet:.2f}**) riskikertoimella (x1-x5) ja epälineaarisella pelirahastokertoimella. Tulos **pyöristetään ja säädetään** sopimaan pelin todellisiin rajoihin.
                - **Tämä on todellinen 'turvamarginaalisi'**: Mitä suurempi se on, sitä enemmän peliaikaa sinulla on tavoitteen saavuttamiseksi.
                """)
            
            st.subheader("📊 Visuaalinen todennäköisyyksien analyysi", divider="blue")
            fig = calculator.visualize_win_probabilities()
            if fig: st.pyplot(fig)
            st.dataframe(calculator.get_results_table(), use_container_width=True)
            
            st.header("♟️ Henkilökohtainen pelistrategia", divider="rainbow")
            with st.container(border=True):
                st.subheader("1. Päätelmä pelirahastostasi")
                for advice in strategy['min_bank_advice']: 
                    st.markdown(f"➡️ {advice}")
            with st.container(border=True):
                st.subheader("2. Perustelu ja vähimmäispelirahaston laskenta")
                st.markdown("Jotta strategialla olisi merkitystä, pelirahastosi täytyy kestää tappioputket, jotka ovat ominaisia tälle volatiliteetille.")
                st.markdown("\n**Laskennan lähdetiedot:**")
                st.markdown(f" • **Minimipanos**: \${calculator.min_bet:.2f}")
                st.markdown(f" • **Maksimivoitto minimipanoksella**: \${calculator.max_win_at_min_bet:,.2f}")
                st.markdown(f" • **Keskimääräinen merkittävä voitto (minimipanoksella)**: \${calculator.avg_win:,.2f}")
                st.markdown(f" • **Volatiliteetti**: {calculator.volatility.capitalize()}")
                st.markdown("\n**Laskentaprosessi:**")
                st.markdown(f"1. **Kaava** ({calculator.volatility.capitalize()}-volatiliteetille): `{calculator.min_bankroll_formula}`")
                st.markdown(f"2. **Sijoita arvot**: `{calculator.min_bankroll_calculation}`")
                min_bankroll_final_str = ''.join(filter(lambda char: char.isdigit() or char in '.,', strategy['min_bank_advice'][0].split('$')[-1]))
                st.success(f"**Tulos**: Lopullinen suositeltu minimi on **\${min_bankroll_final_str}**")
            with st.container(border=True):
                st.subheader("3. Karu totuus todennäköisyyksistä (suoraan)")
                for truth in strategy['harsh_truths']: 
                    st.markdown(f"➡️ {truth}")
            with st.container(border=True):
                st.subheader("4. Optimaalinen askel-askeleelta-strategia")
                for i, step in enumerate(strategy['optimal_strategy'], 1): 
                    st.markdown(f"**Vaihe {i}**: {step}")
                    
        except json.JSONDecodeError:
            st.error("Virhe: Valittu tiedosto ei ole kelvollinen JSON.")
        except Exception as e:
            st.error(f"Tiedoston analysoinnissa tapahtui virhe. Varmista, että JSON-tiedostolla on oikea rakenne. Virhe: {e}")
    elif analyze_button and config_file is None:
        st.warning("Ole hyvä ja lataa kolikkopelin JSON-konfiguraatiotiedosto tai valitse listasta aloittaaksesi analyysin.")

if __name__ == "__main__":
    main()
