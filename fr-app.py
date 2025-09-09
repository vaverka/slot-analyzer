# ==============================================================================
#  app.py - ANALYSEUR UNIVERSEL DE MACHINES À SOUS V7.7 (avec sélection de fichier depuis le dépôt)
# ==============================================================================
import json
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os  # Pour travailler avec le système de fichiers

# --- Configuration ---
CONFIGS_FOLDER = "."  # Dossier avec les configurations prédéfinies

# --- Fonction utilitaire pour obtenir la liste des fichiers du dossier dans le dépôt ---
@st.cache_data
def get_local_config_files(folder_path):
    """
    Obtient une liste de fichiers JSON depuis le dossier local spécifié.
    """
    try:
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
            return sorted(files)
        else:
            # Si le dossier n'existe pas, retourne une liste vide
            # Ce n'est pas une erreur, il n'y a simplement pas de fichiers prédéfinis
            return []
    except Exception as e:
        # En cas d'autres erreurs système, logge et retourne une liste vide
        # st.write peut être trop tôt, utilise print pour la journalisation côté serveur
        print(f"Erreur lors de la récupération de la liste des fichiers depuis {folder_path}: {e}")
        return []

# --- Classe calculatrice avec initialisation fiable ---
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
        """Exécute tous les calculs de base dans le bon ordre."""
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
            self.min_bankroll_formula = "max(100 * Mise Min, 5% * Gain Moyen)"
            self.min_bankroll_calculation, min_bankroll = f"max(\${part1:.2f}, \${part2:.2f})", max(part1, part2)
        elif self.volatility == 'medium':
            part1, part2 = 75 * self.min_bet, 0.03 * self.avg_win
            self.min_bankroll_formula = "max(75 * Mise Min, 3% * Gain Moyen)"
            self.min_bankroll_calculation, min_bankroll = f"max(\${part1:.2f}, \${part2:.2f})", max(part1, part2)
        else:  # low
            part1, part2 = 50 * self.min_bet, 0.01 * self.avg_win
            self.min_bankroll_formula = "max(50 * Mise Min, 1% * Gain Moyen)"
            self.min_bankroll_calculation, min_bankroll = f"max(\${part1:.2f}, \${part2:.2f})", max(part1, part2)
        return round(min_bankroll, 2)

    def generate_bankroll_strategy(self, personal_bankroll, risk_level='medium'):
        min_bankroll = self.calculate_min_bankroll()
        min_bank_advice = []
        if personal_bankroll < min_bankroll:
            pb_formatted = f"{personal_bankroll:,.2f}"
            mb_formatted = f"{min_bankroll:,.2f}"
            min_bank_advice.append(f"🚨 **RISQUE CRITIQUE** : Votre bankroll (\${pb_formatted}) est **SIGNIFICATIVEMENT INFÉRIEURE** au minimum (\${mb_formatted}) !")
            min_bank_advice.append("La probabilité de perdre tout le bankroll avant un gain significatif **dépasse 95%**. Nous **DÉCONSEILLONS** de jouer avec ce bankroll.")
        else:
            pb_formatted = f"{personal_bankroll:,.2f}"
            mb_formatted = f"{min_bankroll:,.2f}"
            min_bank_advice.append(f"✅ Votre bankroll (\${pb_formatted}) est suffisante pour cette machine (minimum : \${mb_formatted}).")
        
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
                adjustment_note = f" (Note : la mise théorique \${tb_formatted} a été **ajustée** au minimum possible sur cette machine)."
            elif bet_per_spin < theoretical_bet:
                 adjustment_note = f" (Note : la mise théorique \${tb_formatted} a été **réduite et arrondie** selon le pas de mise)."
        
        base_win_prob, rtp = float(self.config.get('probabilities', {}).get('base_win_probability', 0.25)), self.config.get('game_config', {}).get('rtp', 0.96)
        bwp_pct = base_win_prob * 100
        losing_spins_count = 10 - int(base_win_prob * 10)
        rtp_pct = rtp * 100
        house_edge_val = 1000 * (1 - rtp)
        hev_formatted = f"{house_edge_val:.2f}"
        
        truth1 = f"Probabilité de gain quelconque par tour : **{bwp_pct:.1f}%**. Cela signifie qu'en moyenne **~{losing_spins_count} tours sur 10 seront perdants**."
        truth2 = f"**RTP {rtp_pct:.1f}%** signifie que pour chaque \$1,000 misé, le casino garde en moyenne **\${hev_formatted}**."

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
        
        strategy1 = f"**Mise recommandée** : Pour votre bankroll et niveau de risque, la mise réelle est **\${bps_formatted}**.{adjustment_note}"
        strategy2 = f"**Gestion des mises** : Commencez avec la mise minimale **\${mbet_formatted}**. Si le jeu se passe bien, augmentez progressivement la mise mais ne dépassez pas la recommandée."
        strategy3 = f"**Stop-loss (règle absolue)** : Arrêtez immédiatement de jouer si votre bankroll descend à **\${sll_val_f}** (perte de \${sll_loss_f})."
        strategy4 = f"**Objectif de gain** : Sécurisez les profits et arrêtez de jouer si votre bankroll atteint **\${wgl_val_f}** (profit de \${wgl_profit_f})."
        strategy5 = "**Psychologie** : **NE TENTEZ JAMAIS** de 'vous rattraper'. Chaque tour est indépendant."

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
        ax.set_title(f'Probabilité de combinaison gagnante avec symbole (Niveau : {level})', fontsize=16, pad=20)
        ax.set_xlabel('Probabilité par tour (avec Wild), %', fontsize=12); ax.set_ylabel('Symbole', fontsize=12)
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
            'Symbole': df_sorted['name'],
            'Probabilité Pure (%)': df_sorted['pure_probability'].apply(lambda x: f"{x*100:.2f}%"),
            'Probabilité Combinaison (avec Wild, %)': df_sorted['combo_probability'].apply(lambda x: f"{x*100:.2f}%"),
            'Tours pour 99% de chance': df_sorted['spins_for_99_prob'].apply(lambda x: f"{int(x)}" if x != float('inf') else "∞")
        })
        return df_display

# --- Bloc principal de l'application web Streamlit ---
def main():
    st.set_page_config(page_title="Analyseur de Machines à Sous", layout="wide", initial_sidebar_state="expanded")
    
    # --- Obtenir la liste des fichiers locaux ---
    local_config_files = get_local_config_files(CONFIGS_FOLDER)
    
    with st.sidebar:
        st.title("🎰 Paramètres d'Analyse")
        
        # --- Nouveau bloc de sélection de source de fichier ---
        file_source = st.radio(
            "Sélectionnez la source de configuration :",
            ('Télécharger un fichier depuis l\'ordinateur', 'Sélectionner parmi les prédéfinis'),
            index=0  # Par défaut "Télécharger un fichier"
        )
        
        config_file = None
        
        if file_source == 'Télécharger un fichier depuis l\'ordinateur':
            config_file = st.file_uploader("1a. Téléchargez la configuration JSON de la machine", type="json")
        elif file_source == 'Sélectionner parmi les prédéfinis' and local_config_files:
            selected_filename = st.selectbox(
                "1b. Sélectionnez la configuration de machine",
                options=local_config_files,
                format_func=lambda x: x  # Afficher le nom du fichier tel quel
            )
            if selected_filename:
                # Essayer d'ouvrir le fichier depuis le dossier local
                try:
                    full_path = os.path.join(CONFIGS_FOLDER, selected_filename)
                    # Ouvrir le fichier en mode binaire et créer un objet BytesIO,
                    # qui imite un fichier téléchargé pour st.file_uploader
                    with open(full_path, 'rb') as f:
                        config_file = f.read()
                    # st.file_uploader attend un objet avec l'attribut 'name'
                    # Encapsuler les bytes dans un objet compatible avec UploaderFile
                    from io import BytesIO
                    config_file = BytesIO(config_file)
                    config_file.name = selected_filename  # Ajouter le nom du fichier
                except Exception as e:
                     st.error(f"Erreur lors du chargement du fichier {selected_filename}: {e}")
                     config_file = None
        elif file_source == 'Sélectionner parmi les prédéfinis' and not local_config_files:
             st.info(f"Dossier '{CONFIGS_FOLDER}' introuvable ou vide.")
        
        personal_bankroll, desired_win, risk_level, analyze_button = 0, 0, 'medium', False
        if config_file is not None:
            personal_bankroll = st.number_input("2. Votre bankroll de départ ($)", min_value=0.01, value=200.0, step=10.0, format="%.2f")
            desired_win = st.number_input("3. Votre gain net désiré ($)", min_value=1.0, value=500.0, step=10.0, format="%.2f")
            risk_level = st.selectbox("4. Votre niveau de risque", options=['low', 'medium', 'high'], index=1).lower()
            analyze_button = st.button("🚀 Lancer l'analyse complète", type="primary", use_container_width=True)
    
    st.title("Analyseur Universel de Probabilités de Machines à Sous")
    st.markdown("Cet outil vous aide à comprendre les vraies probabilités et à développer une stratégie pour toute machine à sous basée sur ses paramètres mathématiques.")
    
    if analyze_button and config_file is not None:
        try:
            # Pour BytesIO, besoin de repositionner le pointeur au début
            if hasattr(config_file, 'seek'):
                config_file.seek(0)
            config = json.load(config_file)
            calculator = SlotProbabilityCalculator(config)
            if personal_bankroll < calculator.min_bet:
                pb_formatted_error = f"{personal_bankroll:.2f}"
                mb_formatted_error = f"{calculator.min_bet:.2f}"
                st.error(f"**Votre bankroll (\${pb_formatted_error}) est inférieure à la mise minimale de cette machine (\${mb_formatted_error}).**")
                st.warning("Malheureusement, l'analyse est impossible. Veuillez augmenter votre bankroll.")
                st.stop()
            game_config = config.get('game_config', {})
            
            gn_formatted = game_config.get('game_name', 'N/A')
            pb_formatted = f"{personal_bankroll:,.2f}"
            dw_formatted = f"{desired_win:,.2f}"
            rl_formatted = risk_level.capitalize()

            st.header(f"🎰 Analyse Complète de la Machine : {gn_formatted}", divider="rainbow")
            st.markdown(f"### Vos Paramètres : Bankroll : \${pb_formatted} | Gain Désiré : +\${dw_formatted} | Risque : **{rl_formatted}**")
            
            goal_result = calculator.estimate_goal_chance(personal_bankroll, desired_win)
            strategy = calculator.generate_bankroll_strategy(personal_bankroll, risk_level)
            bet_per_spin = strategy.get('bet_per_spin')
            guaranteed_spins = int(personal_bankroll / bet_per_spin) if bet_per_spin > 0 else float('inf')
            
            st.subheader("🎯 Analyse de Votre Objectif", divider="blue")
            col1, col2 = st.columns(2)
            with col1:
                dw_label_formatted = f"{desired_win:,.2f}"
                st.metric(label=f"Chance estimée de gagner \${dw_label_formatted}", value=f"{goal_result['probability']*100:.4f}%")
            with col2:
                spins_str = f"{guaranteed_spins}" if guaranteed_spins != float('inf') else "∞"
                st.metric(label="Nombre garanti de tours (avec mise rec.)", value=spins_str)
            
            with st.expander("Comment comprendre ces chiffres ? 🤔"):
                st.markdown(f"""
                #### Chance de gain
                C'est votre probabilité mathématique d'atteindre l'objectif **avant que l'avantage du casino (RTP < 100%) n'épuise votre bankroll**.
                #### Nombre garanti de tours
                C'est le **nombre réel de tours** que vous pouvez faire avec votre bankroll en jouant avec la **Mise Recommandée** (\${bet_per_spin:.2f}).
                - **Comment la mise est-elle déterminée ?** Nous multiplions la mise minimale de la machine (**\${calculator.min_bet:.2f}**) par le coefficient de risque (x1-x5) et par un coefficient non linéaire de votre bankroll. Puis le résultat est **arrondi et ajusté** pour respecter les limites réelles de la machine.
                - **C'est votre vraie 'marge de sécurité'** : Plus elle est grande, plus vous avez de temps de jeu pour atteindre l'objectif.
                """)
            
            st.subheader("📊 Analyse Visuelle des Probabilités", divider="blue")
            fig = calculator.visualize_win_probabilities()
            if fig: st.pyplot(fig)
            st.dataframe(calculator.get_results_table(), use_container_width=True)
            
            st.header("♟️ Stratégie Personnelle de Jeu", divider="rainbow")
            with st.container(border=True):
                st.subheader("1. Verdict sur Votre Bankroll")
                for advice in strategy['min_bank_advice']: 
                    st.markdown(f"➡️ {advice}")
            with st.container(border=True):
                st.subheader("2. Justification et Calcul du Bankroll Minimum")
                st.markdown("Pour que la stratégie ait du sens, votre bankroll doit pouvoir supporter les séries de pertes caractéristiques de cette volatilité.")
                st.markdown("\n**Données sources pour le calcul :**")
                st.markdown(f" • **Mise minimale** : \${calculator.min_bet:.2f}")
                st.markdown(f" • **Gain maximum avec mise minimale** : \${calculator.max_win_at_min_bet:,.2f}")
                st.markdown(f" • **Gain moyen significatif (avec mise minimale)** : \${calculator.avg_win:,.2f}")
                st.markdown(f" • **Volatilité** : {calculator.volatility.capitalize()}")
                st.markdown("\n**Processus de calcul :**")
                st.markdown(f"1. **Formule** (pour volatilité {calculator.volatility.capitalize()}) : `{calculator.min_bankroll_formula}`")
                st.markdown(f"2. **Substitution des valeurs** : `{calculator.min_bankroll_calculation}`")
                min_bankroll_final_str = ''.join(filter(lambda char: char.isdigit() or char in '.,', strategy['min_bank_advice'][0].split('$')[-1]))
                st.success(f"**Résultat** : Le minimum recommandé final est **\${min_bankroll_final_str}**")
            with st.container(border=True):
                st.subheader("3. Dure Vérité sur les Chances (sans fard)")
                for truth in strategy['harsh_truths']: 
                    st.markdown(f"➡️ {truth}")
            with st.container(border=True):
                st.subheader("4. Stratégie Optimale Étape par Étape")
                for i, step in enumerate(strategy['optimal_strategy'], 1): 
                    st.markdown(f"**Étape {i}** : {step}")
                    
        except json.JSONDecodeError:
            st.error("Erreur : Le fichier sélectionné n'est pas un JSON valide.")
        except Exception as e:
            st.error(f"Une erreur s'est produite lors de l'analyse du fichier. Assurez-vous que le fichier JSON a la structure correcte. Erreur : {e}")
    elif analyze_button and config_file is None:
        st.warning("Veuillez télécharger un fichier de configuration JSON de machine à sous ou sélectionner dans la liste pour commencer l'analyse.")

if __name__ == "__main__":
    main()
