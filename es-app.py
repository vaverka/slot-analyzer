# ==============================================================================
#  app.py - ANALIZADOR UNIVERSAL DE TRAGAMONEDAS V7.7 (con selección de archivo desde repositorio)
# ==============================================================================
import json
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os  # Para trabajar con el sistema de archivos

# --- Configuración ---
CONFIGS_FOLDER = "."  # Carpeta con configuraciones predefinidas

# --- Función auxiliar para obtener lista de archivos de la carpeta en el repositorio ---
@st.cache_data
def get_local_config_files(folder_path):
    """
    Obtiene una lista de archivos JSON de la carpeta local especificada.
    """
    try:
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
            return sorted(files)
        else:
            # Si la carpeta no existe, retorna lista vacía
            # Esto no es un error, simplemente no hay archivos predefinidos
            return []
    except Exception as e:
        # En caso de otros errores del sistema, registra y retorna lista vacía
        # st.write podría ser muy temprano, usa print para registro en el servidor
        print(f"Error obteniendo lista de archivos de {folder_path}: {e}")
        return []

# --- Clase calculadora con inicialización confiable ---
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
        """Ejecuta todos los cálculos básicos en el orden correcto."""
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
            self.min_bankroll_formula = "max(100 * Apuesta Mín, 5% * Ganancia Promedio)"
            self.min_bankroll_calculation, min_bankroll = f"max(\${part1:.2f}, \${part2:.2f})", max(part1, part2)
        elif self.volatility == 'medium':
            part1, part2 = 75 * self.min_bet, 0.03 * self.avg_win
            self.min_bankroll_formula = "max(75 * Apuesta Mín, 3% * Ganancia Promedio)"
            self.min_bankroll_calculation, min_bankroll = f"max(\${part1:.2f}, \${part2:.2f})", max(part1, part2)
        else:  # low
            part1, part2 = 50 * self.min_bet, 0.01 * self.avg_win
            self.min_bankroll_formula = "max(50 * Apuesta Mín, 1% * Ganancia Promedio)"
            self.min_bankroll_calculation, min_bankroll = f"max(\${part1:.2f}, \${part2:.2f})", max(part1, part2)
        return round(min_bankroll, 2)

    def generate_bankroll_strategy(self, personal_bankroll, risk_level='medium'):
        min_bankroll = self.calculate_min_bankroll()
        min_bank_advice = []
        if personal_bankroll < min_bankroll:
            pb_formatted = f"{personal_bankroll:,.2f}"
            mb_formatted = f"{min_bankroll:,.2f}"
            min_bank_advice.append(f"🚨 **RIESGO CRÍTICO**: ¡Tu bankroll (\${pb_formatted}) está **SIGNIFICATIVAMENTE POR DEBAJO** del mínimo (\${mb_formatted})!")
            min_bank_advice.append("La probabilidad de perder todo el bankroll antes de una ganancia significativa **supera el 95%**. **NO RECOMENDAMOS** jugar con este bankroll.")
        else:
            pb_formatted = f"{personal_bankroll:,.2f}"
            mb_formatted = f"{min_bankroll:,.2f}"
            min_bank_advice.append(f"✅ Tu bankroll (\${pb_formatted}) es suficiente para esta tragamonedas (mínimo: \${mb_formatted}).")
        
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
                adjustment_note = f" (Nota: la apuesta teórica \${tb_formatted} fue **ajustada** al mínimo posible en esta tragamonedas)."
            elif bet_per_spin < theoretical_bet:
                 adjustment_note = f" (Nota: la apuesta teórica \${tb_formatted} fue **reducida y redondeada** según el paso de apuesta)."
        
        base_win_prob, rtp = float(self.config.get('probabilities', {}).get('base_win_probability', 0.25)), self.config.get('game_config', {}).get('rtp', 0.96)
        bwp_pct = base_win_prob * 100
        losing_spins_count = 10 - int(base_win_prob * 10)
        rtp_pct = rtp * 100
        house_edge_val = 1000 * (1 - rtp)
        hev_formatted = f"{house_edge_val:.2f}"
        
        truth1 = f"Probabilidad de cualquier ganancia por giro: **{bwp_pct:.1f}%**. Esto significa que en promedio **~{losing_spins_count} de cada 10 giros serán perdedores**."
        truth2 = f"**RTP {rtp_pct:.1f}%** significa que por cada \$1,000 apostado, el casino se queda en promedio con **\${hev_formatted}**."

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
        
        strategy1 = f"**Apuesta recomendada**: Para tu bankroll y nivel de riesgo, la apuesta real es **\${bps_formatted}**.{adjustment_note}"
        strategy2 = f"**Gestión de apuestas**: Comienza con la apuesta mínima **\${mbet_formatted}**. Si el juego va bien, aumenta gradualmente la apuesta pero no superes la recomendada."
        strategy3 = f"**Stop-loss (regla de hierro)**: Deja de jugar inmediatamente si tu bankroll cae a **\${sll_val_f}** (pérdida de \${sll_loss_f})."
        strategy4 = f"**Objetivo de ganancia**: Asegura las ganancias y deja de jugar si tu bankroll alcanza **\${wgl_val_f}** (ganancia de \${wgl_profit_f})."
        strategy5 = "**Psicología**: **NUNCA** intentes 'recuperar pérdidas'. Cada giro es independiente."

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
        ax.set_title(f'Probabilidad de combinación ganadora con símbolo (Nivel: {level})', fontsize=16, pad=20)
        ax.set_xlabel('Probabilidad por giro (considerando Wild), %', fontsize=12); ax.set_ylabel('Símbolo', fontsize=12)
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
            'Probabilidad Pura (%)': df_sorted['pure_probability'].apply(lambda x: f"{x*100:.2f}%"),
            'Probabilidad Combinación (con Wild, %)': df_sorted['combo_probability'].apply(lambda x: f"{x*100:.2f}%"),
            'Giros para 99% de probabilidad': df_sorted['spins_for_99_prob'].apply(lambda x: f"{int(x)}" if x != float('inf') else "∞")
        })
        return df_display

# --- Bloque principal de la aplicación web Streamlit ---
def main():
    st.set_page_config(page_title="Analizador de Tragamonedas", layout="wide", initial_sidebar_state="expanded")
    
    # --- Obtener lista de archivos locales ---
    local_config_files = get_local_config_files(CONFIGS_FOLDER)
    
    with st.sidebar:
        st.title("🎰 Parámetros de Análisis")
        
        # --- Nuevo bloque de selección de fuente de archivo ---
        file_source = st.radio(
            "Selecciona la fuente de configuración:",
            ('Cargar archivo desde computadora', 'Seleccionar de predefinidos'),
            index=0  # Por defecto "Cargar archivo"
        )
        
        config_file = None
        
        if file_source == 'Cargar archivo desde computadora':
            config_file = st.file_uploader("1a. Carga configuración JSON de tragamonedas", type="json")
        elif file_source == 'Seleccionar de predefinidos' and local_config_files:
            selected_filename = st.selectbox(
                "1b. Selecciona configuración de tragamonedas",
                options=local_config_files,
                format_func=lambda x: x  # Mostrar nombre de archivo tal cual
            )
            if selected_filename:
                # Intentar abrir archivo desde carpeta local
                try:
                    full_path = os.path.join(CONFIGS_FOLDER, selected_filename)
                    # Abrir archivo en modo binario y crear objeto BytesIO,
                    # que imita archivo cargado para st.file_uploader
                    with open(full_path, 'rb') as f:
                        config_file = f.read()
                    # st.file_uploader espera objeto con atributo 'name'
                    # Envolver bytes en objeto compatible con UploaderFile
                    from io import BytesIO
                    config_file = BytesIO(config_file)
                    config_file.name = selected_filename  # Agregar nombre de archivo
                except Exception as e:
                     st.error(f"Error cargando archivo {selected_filename}: {e}")
                     config_file = None
        elif file_source == 'Seleccionar de predefinidos' and not local_config_files:
             st.info(f"Carpeta '{CONFIGS_FOLDER}' no encontrada o vacía.")
        
        personal_bankroll, desired_win, risk_level, analyze_button = 0, 0, 'medium', False
        if config_file is not None:
            personal_bankroll = st.number_input("2. Tu bankroll inicial ($)", min_value=0.01, value=200.0, step=10.0, format="%.2f")
            desired_win = st.number_input("3. Tu ganancia neta deseada ($)", min_value=1.0, value=500.0, step=10.0, format="%.2f")
            risk_level = st.selectbox("4. Tu nivel de riesgo", options=['low', 'medium', 'high'], index=1).lower()
            analyze_button = st.button("🚀 Ejecutar análisis completo", type="primary", use_container_width=True)
    
    st.title("Analizador Universal de Probabilidades de Tragamonedas")
    st.markdown("Esta herramienta te ayuda a entender las probabilidades reales y desarrollar una estrategia para cualquier tragamonedas basada en sus parámetros matemáticos.")
    
    if analyze_button and config_file is not None:
        try:
            # Para BytesIO necesita reposicionar el puntero al inicio
            if hasattr(config_file, 'seek'):
                config_file.seek(0)
            config = json.load(config_file)
            calculator = SlotProbabilityCalculator(config)
            if personal_bankroll < calculator.min_bet:
                pb_formatted_error = f"{personal_bankroll:.2f}"
                mb_formatted_error = f"{calculator.min_bet:.2f}"
                st.error(f"**Tu bankroll (\${pb_formatted_error}) es menor que la apuesta mínima en esta tragamonedas (\${mb_formatted_error}).**")
                st.warning("Desafortunadamente, el análisis es imposible. Por favor, aumenta tu bankroll.")
                st.stop()
            game_config = config.get('game_config', {})
            
            gn_formatted = game_config.get('game_name', 'N/A')
            pb_formatted = f"{personal_bankroll:,.2f}"
            dw_formatted = f"{desired_win:,.2f}"
            rl_formatted = risk_level.capitalize()

            st.header(f"🎰 Análisis Completo de Tragamonedas: {gn_formatted}", divider="rainbow")
            st.markdown(f"### Tus Parámetros: Bankroll: \${pb_formatted} | Ganancia Deseada: +\${dw_formatted} | Riesgo: **{rl_formatted}**")
            
            goal_result = calculator.estimate_goal_chance(personal_bankroll, desired_win)
            strategy = calculator.generate_bankroll_strategy(personal_bankroll, risk_level)
            bet_per_spin = strategy.get('bet_per_spin')
            guaranteed_spins = int(personal_bankroll / bet_per_spin) if bet_per_spin > 0 else float('inf')
            
            st.subheader("🎯 Análisis de Tu Objetivo", divider="blue")
            col1, col2 = st.columns(2)
            with col1:
                dw_label_formatted = f"{desired_win:,.2f}"
                st.metric(label=f"Probabilidad estimada de ganar \${dw_label_formatted}", value=f"{goal_result['probability']*100:.4f}%")
            with col2:
                spins_str = f"{guaranteed_spins}" if guaranteed_spins != float('inf') else "∞"
                st.metric(label="Número garantizado de giros (con apuesta rec.)", value=spins_str)
            
            with st.expander("¿Cómo entender estos números? 🤔"):
                st.markdown(f"""
                #### Probabilidad de ganancia
                Esta es tu probabilidad matemática de alcanzar el objetivo **antes de que la ventaja del casino (RTP < 100%) agote tu bankroll**.
                #### Número garantizado de giros
                Este es el **número real de giros** que puedes hacer con tu bankroll jugando con la **Apuesta Recomendada** (\${bet_per_spin:.2f}).
                - **¿Cómo se determina la apuesta?** Multiplicamos la apuesta mínima de la tragamonedas (**\${calculator.min_bet:.2f}**) por el coeficiente de riesgo (x1-x5) y por un coeficiente no lineal de tu bankroll. Luego el resultado es **redondeado y ajustado** para que se ajuste a los límites reales de la tragamonedas.
                - **Este es tu 'margen de seguridad' real**: Cuanto más grande sea, más tiempo de juego tendrás para alcanzar el objetivo.
                """)
            
            st.subheader("📊 Análisis Visual de Probabilidades", divider="blue")
            fig = calculator.visualize_win_probabilities()
            if fig: st.pyplot(fig)
            st.dataframe(calculator.get_results_table(), use_container_width=True)
            
            st.header("♟️ Estrategia Personal de Juego", divider="rainbow")
            with st.container(border=True):
                st.subheader("1. Veredicto sobre Tu Bankroll")
                for advice in strategy['min_bank_advice']: 
                    st.markdown(f"➡️ {advice}")
            with st.container(border=True):
                st.subheader("2. Fundamentación y Cálculo del Bankroll Mínimo")
                st.markdown("Para que la estrategia tenga sentido, tu bankroll debe poder soportar series de pérdidas características de esta volatilidad.")
                st.markdown("\n**Datos de origen para el cálculo:**")
                st.markdown(f" • **Apuesta mínima**: \${calculator.min_bet:.2f}")
                st.markdown(f" • **Ganancia máxima con apuesta mínima**: \${calculator.max_win_at_min_bet:,.2f}")
                st.markdown(f" • **Ganancia promedio significativa (con apuesta mínima)**: \${calculator.avg_win:,.2f}")
                st.markdown(f" • **Volatilidad**: {calculator.volatility.capitalize()}")
                st.markdown("\n**Proceso de cálculo:**")
                st.markdown(f"1. **Fórmula** (para volatilidad {calculator.volatility.capitalize()}): `{calculator.min_bankroll_formula}`")
                st.markdown(f"2. **Sustituye valores**: `{calculator.min_bankroll_calculation}`")
                min_bankroll_final_str = ''.join(filter(lambda char: char.isdigit() or char in '.,', strategy['min_bank_advice'][0].split('$')[-1]))
                st.success(f"**Resultado**: El mínimo recomendado final es **\${min_bankroll_final_str}**")
            with st.container(border=True):
                st.subheader("3. Verdad Dura sobre las Probabilidades (sin adornos)")
                for truth in strategy['harsh_truths']: 
                    st.markdown(f"➡️ {truth}")
            with st.container(border=True):
                st.subheader("4. Estrategia Óptima Paso a Paso")
                for i, step in enumerate(strategy['optimal_strategy'], 1): 
                    st.markdown(f"**Paso {i}**: {step}")
                    
        except json.JSONDecodeError:
            st.error("Error: El archivo seleccionado no es un JSON válido.")
        except Exception as e:
            st.error(f"Ocurrió un error al analizar el archivo. Asegúrate de que el archivo JSON tenga la estructura correcta. Error: {e}")
    elif analyze_button and config_file is None:
        st.warning("Por favor, carga un archivo de configuración JSON de tragamonedas o selecciona de la lista para comenzar el análisis.")

if __name__ == "__main__":
    main()
