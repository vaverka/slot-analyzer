# ==============================================================================
#  app.py - UNIVERSAL SLOT ANALYZER V8.1 (with batch analysis bugfix)
# ==============================================================================
import json
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os  # For working with file system
from io import BytesIO # To handle file streams

# --- Configuration ---
CONFIGS_FOLDER = "."  # Folder with preset configs

# --- Helper function to get file list from folder in repo ---
@st.cache_data
def get_local_config_files(folder_path):
    """
    Gets a list of JSON files from the specified local folder.
    """
    try:
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
            return sorted(files)
        else:
            return []
    except Exception as e:
        print(f"Error getting file list from {folder_path}: {e}")
        return []

# --- Calculator class with reliable initialization ---
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
        """Runs all basic calculations in correct order."""
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

    def get_spins_for_99_range(self, level='base'):
        """Calculates the min-max range of spins for a 99% probability of a win."""
        if not self.win_probabilities: return "N/A"
        level_data = self.win_probabilities.get(level)
        if not level_data: return "N/A"
        
        spins_values = [
            data['spins_for_99_prob'] 
            for data in level_data.values() 
            if data['spins_for_99_prob'] != float('inf')
        ]
        
        if not spins_values:
            return "N/A"
            
        min_spins = min(spins_values)
        max_spins = max(spins_values)
        
        return f"{int(min_spins)} - {int(max_spins)}"

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
            self.min_bankroll_formula = "max(100 * Min. Bet, 5% * Average Win)"
            self.min_bankroll_calculation, min_bankroll = f"max(${part1:.2f}, ${part2:.2f})", max(part1, part2)
        elif self.volatility == 'medium':
            part1, part2 = 75 * self.min_bet, 0.03 * self.avg_win
            self.min_bankroll_formula = "max(75 * Min. Bet, 3% * Average Win)"
            self.min_bankroll_calculation, min_bankroll = f"max(${part1:.2f}, ${part2:.2f})", max(part1, part2)
        else:  # low
            part1, part2 = 50 * self.min_bet, 0.01 * self.avg_win
            self.min_bankroll_formula = "max(50 * Min. Bet, 1% * Average Win)"
            self.min_bankroll_calculation, min_bankroll = f"max(${part1:.2f}, ${part2:.2f})", max(part1, part2)
        return round(min_bankroll, 2)

    def generate_bankroll_strategy(self, personal_bankroll, risk_level='medium'):
        min_bankroll = self.calculate_min_bankroll()
        min_bank_advice = []
        if personal_bankroll < min_bankroll:
            pb_formatted = f"{personal_bankroll:,.2f}"
            mb_formatted = f"{min_bankroll:,.2f}"
            min_bank_advice.append(f"🚨 **КРИТИЧЕСКИЙ РИСК**: Ваш банкролл (${pb_formatted}) **ЗНАЧИТЕЛЬНО НИЖЕ** минимального (${mb_formatted})!")
            min_bank_advice.append("Вероятность проигрыша всего банкролла до значительного выигрыша **превышает 95%**. Мы **НЕ РЕКОМЕНДУЕМ** играть с таким банкроллом.")
        else:
            pb_formatted = f"{personal_bankroll:,.2f}"
            mb_formatted = f"{min_bankroll:,.2f}"
            min_bank_advice.append(f"✅ Ваш банкролл (${pb_formatted}) достаточен для этого слота (минимум: ${mb_formatted}).")
        
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
                adjustment_note = f" (Примечание: теоретическая ставка ${tb_formatted} была **скорректирована** до минимума)."
            elif bet_per_spin < theoretical_bet:
                 adjustment_note = f" (Примечание: теоретическая ставка ${tb_formatted} была **уменьшена и округлена**)."
        
        base_win_prob, rtp = float(self.config.get('probabilities', {}).get('base_win_probability', 0.25)), self.config.get('game_config', {}).get('rtp', 0.96)
        bwp_pct = base_win_prob * 100
        losing_spins_count = 10 - int(base_win_prob * 10)
        rtp_pct = rtp * 100
        house_edge_val = 1000 * (1 - rtp)
        hev_formatted = f"{house_edge_val:.2f}"
        
        truth1 = f"Вероятность любого выигрыша за спин: **{bwp_pct:.1f}%**. Это означает, что в среднем **~{losing_spins_count} из 10 спинов будут проигрышными**."
        truth2 = f"**RTP {rtp_pct:.1f}%** означает, что с каждой поставленной $1,000 казино в среднем забирает **${hev_formatted}**."

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
        
        strategy1 = f"**Рекомендуемая ставка**: Для вашего банкролла и уровня риска реальная ставка составляет **${bps_formatted}**.{adjustment_note}"
        strategy2 = f"**Управление ставками**: Начните с минимальной ставки **${mbet_formatted}**. Если игра идет хорошо, постепенно увеличивайте, но не превышайте рекомендуемую."
        strategy3 = f"**Стоп-лосс (железное правило)**: Немедленно прекратите игру, если ваш банкролл упадет до **${sll_val_f}** (потеря ${sll_loss_f})."
        strategy4 = f"**Цель по выигрышу**: Зафиксируйте прибыль и прекратите игру, если ваш банкролл достигнет **${wgl_val_f}** (прибыль ${wgl_profit_f})."
        strategy5 = "**Психология**: **НИКОГДА** не пытайтесь 'отыграться'. Каждый спин независим."

        optimal_strategy = [strategy1, strategy2, strategy3, strategy4, strategy5]
        
        return {'min_bank_advice': min_bank_advice, 'harsh_truths': harsh_truths, 'optimal_strategy': optimal_strategy, 'bet_per_spin': bet_per_spin}

    def estimate_goal_chance(self, personal_bankroll, desired_win):
        rtp = self.config.get('game_config', {}).get('rtp', 0.96)
        if desired_win <= 0: return {"probability": 1.0}
        if personal_bankroll <= 0: return {"probability": 0.0}
        target_amount = personal_bankroll + desired_win
        effective_bankroll = personal_bankroll * rtp
        probability = effective_bankroll / target_amount
        return {"probability": min(1.0, max(0.0, probability))} # Ensure probability is between 0 and 1

    def visualize_win_probabilities(self, level='base'):
        if not self.win_probabilities: return None
        level_data = self.win_probabilities.get(level)
        if not level_data: return None
        df = pd.DataFrame.from_dict(level_data, orient='index').sort_values('combo_probability', ascending=False)
        if df.empty: return None
        df['combo_probability_pct'] = df['combo_probability'] * 100
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x='combo_probability_pct', y='name', data=df, palette='viridis_r', orient='h', hue='name', legend=False, ax=ax)
        ax.set_title(f'Probability of winning combination with symbol (Level: {level})', fontsize=16, pad=20)
        ax.set_xlabel('Probability per spin (with Wild), %', fontsize=12); ax.set_ylabel('Symbol', fontsize=12)
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
            'Pure Probability (%)': df_sorted['pure_probability'].apply(lambda x: f"{x*100:.2f}%"),
            'Combination Probability (with Wild, %)': df_sorted['combo_probability'].apply(lambda x: f"{x*100:.2f}%"),
            'Spins for 99% chance': df_sorted['spins_for_99_prob'].apply(lambda x: f"{int(x)}" if x != float('inf') else "∞")
        })
        return df_display

# --- Main Streamlit web application block ---
def main():
    st.set_page_config(page_title="Slot Analyzer", layout="wide", initial_sidebar_state="expanded")
    
    st.title("Universal Slot Probability Analyzer")
    st.markdown("Этот инструмент помогает понять реальные шансы и разработать стратегию для любого слота на основе его математических параметров.")
    
    analysis_mode = st.radio(
        "Выберите режим анализа:",
        ("Анализ одного слота", "Анализ всех слотов в папке"),
        horizontal=True,
    )
    
    local_config_files = get_local_config_files(CONFIGS_FOLDER)
    
    if analysis_mode == "Анализ одного слота":
        run_single_slot_analysis(local_config_files)
    else:
        run_batch_analysis(local_config_files)

def run_single_slot_analysis(local_config_files):
    with st.sidebar:
        st.title("🎰 Параметры анализа")
        
        file_source = st.radio(
            "Выберите источник конфигурации:",
            ('Загрузить файл с компьютера', 'Выбрать из предустановок'),
            index=0
        )
        
        config_file = None
        
        if file_source == 'Загрузить файл с компьютера':
            config_file = st.file_uploader("1a. Загрузите JSON конфигурацию слота", type="json")
        elif file_source == 'Выбрать из предустановок' and local_config_files:
            selected_filename = st.selectbox(
                "1b. Выберите конфигурацию слота",
                options=local_config_files,
                format_func=lambda x: x
            )
            if selected_filename:
                try:
                    full_path = os.path.join(CONFIGS_FOLDER, selected_filename)
                    with open(full_path, 'rb') as f:
                        config_bytes = f.read()
                    config_file = BytesIO(config_bytes)
                    config_file.name = selected_filename
                except Exception as e:
                     st.error(f"Ошибка загрузки файла {selected_filename}: {e}")
                     config_file = None
        elif file_source == 'Выбрать из предустановок' and not local_config_files:
             st.info(f"Папка '{CONFIGS_FOLDER}' не найдена или пуста.")
        
        personal_bankroll, desired_win, risk_level, analyze_button = 0, 0, 'medium', False
        if config_file is not None:
            personal_bankroll = st.number_input("2. Ваш стартовый банкролл ($)", min_value=0.01, value=200.0, step=10.0, format="%.2f")
            desired_win = st.number_input("3. Ваш желаемый чистый выигрыш ($)", min_value=1.0, value=500.0, step=10.0, format="%.2f")
            risk_level = st.selectbox("4. Ваш уровень риска", options=['low', 'medium', 'high'], index=1).lower()
            analyze_button = st.button("🚀 Запустить полный анализ", type="primary", use_container_width=True)
    
    if analyze_button and config_file is not None:
        try:
            if hasattr(config_file, 'seek'):
                config_file.seek(0)
            config = json.load(config_file)
            calculator = SlotProbabilityCalculator(config)
            if personal_bankroll < calculator.min_bet:
                pb_formatted_error = f"{personal_bankroll:.2f}"
                mb_formatted_error = f"{calculator.min_bet:.2f}"
                st.error(f"**Ваш банкролл (${pb_formatted_error}) меньше минимальной ставки в этом слоте (${mb_formatted_error}).**")
                st.warning("К сожалению, анализ невозможен. Пожалуйста, увеличьте свой банкролл.")
                st.stop()
            game_config = config.get('game_config', {})
            
            gn_formatted = game_config.get('game_name', 'N/A')
            pb_formatted = f"{personal_bankroll:,.2f}"
            dw_formatted = f"{desired_win:,.2f}"
            rl_formatted = risk_level.capitalize()

            st.header(f"🎰 Полный анализ слота: {gn_formatted}", divider="rainbow")
            st.markdown(f"### Ваши параметры: Банкролл: ${pb_formatted} | Желаемый выигрыш: +${dw_formatted} | Риск: **{rl_formatted}**")
            
            goal_result = calculator.estimate_goal_chance(personal_bankroll, desired_win)
            strategy = calculator.generate_bankroll_strategy(personal_bankroll, risk_level)
            bet_per_spin = strategy.get('bet_per_spin')
            guaranteed_spins = int(personal_bankroll / bet_per_spin) if bet_per_spin > 0 else float('inf')
            
            st.subheader("🎯 Анализ вашей цели", divider="blue")
            col1, col2 = st.columns(2)
            with col1:
                dw_label_formatted = f"{desired_win:,.2f}"
                st.metric(label=f"Оценочный шанс выиграть ${dw_label_formatted}", value=f"{goal_result['probability']*100:.4f}%")
            with col2:
                spins_str = f"{guaranteed_spins}" if guaranteed_spins != float('inf') else "∞"
                st.metric(label="Гарантированное количество спинов (при рек. ставке)", value=spins_str)
            
            with st.expander("Как понимать эти цифры? 🤔"):
                st.markdown(f"""
                #### Шанс на выигрыш
                Это ваша математическая вероятность достичь цели **до того, как преимущество казино (RTP < 100%) исчерпает ваш банкролл**.
                #### Гарантированное количество спинов
                Это **реальное количество вращений**, которое вы можете сделать с вашим банкроллом, играя по **Рекомендуемой ставке** (${bet_per_spin:.2f}).
                - **Как определяется ставка?** Мы умножаем минимальную ставку слота (**${calculator.min_bet:.2f}**) на коэффициент риска (x1-x5) и на нелинейный коэффициент банкролла. Затем результат **округляется и корректируется** под реальные лимиты слота.
                - **Это ваш реальный 'запас прочности'**: Чем он больше, тем дольше ваше игровое время для достижения цели.
                """)
            
            st.subheader("📊 Визуальный анализ вероятностей", divider="blue")
            fig = calculator.visualize_win_probabilities()
            if fig: st.pyplot(fig)
            st.dataframe(calculator.get_results_table(), use_container_width=True)
            
            st.header("♟️ Персональная стратегия игры", divider="rainbow")
            with st.container(border=True):
                st.subheader("1. Вердикт по вашему банкроллу")
                for advice in strategy['min_bank_advice']: 
                    st.markdown(f"➡️ {advice}")
            with st.container(border=True):
                st.subheader("2. Расчет и обоснование минимального банкролла")
                st.markdown("Чтобы стратегия имела смысл, ваш банкролл должен выдерживать серии проигрышей, характерные для данной волатильности.")
                st.markdown("\n**Исходные данные для расчета:**")
                st.markdown(f" • **Минимальная ставка**: ${calculator.min_bet:.2f}")
                st.markdown(f" • **Макс. выигрыш при мин. ставке**: ${calculator.max_win_at_min_bet:,.2f}")
                st.markdown(f" • **Средний значимый выигрыш (при мин. ставке)**: ${calculator.avg_win:,.2f}")
                st.markdown(f" • **Волатильность**: {calculator.volatility.capitalize()}")
                st.markdown("\n**Процесс расчета:**")
                st.markdown(f"1. **Формула** (для {calculator.volatility.capitalize()} волатильности): `{calculator.min_bankroll_formula}`")
                st.markdown(f"2. **Подстановка значений**: `{calculator.min_bankroll_calculation}`")
                min_bankroll_final_str = ''.join(filter(lambda char: char.isdigit() or char in '.,', strategy['min_bank_advice'][0].split('$')[-1]))
                st.success(f"**Результат**: Итоговый рекомендуемый минимум: **${min_bankroll_final_str}**")
            with st.container(border=True):
                st.subheader("3. Суровая правда о шансах (без прикрас)")
                for truth in strategy['harsh_truths']: 
                    st.markdown(f"➡️ {truth}")
            with st.container(border=True):
                st.subheader("4. Оптимальная пошаговая стратегия")
                for i, step in enumerate(strategy['optimal_strategy'], 1): 
                    st.markdown(f"**Шаг {i}**: {step}")
                    
        except json.JSONDecodeError:
            st.error("Ошибка: Выбранный файл не является валидным JSON.")
        except Exception as e:
            st.error(f"Ошибка при анализе файла. Убедитесь, что JSON имеет правильную структуру. Ошибка: {e}")
    elif analyze_button and config_file is None:
        st.warning("Пожалуйста, загрузите JSON-файл конфигурации слота или выберите из списка, чтобы начать анализ.")


def run_batch_analysis(local_config_files):
    st.header("Сравнительный анализ всех слотов", divider="rainbow")
    
    with st.sidebar:
        st.title("🎰 Параметры для всех слотов")
        personal_bankroll = st.number_input("1. Ваш стартовый банкролл ($)", min_value=0.01, value=200.0, step=10.0, format="%.2f")
        desired_win = st.number_input("2. Ваш желаемый чистый выигрыш ($)", min_value=1.0, value=500.0, step=10.0, format="%.2f")
        risk_level = st.selectbox("3. Ваш уровень риска", options=['low', 'medium', 'high'], index=1).lower()
        analyze_button = st.button("🚀 Запустить пакетный анализ", type="primary", use_container_width=True)
    
    if not local_config_files:
        st.warning(f"В папке '{CONFIGS_FOLDER}' не найдено файлов конфигурации (.json).")
        return

    if analyze_button:
        with st.spinner(f"Анализирую {len(local_config_files)} слотов..."):
            all_results = []
            for filename in local_config_files:
                try:
                    full_path = os.path.join(CONFIGS_FOLDER, filename)
                    with open(full_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    
                    calculator = SlotProbabilityCalculator(config)
                    
                    # --- Calculations ---
                    game_name = config.get('game_config', {}).get('game_name', filename)
                    goal_chance = calculator.estimate_goal_chance(personal_bankroll, desired_win)['probability']
                    strategy = calculator.generate_bankroll_strategy(personal_bankroll, risk_level)
                    bet_per_spin = strategy.get('bet_per_spin', calculator.min_bet)
                    guaranteed_spins = int(personal_bankroll / bet_per_spin) if bet_per_spin > 0 else 0
                    spins_99_range = calculator.get_spins_for_99_range()
                    
                    # === BUG FIX AREA: More robust way to determine the verdict ===
                    full_verdict_message = strategy['min_bank_advice'][0]
                    if "КРИТИЧЕСКИЙ РИСК" in full_verdict_message:
                        bankroll_verdict = "Критический риск"
                    elif "достаточен" in full_verdict_message:
                        bankroll_verdict = "Достаточный"
                    else:
                        bankroll_verdict = "N/A" # Fallback
                    # === END OF BUG FIX ===
                        
                    any_win_prob = config.get('probabilities', {}).get('base_win_probability', 0)
                    min_bet = calculator.min_bet
                    
                    all_results.append({
                        "Название слота": game_name,
                        "Estimated chance to win": f"{goal_chance * 100:.4f}%",
                        "Guaranteed number of spins": guaranteed_spins,
                        "spins for 99% probability (min - max)": spins_99_range,
                        "Bankroll Verdict": bankroll_verdict,
                        "Probability of any win per spin": f"{any_win_prob * 100:.1f}%",
                        "Minimum bet": f"${min_bet:.2f}"
                    })

                except Exception as e:
                    st.error(f"Не удалось проанализировать файл {filename}: {e}")
            
            if all_results:
                df = pd.DataFrame(all_results)
                st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()
