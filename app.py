# ==============================================================================
#  app.py - УНИВЕРСАЛЬНЫЙ АНАЛИЗАТОР СЛОТОВ V4.8 (с финальным объяснением)
# ==============================================================================
import json
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# --- Класс-калькулятор с надежной инициализацией ---
sns.set_theme(style="whitegrid", palette="viridis")

class SlotProbabilityCalculator:
    def __init__(self, config):
        self.config = config
        self.win_probabilities = None
        self.min_bet, self.max_bet, self.max_win, self.avg_win = None, None, None, None
        self.min_bankroll_formula, self.min_bankroll_calculation = "", ""
        self.volatility = self.config.get('game_config', {}).get('volatility', 'medium')
        # Все базовые расчеты теперь происходят при создании объекта
        self.calculate_all()

    def calculate_all(self):
        """Запускает все базовые расчеты в правильном порядке."""
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
        self.max_win, self.avg_win = max_win_multiplier * self.max_bet, 0.4 * (max_win_multiplier * self.max_bet)
        min_bankroll = 0
        if self.volatility == 'high':
            part1, part2 = 100 * self.min_bet, 0.05 * self.avg_win
            self.min_bankroll_formula, self.min_bankroll_calculation, min_bankroll = "max(100 * Мин. ставка, 5% * Среднего выигрыша)", f"max(${part1:.2f}, ${part2:.2f})", max(part1, part2)
        elif self.volatility == 'medium':
            part1, part2 = 75 * self.min_bet, 0.03 * self.avg_win
            self.min_bankroll_formula, self.min_bankroll_calculation, min_bankroll = "max(75 * Мин. ставка, 3% * Среднего выигрыша)", f"max(${part1:.2f}, ${part2:.2f})", max(part1, part2)
        else:
            part1, part2 = 50 * self.min_bet, 0.01 * self.avg_win
            self.min_bankroll_formula, self.min_bankroll_calculation, min_bankroll = "max(50 * Мин. ставка, 1% * Среднего выигрыша)", f"max(${part1:.2f}, ${part2:.2f})", max(part1, part2)
        return round(min_bankroll, 2)

    def generate_bankroll_strategy(self, personal_bankroll, risk_level='medium'):
        min_bankroll = self.calculate_min_bankroll()
        min_bank_advice = []
        if personal_bankroll < min_bankroll:
            min_bank_advice.append(f"🚨 **КРИТИЧЕСКИЙ РИСК**: Ваш банкролл (${personal_bankroll:,.2f}) **ЗНАЧИТЕЛЬНО НИЖЕ** минимального (${min_bankroll:,.2f})!")
            min_bank_advice.append("Вероятность проигрыша всего банка до получения значимого выигрыша **превышает 95%**. Мы **НЕ РЕКОМЕНДУЕМ** играть с таким банком.")
        else:
            min_bank_advice.append(f"✅ Ваш банкролл (${personal_bankroll:,.2f}) достаточен для игры в этот слот (минимум: ${min_bankroll:,.2f}).")
        risk_profiles = {'low': {'bet_percent': 0.5, 'stop_loss': 0.25, 'win_goal': 0.4},'medium': {'bet_percent': 1.0, 'stop_loss': 0.4, 'win_goal': 0.6},'high': {'bet_percent': 2.0, 'stop_loss': 0.5, 'win_goal': 1.0}}
        profile = risk_profiles.get(risk_level, risk_profiles['medium'])
        theoretical_bet = personal_bankroll * profile['bet_percent'] / 100
        bet_per_spin = min(theoretical_bet, self.max_bet)
        base_win_prob, rtp = float(self.config.get('probabilities', {}).get('base_win_probability', 0.25)), self.config.get('game_config', {}).get('rtp', 0.96)
        harsh_truths = [f"Вероятность любого выигрыша за спин: **{base_win_prob*100:.1f}%**. Это означает, что в среднем **~{10 - int(base_win_prob * 10)} из 10 спинов будут проигрышными**.", f"**RTP {rtp*100:.1f}%** означает, что на каждый поставленный $1,000, казино в среднем оставляет себе **${1000 * (1 - rtp):.2f}**."]
        optimal_strategy = [f"**Ставка**: Ваша рекомендуемая ставка **${bet_per_spin:.2f}**. Начинайте с минимальной ставки **${self.min_bet:.2f}**.", f"**Управление ставками**: Увеличивайте ставку до **${bet_per_spin:.2f}** ТОЛЬКО после активации бонусной функции или крупного выигрыша.", f"**Стоп-лосс (железное правило)**: Немедленно прекратите игру, если ваш банк опустится до **${personal_bankroll * (1-profile['stop_loss']):.2f}** (потеря ${personal_bankroll * profile['stop_loss']:.2f}).", f"**Цель выигрыша**: Зафиксируйте прибыль и прекратите игру, если ваш банк достигнет **${personal_bankroll * (1+profile['win_goal']):.2f}** (прибыль ${personal_bankroll * profile['win_goal']:.2f}).", "**Психология**: **НИКОГДА** не пытайтесь 'отыграться'. Каждый спин независим. После крупного выигрыша или проигрыша сделайте перерыв."]
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
        ax.set_title(f'Вероятность выигрышной комбинации с символом (Уровень: {level})', fontsize=16, pad=20)
        ax.set_xlabel('Вероятность за один спин (с учетом Wild), %', fontsize=12); ax.set_ylabel('Символ', fontsize=12)
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
            'Символ': df_sorted['name'],
            'Чистая вероятность (%)': df_sorted['pure_probability'].apply(lambda x: f"{x*100:.2f}%"),
            'Вероятность комбинации (с Wild, %)': df_sorted['combo_probability'].apply(lambda x: f"{x*100:.2f}%"),
            'Спинов для 99% шанса': df_sorted['spins_for_99_prob'].apply(lambda x: f"{int(x)}" if x != float('inf') else "∞")
        })
        return df_display

# --- Основной блок веб-приложения Streamlit ---
def main():
    st.set_page_config(page_title="Анализатор слотов", layout="wide", initial_sidebar_state="expanded")

    with st.sidebar:
        st.title("🎰 Параметры Анализа")
        uploaded_file = st.file_uploader("1. Загрузите JSON-конфигурацию слота", type="json")
        personal_bankroll, desired_win, risk_level, analyze_button = 0, 0, 'medium', False
        if uploaded_file:
            personal_bankroll = st.number_input("2. Ваш стартовый банкролл ($)", min_value=1.0, value=200.0, step=10.0, format="%.2f")
            desired_win = st.number_input("3. Ваш желаемый чистый выигрыш ($)", min_value=1.0, value=500.0, step=10.0, format="%.2f")
            risk_level = st.selectbox("4. Ваш уровень риска", options=['low', 'medium', 'high'], index=1).lower()
            analyze_button = st.button("🚀 Провести полный анализ", type="primary", use_container_width=True)

    st.title("Универсальный анализатор вероятностей слотов")
    st.markdown("Этот инструмент поможет вам понять реальные шансы и разработать стратегию для любого слота, основываясь на его математических параметрах.")

    if analyze_button and uploaded_file:
        try:
            uploaded_file.seek(0)
            config = json.load(uploaded_file)
            calculator = SlotProbabilityCalculator(config)
            game_config = config.get('game_config', {})
            
            st.header(f"🎰 Полный анализ слота: {game_config.get('game_name', 'N/A')}", divider="rainbow")
            st.subheader(f"Ваши параметры: Банкролл: **${personal_bankroll:,.2f}** | Желаемый выигрыш: **+${desired_win:,.2f}** | Риск: **{risk_level.capitalize()}**")

            goal_result = calculator.estimate_goal_chance(personal_bankroll, desired_win)
            rtp = game_config.get('rtp', 0.96)
            strategy = calculator.generate_bankroll_strategy(personal_bankroll, risk_level)
            bet_per_spin = strategy.get('bet_per_spin', calculator.min_bet)
            net_loss_per_spin = bet_per_spin * (1 - rtp) if bet_per_spin > 0 else 0
            expected_spins = int(personal_bankroll / net_loss_per_spin) if net_loss_per_spin > 0 else float('inf')

            st.subheader("🎯 Анализ вашей цели", divider="blue")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label=f"Оценочный шанс выиграть ${desired_win:,.2f}", value=f"{goal_result['probability']*100:.4f}%")
            with col2:
                spins_str = f"~{expected_spins}" if expected_spins != float('inf') else "∞"
                st.metric(label="Примерное игровое время (кол-во спинов)", value=spins_str)
            
            # --- ВОЗВРАЩЕННЫЙ БЛОК С РАЗВЕРНУТЫМ ОБЪЯСНЕНИЕМ ---
            with st.expander("Как понимать эти цифры и почему мы считаем спины 'до разорения'? 🤔"):
                st.markdown("""
                Это ключевой момент для понимания стратегии игры в слоты.

                #### 1. Математическое ожидание всегда отрицательно
                В любой игре казино с RTP (Return to Player) меньше 100%, каждый ваш спин в среднем **уменьшает** ваш банкролл. 
                Величина этого уменьшения равна `Ваша Ставка * (1 - RTP)`. Это означает, что при бесконечно долгой игре **математически неизбежным исходом является проигрыш всего банкролла**. 
                Поэтому "количество спинов до разорения" — это единственная величина, которую можно рассчитать как математическое ожидание. **Это ваш "запас прочности" против преимущества казино.**

                #### 2. Выигрыш — это отклонение от нормы (дисперсия)
                Ваш желаемый выигрыш — это не математически ожидаемый результат, а **счастливая случайность (положительное отклонение от среднего)**. Вы не "движетесь" к выигрышу с каждым спином. Вместо этого, вы медленно "дрейфуете" в сторону проигрыша, надеясь поймать одну или несколько крупных выигрышных комбинаций, которые перекроют все предыдущие потери и выведут вас в плюс.

                #### 3. Максимизация спинов = Максимизация шансов
                Учитывая это, стратегия игрока сводится к следующему: **продержаться в игре как можно дольше, чтобы дать удаче (дисперсии) шанс сработать в вашу пользу.** 
                Чем больше спинов вы можете сделать на свой банкролл, тем больше у вас попыток "поймать" тот самый крупный выигрыш, который позволит достичь цели.

                ---
                
                #### Аналогия: Пловец против течения
                
                Представьте, что вы плывете против течения (**преимущество казино**). Ваша цель — доплыть до спасательного круга (**желаемый выигрыш**), который случайным образом появляется где-то впереди. Ваша выносливость (**банкролл**) — это то, как долго вы можете плыть. 
                
                **"Примерное игровое время"** — это расчетное время, которое вы можете продержаться на воде, прежде чем течение вас унесет (**разорение**). Чем дольше вы держитесь, тем выше шанс, что за это время рядом с вами появится спасательный круг.
                """)

            st.subheader("📊 Визуальный анализ вероятностей", divider="blue")
            fig = calculator.visualize_win_probabilities()
            if fig: st.pyplot(fig)
            st.dataframe(calculator.get_results_table(), use_container_width=True)

            st.header("♟️ Персональная стратегия игры", divider="rainbow")
            
            with st.container(border=True):
                st.subheader("1. Вердикт о вашем банкролле")
                for advice in strategy['min_bank_advice']: st.markdown(f"➡️ {advice}")
            
            with st.container(border=True):
                st.subheader("2. Обоснование и Расчет Минимального Банка")
                st.markdown("Чтобы стратегия имела смысл, ваш банкролл должен позволять пережить серии проигрышей, характерные для данной волатильности.")
                
                st.markdown("\n**Исходные данные для расчета:**")
                st.markdown(f" • **Минимальная ставка**: ${calculator.min_bet:.2f}")
                st.markdown(f" • **Максимальный выигрыш в слоте**: ${calculator.max_win:,.2f}")
                st.markdown(f" • **Средний значимый выигрыш**: ${calculator.avg_win:,.2f} (эмпирическая оценка)")
                st.markdown(f" • **Волатильность**: {calculator.volatility.capitalize()}")
                
                st.markdown("\n**Процесс расчета:**")
                st.markdown(f"1. **Формула** (для {calculator.volatility.capitalize()} волатильности): `{calculator.min_bankroll_formula}`")
                st.markdown(f"2. **Подставляем значения**: `{calculator.min_bankroll_calculation}`")

                min_bankroll_final_str = ''.join(filter(lambda char: char.isdigit() or char in '.,', strategy['min_bank_advice'][0].split('$')[-1]))
                st.success(f"**Результат**: Итоговый рекомендуемый минимум составляет **${min_bankroll_final_str}**")

            with st.container(border=True):
                st.subheader("3. Жесткая правда о шансах (без прикрас)")
                for truth in strategy['harsh_truths']: st.markdown(f"➡️ {truth}")
            with st.container(border=True):
                st.subheader("4. Оптимальная пошаговая стратегия")
                for i, step in enumerate(strategy['optimal_strategy'], 1): st.markdown(f"**Шаг {i}**: {step}")
        except Exception as e:
            st.error(f"Произошла ошибка при анализе файла. Убедитесь, что JSON-файл имеет верную структуру. Ошибка: {e}")
    elif not uploaded_file and analyze_button:
        st.warning("Пожалуйста, загрузите JSON-файл конфигурации слота, чтобы начать анализ.")

if __name__ == "__main__":
    main()
