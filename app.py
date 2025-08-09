# ==============================================================================
#  app.py - УНИВЕРСАЛЬНЫЙ АНАЛИЗАТОР СЛОТОВ V8.1
#  Изменения:
#   • bet_step: если не задан в JSON, равен min_bet (совместимо с Dragon Hatch)
#   • Оценка шанса цели: броуновская модель с волатильностью (реалистичнее, чем RTP-эвристика)
#   • Вероятность с учётом Wild: pure + (1 - pure) * wild_prob * wild_power + клампы
#   • «Гарантированное» → «Минимально оплачиваемых спинов»
# ==============================================================================

import json
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
from decimal import Decimal, ROUND_FLOOR, getcontext

getcontext().prec = 28

# --- Конфигурация ---
CONFIGS_FOLDER = "."  # Папка с предустановленными конфигами

@st.cache_data
def get_local_config_files(folder_path):
    """Получает список JSON файлов из указанной локальной папки."""
    try:
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
            return sorted(files)
        return []
    except Exception as e:
        print(f"Ошибка при получении списка файлов из {folder_path}: {e}")
        return []

sns.set_theme(style="whitegrid", palette="viridis")

class SlotProbabilityCalculator:
    def __init__(self, config):
        self.config = config
        self.win_probabilities = None

        gc = self.config.get('game_config', {}) or {}
        bet_range = gc.get('bet_range', [0.10, 100.00]) or [0.10, 100.00]
        if not isinstance(bet_range, list) or len(bet_range) < 2:
            bet_range = [0.10, 100.00]

        self.min_bet = float(bet_range[0])
        self.max_bet = float(bet_range[1])

        # Шаг ставки: сначала ищем явные поля, иначе = min_bet (совместимо с твоей логикой)
        self.bet_step = float(
            gc.get('bet_step') or
            gc.get('bet_increment') or
            gc.get('stake_step') or
            self.min_bet
        )

        self.max_win_at_min_bet, self.avg_win = None, None
        self.min_bankroll_formula, self.min_bankroll_calculation = "", ""
        self.volatility = gc.get('volatility', 'medium')
        self.calculate_all()

    # --- помощник: округление вниз к шагу ---
    def _floor_to_step(self, value, step):
        v = Decimal(str(value))
        s = Decimal(str(step))
        if s <= 0:
            return float(v)
        steps = (v / s).to_integral_value(rounding=ROUND_FLOOR)
        return float(steps * s)

    def _clamp_prob(self, p):
        return max(0.0, min(0.999999, float(p)))

    def calculate_all(self):
        self.calculate_min_bankroll()
        self.calculate_win_probabilities()

    def calculate_win_probabilities(self):
        symbols = {s['id']: s for s in self.config.get('symbols', [])}
        if not symbols:
            self.win_probabilities = {}
            return

        probabilities = self.config.get('probabilities', {})
        wild_power = float(probabilities.get('wild_substitution_power', 0.8))
        wild_symbol = next((s for s in symbols.values() if s.get('type') == 'wild'), None)
        wild_prob = float(wild_symbol.get('base_frequency', 0)) if wild_symbol else 0.0

        win_probs = {}
        for symbol_id, data in symbols.items():
            pure_prob = float(data.get('base_frequency', 0.0))
            if data.get('type') == 'wild':
                combo_prob = pure_prob
            else:
                # Эвристика: вклад вайлда учитываем только если базовый символ не случился
                combo_prob = pure_prob + (1.0 - pure_prob) * wild_prob * wild_power

            pure_prob = self._clamp_prob(pure_prob)
            combo_prob = self._clamp_prob(combo_prob)

            if combo_prob <= 0:
                spins_for_99 = float('inf')
            else:
                spins_for_99 = math.log(0.01) / math.log(1 - combo_prob)

            win_probs[symbol_id] = {
                'name': data.get('name', symbol_id),
                'pure_probability': pure_prob,
                'combo_probability': combo_prob,
                'spins_for_99_prob': spins_for_99
            }
        self.win_probabilities = {'base': win_probs}

    def calculate_min_bankroll(self):
        bet_range = self.config.get('game_config', {}).get('bet_range', [0.10, 100.00])
        if not isinstance(bet_range, list) or len(bet_range) < 2:
            bet_range = [0.10, 100.00]
        self.min_bet, self.max_bet = float(bet_range[0]), float(bet_range[1])

        max_win_multiplier = float(self.config.get('probabilities', {}).get('max_win_multiplier', 2000))
        self.max_win_at_min_bet = max_win_multiplier * self.min_bet
        self.avg_win = 0.4 * self.max_win_at_min_bet

        if self.volatility == 'high':
            part1, part2 = 100 * self.min_bet, 0.05 * self.avg_win
            self.min_bankroll_formula = "max(100 * Мин. ставка, 5% * Среднего выигрыша)"
            self.min_bankroll_calculation = f"max(${part1:.2f}, ${part2:.2f})"
            min_bankroll = max(part1, part2)
        elif self.volatility == 'medium':
            part1, part2 = 75 * self.min_bet, 0.03 * self.avg_win
            self.min_bankroll_formula = "max(75 * Мин. ставка, 3% * Среднего выигрыша)"
            self.min_bankroll_calculation = f"max(${part1:.2f}, ${part2:.2f})"
            min_bankroll = max(part1, part2)
        else:
            part1, part2 = 50 * self.min_bet, 0.01 * self.avg_win
            self.min_bankroll_formula = "max(50 * Мин. ставка, 1% * Среднего выигрыша)"
            self.min_bankroll_calculation = f"max(${part1:.2f}, ${part2:.2f})"
            min_bankroll = max(part1, part2)

        return round(min_bankroll, 2)

    def generate_bankroll_strategy(self, personal_bankroll, risk_level='medium'):
        min_bankroll = self.calculate_min_bankroll()
        min_bank_advice = []
        if personal_bankroll < min_bankroll:
            pb_formatted = f"{personal_bankroll:,.2f}"
            mb_formatted = f"{min_bankroll:,.2f}"
            min_bank_advice.append(
                f"🚨 **КРИТИЧЕСКИЙ РИСК**: Ваш банкролл (${pb_formatted}) **ЗНАЧИТЕЛЬНО НИЖЕ** минимального (${mb_formatted})!"
            )
            min_bank_advice.append(
                "Вероятность проигрыша всего банка до получения значимого выигрыша **превышает 95%**. Мы **НЕ РЕКОМЕНДУЕМ** играть с таким банком."
            )
        else:
            pb_formatted = f"{personal_bankroll:,.2f}"
            mb_formatted = f"{min_bankroll:,.2f}"
            min_bank_advice.append(f"✅ Ваш банкролл (${pb_formatted}) достаточен для игры в этот слот (минимум: ${mb_formatted}).")

        risk_multiplier_map = {'low': 1, 'medium': 2, 'high': 5}
        risk_multiplier = risk_multiplier_map.get(risk_level, 2)

        bankroll_power_base = 50
        bankroll_multiplier = (
            max(1, 1 + math.log10(personal_bankroll / bankroll_power_base))
            if personal_bankroll > bankroll_power_base else 1
        )

        theoretical_bet = self.min_bet * risk_multiplier * bankroll_multiplier

        bet_step = self.bet_step
        snapped_bet = self._floor_to_step(theoretical_bet, bet_step)

        safe_max_bet_raw = min(self.max_bet, personal_bankroll / 20)
        safe_max_bet = self._floor_to_step(safe_max_bet_raw, bet_step)

        bet_per_spin = max(self.min_bet, min(snapped_bet, safe_max_bet))
        bet_per_spin = self._floor_to_step(bet_per_spin, bet_step)

        tb_formatted = f"{theoretical_bet:.2f}"
        bps_formatted = f"{bet_per_spin:.2f}"
        mbet_formatted = f"{self.min_bet:.2f}"

        adjustment_note = ""
        if abs(bet_per_spin - theoretical_bet) > 0.01:
            if bet_per_spin == self.min_bet:
                adjustment_note = (
                    f" (Примечание: теоретическая ставка ${tb_formatted} была **скорректирована** до минимально возможной в этом слоте)."
                )
            elif bet_per_spin < theoretical_bet:
                adjustment_note = (
                    f" (Примечание: теоретическая ставка ${tb_formatted} была **уменьшена и округлена** "
                    f"до шага ${bet_step:.2f} для вашей безопасности и соответствия лимитам слота)."
                )

        base_win_prob = float(self.config.get('probabilities', {}).get('base_win_probability', 0.25))
        rtp = float(self.config.get('game_config', {}).get('rtp', 0.96))

        bwp_pct = base_win_prob * 100
        losing_spins_count = 10 - int(base_win_prob * 10)
        rtp_pct = rtp * 100
        house_edge_val = 1000 * (1 - rtp)
        hev_formatted = f"{house_edge_val:.2f}"

        truth1 = f"Вероятность любого выигрыша за спин: **{bwp_pct:.1f}%**. Это означает, что в среднем **~{losing_spins_count} из 10 спинов будут проигрышными**."
        truth2 = f"**RTP {rtp_pct:.1f}%** означает, что на каждый поставленный $1,000, казино в среднем оставляет себе **${hev_formatted}**."
        harsh_truths = [truth1, truth2]

        stop_loss_profile = {'low': 0.25, 'medium': 0.4, 'high': 0.5}
        win_goal_profile = {'low': 0.4, 'medium': 0.6, 'high': 1.0}

        sll_val = personal_bankroll * (1 - stop_loss_profile[risk_level])
        sll_loss = personal_bankroll * stop_loss_profile[risk_level]
        wgl_val = personal_bankroll * (1 + win_goal_profile[risk_level])
        wgl_profit = personal_bankroll * win_goal_profile[risk_level]

        sll_val_f = f"{sll_val:.2f}"
        sll_loss_f = f"{sll_loss:.2f}"
        wgl_val_f = f"{wgl_val:.2f}"
        wgl_profit_f = f"{wgl_profit:.2f}"

        strategy1 = (
            f"**Рекомендуемая ставка**: Для вашего банка и уровня риска реальная ставка составляет **${bps_formatted}** "
            f"(шаг ставки: ${bet_step:.2f}).{adjustment_note}"
        )
        strategy2 = f"**Управление ставками**: Начинайте с минимальной ставки **${mbet_formatted}**. Если игра идет хорошо, можно постепенно повышать ставку, но не превышать рекомендуемую."
        strategy3 = f"**Стоп-лосс (железное правило)**: Немедленно прекратите игру, если ваш банк опустится до **${sll_val_f}** (потеря ${sll_loss_f})."
        strategy4 = f"**Цель выигрыша**: Зафиксируйте прибыль и прекратите игру, если ваш банк достигнет **${wgl_val_f}** (прибыль ${wgl_profit_f})."
        strategy5 = "**Психология**: **НИКОГДА** не пытайтесь 'отыграться'. Каждый спин независим."

        optimal_strategy = [strategy1, strategy2, strategy3, strategy4, strategy5]

        return {
            'min_bank_advice': min_bank_advice,
            'harsh_truths': harsh_truths,
            'optimal_strategy': optimal_strategy,
            'bet_per_spin': bet_per_spin
        }

    def estimate_goal_chance(self, personal_bankroll, desired_win, bet_per_spin=None):
        """
        Оценка вероятности достичь цели до разорения.
        Броуновская модель с дрейфом и волатильностью:
        μ = bet * (RTP - 1);  σ ≈ k(volatility) * bet
        """
        if desired_win <= 0:
            return {"probability": 1.0}
        if personal_bankroll <= 0:
            return {"probability": 0.0}

        rtp = float(self.config.get('game_config', {}).get('rtp', 0.96))

        if bet_per_spin is None:
            # На всякий случай (обычно передают рекомендованную ставку)
            bet_per_spin = max(self.min_bet, min(personal_bankroll / 20, self.max_bet))
            bet_per_spin = self._floor_to_step(bet_per_spin, self.bet_step)

        mu = bet_per_spin * (rtp - 1.0)
        vol_map = {"low": 0.6, "medium": 1.0, "high": 1.8}
        sigma = vol_map.get(self.volatility, 1.0) * bet_per_spin
        sigma2 = max(1e-12, sigma * sigma)

        B = float(personal_bankroll)
        W = float(desired_win)

        if abs(mu) < 1e-12:
            prob = B / (B + W)
        else:
            num = 1.0 - math.exp(-2.0 * mu * B / sigma2)
            den = 1.0 - math.exp(-2.0 * mu * (B + W) / sigma2)
            prob = 0.0 if abs(den) < 1e-18 else num / den

        prob = max(0.0, min(1.0, prob))
        return {"probability": prob}

    def visualize_win_probabilities(self, level='base'):
        if not self.win_probabilities:
            return None
        level_data = self.win_probabilities.get(level)
        if not level_data:
            return None
        df = pd.DataFrame.from_dict(level_data, orient='index').sort_values('combo_probability', ascending=False)
        if df.empty:
            return None
        df['combo_probability_pct'] = df['combo_probability'] * 100
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            x='combo_probability_pct', y='name', data=df,
            palette='viridis_r', orient='h', hue='name', legend=False, ax=ax
        )
        ax.set_title(f'Оценка вероятности хита с символом (Уровень: {level})', fontsize=16, pad=20)
        ax.set_xlabel('Оценка вероятности за спин (с учётом Wild), %', fontsize=12)
        ax.set_ylabel('Символ', fontsize=12)
        for p in ax.patches:
            width = p.get_width()
            ax.text(width + 0.05, p.get_y() + p.get_height() / 2., f'{width:.3f}%', va='center', fontsize=10)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.tight_layout()
        return fig

    def get_results_table(self, level='base'):
        if not self.win_probabilities:
            return pd.DataFrame()
        level_data = self.win_probabilities.get(level)
        if not level_data:
            return pd.DataFrame()
        df = pd.DataFrame.from_dict(level_data, orient='index')
        if df.empty:
            return df
        df_sorted = df.sort_values(by='combo_probability', ascending=False)
        df_display = pd.DataFrame({
            'Символ': df_sorted['name'],
            'Чистая вероятность (%)': df_sorted['pure_probability'].apply(lambda x: f"{x*100:.2f}%"),
            'Оценка вероятности хита (с Wild, %)': df_sorted['combo_probability'].apply(lambda x: f"{x*100:.2f}%"),
            'Спинов до 99% вероятности хита': df_sorted['spins_for_99_prob'].apply(lambda x: f"{int(x)}" if x != float('inf') else "∞")
        })
        return df_display

# --- Основной блок Streamlit ---
def main():
    st.set_page_config(page_title="Анализатор слотов", layout="wide", initial_sidebar_state="expanded")

    local_config_files = get_local_config_files(CONFIGS_FOLDER)

    with st.sidebar:
        st.title("🎰 Параметры Анализа")

        file_source = st.radio(
            "Выберите источник конфигурации:",
            ('Загрузить файл с компьютера', 'Выбрать из предустановленных'),
            index=0
        )

        config_file = None

        if file_source == 'Загрузить файл с компьютера':
            config_file = st.file_uploader("1a. Загрузите JSON-конфигурацию слота", type="json")
        elif file_source == 'Выбрать из предустановленных' and local_config_files:
            selected_filename = st.selectbox(
                "1b. Выберите конфигурацию слота",
                options=local_config_files,
                format_func=lambda x: x
            )
            if selected_filename:
                try:
                    full_path = os.path.join(CONFIGS_FOLDER, selected_filename)
                    with open(full_path, 'rb') as f:
                        raw_bytes = f.read()
                    from io import BytesIO
                    config_file = BytesIO(raw_bytes)
                    config_file.name = selected_filename
                except Exception as e:
                    st.error(f"Ошибка при загрузке файла {selected_filename}: {e}")
                    config_file = None
        elif file_source == 'Выбрать из предустановленных' and not local_config_files:
            st.info(f"Папка '{CONFIGS_FOLDER}' не найдена или пуста.")

        personal_bankroll, desired_win, risk_level, analyze_button = 0, 0, 'medium', False
        if config_file is not None:
            personal_bankroll = st.number_input("2. Ваш стартовый банкролл ($)", min_value=0.01, value=200.0, step=10.0, format="%.2f")
            desired_win = st.number_input("3. Ваш желаемый чистый выигрыш ($)", min_value=1.0, value=500.0, step=10.0, format="%.2f")
            risk_level = st.selectbox("4. Ваш уровень риска", options=['low', 'medium', 'high'], index=1).lower()
            analyze_button = st.button("🚀 Провести полный анализ", type="primary", use_container_width=True)

    st.title("Универсальный анализатор вероятностей слотов")
    st.markdown("Этот инструмент помогает оценить шансы и подобрать ставку под ваш банкролл. Часть метрик — эвристики; используйте их как ориентир, а не истину в последней инстанции.")

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
                st.warning("К сожалению, анализ невозможен. Пожалуйста, увеличьте банкролл.")
                st.stop()

            game_config = config.get('game_config', {})
            gn_formatted = game_config.get('game_name', 'N/A')
            pb_formatted = f"{personal_bankroll:,.2f}"
            dw_formatted = f"{desired_win:,.2f}"
            rl_formatted = risk_level.capitalize()

            st.header(f"🎰 Полный анализ слота: {gn_formatted}", divider="rainbow")
            st.markdown(f"### Ваши параметры: Банкролл: ${pb_formatted} | Цель: +${dw_formatted} | Риск: **{rl_formatted}**")

            # Стратегия и ставка
            strategy = calculator.generate_bankroll_strategy(personal_bankroll, risk_level)
            bet_per_spin = strategy.get('bet_per_spin')

            # Оценка цели (улучшенная модель)
            goal_result = calculator.estimate_goal_chance(personal_bankroll, desired_win, bet_per_spin)

            # Минимально оплачиваемых спинов (без единого выигрыша)
            min_payable_spins = int(personal_bankroll / bet_per_spin) if bet_per_spin > 0 else float('inf')

            st.subheader("🎯 Вероятность достижения цели и ресурс спинов", divider="blue")
            col1, col2 = st.columns(2)
            with col1:
                dw_label_formatted = f"{desired_win:,.2f}"
                st.metric(
                    label=f"Оценочный шанс (модель с волатильностью) на +${dw_label_formatted}",
                    value=f"{goal_result['probability']*100:.2f}%"
                )
            with col2:
                spins_str = f"{min_payable_spins}" if min_payable_spins != float('inf') else "∞"
                st.metric(label="Минимально оплачиваемых спинов (при рек. ставке)", value=spins_str)

            with st.expander("Как это понимать? 🤔"):
                st.markdown(f"""
                - **Оценочный шанс** рассчитывается по стохастической модели с учётом RTP и волатильности (не точная матмодель слота, а приближение).
                - **Минимально оплачиваемых спинов** — сколько спинов вы гарантированно оплатите даже если не будет ни одного выигрыша.
                - **Рекомендуемая ставка**: ${bet_per_spin:.2f} (шаг ставки: ${calculator.bet_step:.2f}; мин. ставка: ${calculator.min_bet:.2f}).
                """)

            st.subheader("📊 Оценки вероятностей символов", divider="blue")
            fig = calculator.visualize_win_probabilities()
            if fig:
                st.pyplot(fig)
            st.dataframe(calculator.get_results_table(), use_container_width=True)
            st.caption("Примечание: для кластер-слотов оценки носят эвристический характер; точная вероятность зависит от расположения/каскадов.")

            st.header("♟️ Персональная стратегия игры", divider="rainbow")
            with st.container(border=True):
                st.subheader("1) Вердикт о банкролле")
                for advice in strategy['min_bank_advice']:
                    st.markdown(f"➡️ {advice}")

            with st.container(border=True):
                st.subheader("2) Обоснование минимума банка")
                st.markdown("Ваш банк должен выдерживать типичные луз-стрейки для данной волатильности.")
                st.markdown("\n**Исходные данные для расчёта:**")
                st.markdown(f" • **Мин. ставка**: ${calculator.min_bet:.2f}")
                st.markdown(f" • **Шаг ставки**: ${calculator.bet_step:.2f}")
                st.markdown(f" • **Макс. выигрыш при мин. ставке**: ${calculator.max_win_at_min_bet:,.2f}")
                st.markdown(f" • **Средний значимый выигрыш (мин. ставка)**: ${calculator.avg_win:,.2f}")
                st.markdown(f" • **Волатильность**: {calculator.volatility.capitalize()}")
                st.markdown("\n**Формула и подстановка:**")
                st.markdown(f"`{calculator.min_bankroll_formula}` → `{calculator.min_bankroll_calculation}`")
                min_bankroll_value = calculator.calculate_min_bankroll()
                st.success(f"**Итоговый рекомендуемый минимум**: **${min_bankroll_value:,.2f}**")

            with st.container(border=True):
                st.subheader("3) Жёсткие факты о шансах")
                for truth in strategy['harsh_truths']:
                    st.markdown(f"➡️ {truth}")

            with st.container(border=True):
                st.subheader("4) Оптимальная пошаговая стратегия")
                for i, step in enumerate(strategy['optimal_strategy'], 1):
                    st.markdown(f"**Шаг {i}.** {step}")

        except json.JSONDecodeError:
            st.error("Ошибка: Выбранный файл не является корректным JSON.")
        except Exception as e:
            st.error(f"Произошла ошибка при анализе файла. Проверьте структуру JSON. Ошибка: {e}")
    elif analyze_button and config_file is None:
        st.warning("Пожалуйста, загрузите JSON-файл конфигурации слота или выберите его из списка.")

if __name__ == "__main__":
    main()
