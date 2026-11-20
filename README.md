# Лабораторная работа 4 — методы градиента политики

**Исполнитель:** Баранов Вадим, вариант 2.  
**Тема:** REINFORCE, инженерная базовая линия, VPG и TRPO в `CartPole-v1`.

## Результат

| Метод | Средняя награда | std | Доля эпизодов ≥ 475 |
|---|---:|---:|---:|
| REINFORCE | 108.3 | 58.2 | 0.0% |
| REINFORCE + baseline | 307.2 | 152.7 | 27.3% |
| VPG | 464.2 | 24.9 | 71.3% |
| TRPO | 481.3 | 20.6 | 87.3% |

Одинаковые условия: пять зерен `(12, 34, 56, 78, 90)`, `30 000` шагов на зерно, сеть политики `4–128–64–2`, жадная оценка на 30 эпизодах.

## Файлы

- `ЛР4_Методы_градиента_политики_2_hard.ipynb` — оформленный ноутбук с кодом, графиками и ответами на четыре обсуждения;
- `lab4_experiment.py` — исходный код агентов и экспериментального цикла;
- `results/metrics.json` — метрики пяти зерен для каждого метода;
- `results/figures/` — пять сохранённых графиков.

## Запуск

```powershell
py -3.12 -m pip install numpy pandas matplotlib torch gymnasium
py -3.12 -c "from lab4_experiment import run_all, to_jsonable; import json; from pathlib import Path; r=run_all(); Path('results').mkdir(exist_ok=True); Path('results/metrics.json').write_text(json.dumps(to_jsonable(r), ensure_ascii=False, indent=2), encoding='utf-8')"
```

После этого откройте ноутбук и выполните ячейки сверху вниз. Для просмотра сохранённых результатов оставьте `RUN_EXPERIMENTS = False`.
