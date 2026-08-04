# CayleyPy Integration: серия PR для модельного слоя, тренера и поиска

## Overview

Интеграция в библиотеку [cayleypy/cayleypy](https://github.com/cayleypy/cayleypy) функциональности, которая сейчас живёт вокруг неё в ноутбуках и личных репозиториях группы:

- контракт `score_children` для многовыходных моделей (Q-модели с выходом на каждый генератор) **и его потребитель в beam search** (child-scored шаг луча — без него модельный слой не даёт пользы конечному пользователю);
- расширенный `ModelConfig` + самоописывающий формат чекпойнта (лечит проблему «голых .pth»);
- архитектуры: ResMLP, Q-MLP (n выходов), токенизатор + Q-трансформер (архитектура Влада), AZ-головы (q+v, v-consistency Андрея — inference-time, внутри `QVModel.score_children`);
- ансамбли предикторов (плоская взвешенная сумма скоров — даёт эффект ×2 к ширине луча);
- симметрии: `SymmetryGroup` + TTA (один PR) и канонический дедуп в луче (отдельный PR);
- non-backtracking для `search_simple` (в нём сегодня нет никакого бана; в `search_advanced` это уже покрыто `history_depth`) — кандидат на выброс, если мейнтейнер возразит;
- тренер (лоссы отдельным PR; пайплайн, к которому независимо сошлись Влад и Андрей: random walks + sparse-Q + BFS-anchors + данные из путей beam search);
- **Bellman/DAVI-дообучение** — bootstrapped-таргеты `y(s) = 1 + min_a V(child_a)` с target-копией (идея из ноутбуков `alexandervc/cayleypy-rw-modelbaselines-megaminx`, включая подмешивание solved/соседей в Bellman-батч);
- **демонстрационный чекпойнт**, обученный этим тренером и зарегистрированный в `PREDICTOR_MODELS` — чтобы архитектурные PR не были «мёртвым кодом» (режим отказа, убивший #151);
- интерфейс `LowerBound` — жёсткие отсечения в луче по допустимой нижней оценке.

**Вне охвата** (решение пользователя): бэкенды — адаптер CUDA-луча Ивана (precompiled), JAX/TPU-бэкенд, MITM-beam. См. Post-Completion.

## Context (from discovery + verified by plan-review against upstream)

- **Целевой репозиторий**: `cayleypy/cayleypy` (внешний). У пользователя `push: false` → форк + PR. **Правила upstream (README, «How to contribute»): нужно ДВА апрува от команды ревьюеров; пуш после апрува сбрасывает апрув; имя PR должно отражать изменения.** Последний мерж в `main` — 2026-05-24; PR #157 ждёт ревью с 11.2025; #151 approved 10.2025 и не смержен. Ревью — узкое место всего плана.
- **Реальная структура** (проверено по коду `main`):
  - `cayleypy/predictor.py` — `Predictor`, `__call__` батчит через `torch.hstack` (**корректно только для 1-D выходов**);
  - `cayleypy/models/models.py` — `MlpModel` + `ModelConfig` (frozen dataclass); **фабрика — `ModelConfig._build_model` здесь же**; существующий `model_type` — строка `"MLP"` (uppercase); `cayleypy/models/models_lib.py` — только реестр претрейн-моделей `PREDICTOR_MODELS` (веса с Kaggle);
  - `cayleypy/algo/beam_search.py` — **два независимых пути**: `search_simple` (MITM-приёмы, `return_path`, банов нет) и `search_advanced` (`history_depth` — hash-бан предыдущих уровней). Никакого iterated-режима/`hashed_neigbourhood` в `main` нет (это неслитые #175/#157). **Оба пути скорят плоский глобально-дедуплицированный набор** `get_unique_states(get_neighbors(...))` — связь родитель→ход разрушается до скоринга;
  - `CayleyGraph.get_neighbors` пишет **generator-major** блоки (`neighbors[i*B:(i+1)*B]` = генератор i ко всем состояниям) — reshape в `[B, n_gen]` требует транспонирования;
  - `CayleyGraphDef.generators_inverse_map` **уже существует** (None, если набор не замкнут по обратным), `generators_inverse_closed` тоже;
  - `cayleypy/hasher.py`, `cayleypy/string_encoder.py`, `cayleypy/algo/random_walks.py` (есть `mode="nbt"` + `nbt_history_depth`);
  - тесты `*_test.py` рядом с модулем; тяжёлые — под `RUN_SLOW_TESTS = os.getenv("RUN_SLOW_TESTS") == "1"` + `skipif` (паттерн в 6 файлах); CI: `RUN_SLOW_TESTS=1 pytest` (Ubuntu/macOS), bare `pytest` (Windows), матрица **Python 3.9**–3.13, отдельный job `build-docs` (`docs/api.rst` — autosummary по полным именам, символы должны экспортироваться в `__init__.py`);
  - `./lint.sh` = black==25.1.0 (120) + mypy==1.15.0 + pylint; docstrings — Google style, комментарии заканчиваются точкой, pylint-warning'и чинить, а не отключать; reST `:param x:` в `beam_search.py`. **Уточнение из Task 1: отдельный CI-job `format-check` гоняет `black --check --diff .` по ВСЕМУ репозиторию (а `./lint.sh` — только по `./cayleypy`), поэтому любые новые .py вне пакета тоже должны быть отформатированы; на Python 3.9 CI ставит torch 2.8, а не 2.13 — новые torch-API проверять против 2.8.**
- **Зависимости**: `h5py, kagglehub, numba, numpy, scipy, torch>=2.6.0`. **torch>=2.6 ⇒ `torch.load` по умолчанию `weights_only=True`.** Pydantic НЕТ.
- **Конфликтная обстановка** (открытые PR upstream): **#157, #175, #177, #170 трогают `beam_search.py`; #170 трогает и `predictor.py`**; #151 (trainer, `cayleypy/trainers/`) — approved+stale; #188 (Иван, distributed BFS) — открыт. README: «Do not add new graphs to prepare_graph» — upstream сознательно избегает конфликтных файлов.
- **Доноры кода**: [cayleypy-training-core](https://github.com/AnanasClassic/cayleypy-training-core) (Влад), пайплайн Андрея (SparseQSampler / BFSAnchors / symmetry transport, 02.08.2026), эмпирика Стаса (anchors 1–2%, 10% вредит; 154→139).

## Development Approach

- **testing approach**: Regular (код, затем тесты в том же PR), `*_test.py` рядом с модулем
- **КРИТИЧНО: без новых runtime-зависимостей** — stdlib + имеющиеся torch/numpy; прогресс тренера — `verbose`-параметр + `print` (как в `beam_search.py`), никакого tqdm (в отличие от #151, тянувшего зависимости)
- **КРИТИЧНО: синтаксис Python 3.9** (CI-матрица): `Optional[X]`/`Union` вместо `X | None`, никаких `match`; в frozen dataclass — `field(default_factory=...)` для изменяемых дефолтов
- конвенции upstream: black 120, mypy, pylint (чинить, не отключать), Google-докстринги, комментарии с точкой на конце, reST `:param:`; каждый новый публичный класс — экспорт в `cayleypy/__init__.py` (+ subpackage `__init__.py`) и autosummary-запись в `docs/api.rst` в том же PR
- обратная совместимость: существующие `Predictor`/`ModelConfig`/`beam_search`/претрейн-модели (`PREDICTOR_MODELS`) работают без изменений
- **РЕЖИМ ВЫПОЛНЕНИЯ: все PR — только внутрь форка `stasdiener/cayleypy` (base = `main` форка или ветка-родитель). В upstream во время выполнения плана НЕ отправляется ничего — ни PR, ни issue.** Публикация в upstream — отдельная фаза после ручной проверки пользователем (см. Post-Completion)
  - **защита от случайного upstream-PR**: в клоне выполнить `gh repo set-default stasdiener/cayleypy` (иначе `gh pr create` на форке по умолчанию целится в upstream); в веб-интерфейсе при создании PR проверять base-репозиторий
  - мержи внутри форка делает пользователь (или по его решению — после самопроверки/агент-ревью); стекать base-ветками можно свободно — всё своё
  - PR держать маленькими и в конвенциях upstream (линт, 3.9, тесты) — они без переделки станут upstream-PR на фазе публикации
- PR, трогающие `beam_search.py` (PR2 → PR13 → PR14 → PR15), — **строго последовательно** (текстовые конфликты между собой), каждый следующий от ветки предыдущего; за открытыми upstream-PR #157/#175/#177/#170 следить read-only (осведомлённость о будущих конфликтах)
- один Task = один PR; завершать полностью перед началом зависимого
- **CRITICAL: every task MUST include new/updated tests** (success + error отдельными пунктами)
- **CRITICAL: all tests must pass before starting next task** (`./lint.sh && RUN_SLOW_TESTS=1 pytest` из корня)
- **CRITICAL: update this plan file when scope changes during implementation**

## Testing Strategy

- **unit tests**: обязательны в каждом PR; лёгкие тесты — быстрый путь (≤ пары секунд), тяжёлые (тренировка, мегаминкс, глубокий BFS) — под `RUN_SLOW_TESTS`-паттерном репо
- **ground truth**: точный BFS на малых графах; конкретные конструкторы, напр. `PermutationGroups.lrx(5)`; в тестах `device="cpu"` (конвенция после #184)
- **паритет-тесты с Kaggle-весами**: публичные веса грузить как в `models_lib_test` (прецедент — без skipif); skipif только для приватного
- **регресс претрейнов**: `models_lib_test.test_loads_predictor_models` должен оставаться зелёным после любых правок `MlpModel` (совместимость state_dict)
- **детерминизм**: `torch.manual_seed`; ассерты вида «final loss < X% от initial», не «лосс → ~0»
- локально перед пушем: `./lint.sh && RUN_SLOW_TESTS=1 pytest`; PR1 дополнительно прогнать под Python 3.9 (`uv run -p 3.9 pytest`)
- **e2e**: нет UI — не применимо

## Progress Tracking

- отмечать `[x]` сразу; новые задачи — ➕; блокеры (ждём 2-й апрув, ждём ответа автора весов) — ⚠️
- статусы PR вести в таблице в Post-Completion (внешние события — не чекбоксы)

## Solution Overview

**Стратегия: contract-first + consumer-first.** PR1 закладывает контракт `score_children` и ModelConfig v2; PR2 немедленно даёт контракту потребителя — child-scored шаг луча (иначе PR1–PR7 — мёртвый код). Дальше — независимые возможности, тонко нарезанные (уроки #151: большие PR в этом репо умирают).

**Граф зависимостей PR:**

```
PR1 (контракт+конфиг+чекпойнт)
 ├─ PR2 (child-scored beam step)        ── PR13 (канон-дедуп) ── PR14 (nbt) ── PR15 (LowerBound)
 ├─ PR3 (ResMLP/Q-MLP) ── PR6 (QV/AZ)   [PR2→13→14→15 строго последовательно: один файл]
 ├─ PR4 (токенизатор) ── PR5 (Q-трансформер)
 ├─ PR7 (ансамбли, плоские)
 └─ PR9 (тренер-ядро) ── PR10 (anchors+sparse-Q+beam-пути) ── PR11 (демо-чекпойнт в PREDICTOR_MODELS)
                          └─ PR16 (Bellman-дообучение; использует score_children из PR1)
PR8 (лоссы) ── независим, можно параллельно с PR1
PR12 (SymmetryGroup+TTA) ── зависит только от PR1
```

**Ключевые решения:**
1. `score_children(states) -> Tensor[B, n_gen]` в `Predictor`; дефолт — через скалярный предикт детей **с учётом generator-major раскладки `get_neighbors` (транспонирование)**; Q-модели переопределяют (1 forward родителя). Legacy-путь `Predictor.__call__` получает guard: 2-D выход модели → понятная ошибка (не тихий мисранк через `argsort`).
2. Чекпойнт: `torch.save({"config": <dict из примитивов/JSON-строка>, "state_dict": ...})`, загрузка **явно `torch.load(..., weights_only=True)`**; валидация `graph_hash` (sha256 от `{generator_type, generators (perm-списки или matrix.tolist()+modulo), central_state}`) против графа.
3. ModelConfig v2 — **плоские аддитивные поля** (`n_outputs: int = 1`, `tokenizer_groups: Optional[...] = None`, `graph_hash: Optional[str] = None`), а не dict-union: mypy-дружелюбно, `from_dict` расширяется явно. Существующий `model_type="MLP"` (uppercase) сохраняется; новые типы регистрируются в `ModelConfig._build_model`.
4. v-consistency — inference-time (по Андрею: «это только в бим серче»), внутри `QVModel.score_children`, не в API `Predictor`.
5. Тренер — пакет `cayleypy/train/` (не `trainers/` — не конфликтовать с #151; кредитовать vlzm в описании PR9).
6. Симметрии: источник — захардкоженные списки для малых головоломок в `cayleypy/puzzles/moves.py` (конвенция README) + опциональный helper-деривация для малых групп (slow test).

## Technical Details

- `score_children`: семантика «выход/столбец i = применить генератор i»; тест — **поколоночный** (`score_children(s)[:, i] == predict(gen_i(s))` для каждого i отдельно — ловит транспонирование, чего не делает сравнение целых тензоров через общий reshape)
- батчинг 2-D выходов в `Predictor.__call__`/внутренностях: `torch.cat(dim=0)` вместо `hstack` + тест с батчем > `graph.batch_size`
- child-scored beam step (PR2): держать соседей в `[n_states, n_gen]`-раскладке, скорить `score_children(parents)`, разворачивать скоры в порядке `get_neighbors`, затем unique/top-k с индексной картой обратно к скорам; провенанс «какой ход породил слот» сохраняется (нужен PR13/PR14); паритет-тест: Q-модель и её скалярный эквивалент дают идентичный луч на `PermutationGroups.lrx(5)`
- nbt (PR14): реюз `graph_def.generators_inverse_map` (обрабатывать None = не inverse-closed); ценность: бан для `search_simple` (там его нет) и O(1)-память как альтернатива `history_depth=1` для очень широких лучей — цифры в описание PR
- `LowerBound` (PR15): протокол `lb(states) -> Tensor[B]` (документировать допустимость: lb ≤ истина); `BfsLowerBound` поверх `BfsResult`/`bfs_bitmask`; параметр `prune_above: Optional[int]` — известная верхняя оценка длины (например, из прошлого прогона); None = без отсечения; **одиночный `Optional[LowerBound]`, не список** (второй реализации нет — YAGNI)
- ансамбль (PR7): плоский `EnsemblePredictor(members, weights)` — без вложенных ансамблей (спекулятивная общность)
- тренер: `TrainConfig` (frozen dataclass) — rw_length, n_walks, batch, lr, cosine-скедулер, ema_decay, loss (`mse`|`pinball(τ)`|masked-sparse), пропорции смеси данных (anchors default 1–2%)

## What Goes Where

- **Implementation Steps** (`[ ]`): код/тесты/PR в форке + этот план
- **Post-Completion**: статусы мержей (2 апрува — не в моей власти), бэкенды, миграция чужих весов, анонс

## Implementation Steps

Пути — относительно корня форка `cayleypy`.

### Task 1: Форк, окружение, инвентаризация

**Files:**
- Create: локальный клон форка
- Create: `docs/plans/notes/20260803-task1-inventory.md` (зафиксированные сигнатуры + инвентарь upstream-PR) ➕

- [x] `gh repo fork cayleypy/cayleypy --clone`; remote upstream; `pip install -e ".[lint,test]"`; `./lint.sh && RUN_SLOW_TESTS=1 pytest` зелёные на `main` — клон и remotes уже были на месте; окружение поднято через `uv venv --python 3.12 .venv` + `uv pip install -e ".[lint,test,dev]"`; lint зелёный (black 59 файлов, pylint 10.00/10, mypy clean), тесты 299 passed / 12 skipped / 3 xfailed
- [x] прочитать README «How to contribute» (2 апрува, сброс апрува пушем, имя PR) и «How to add a new predictor model» (модель обязана демонстрировать пользу в beam search + веса на Kaggle) — README.md:162‑178 и :194‑216, выжимка в разделе 2 заметок
- [x] прочитать целиком `predictor.py`, `models/models.py`, `algo/beam_search.py`, `hasher.py`, `cayley_graph.py::get_neighbors`, `cayley_graph_def.py::generators_inverse_map` — зафиксировать фактические сигнатуры (раздел 3 заметок; generator-major раскладка и `is_identity`-ветка `get_unique_states` подтверждены по коду)
- [x] инвентаризация открытых upstream-PR, трогающих `beam_search.py`/`predictor.py` (#157, #175, #177, #170) — **read-only**, ничего не постить; заметки для будущей фазы публикации (раздел 4 заметок; добавлены #151 и #188; `predictor.py` трогает только draft #170 — `torch.inference_mode()`)
- [x] `gh repo set-default stasdiener/cayleypy` в клоне — защита от случайного PR в upstream
- [x] включить GitHub Actions в форке (вкладка Actions → Enable) — сделано через API (`gh api -X PUT repos/stasdiener/cayleypy/actions/permissions -F enabled=true -f allowed_actions=all`), оба workflow (`ci.yaml`, `deploy-docs.yaml`) в состоянии `active`
- [x] проверить локальный запуск тестов под Python 3.9 (`uv run -p 3.9 pytest`) — база для PR1; сделано через отдельный `.venv39` (чтобы не пересоздавать основной venv): 3.9.6 + torch 2.8.0, те же 299 passed / 12 skipped / 3 xfailed
- [x] (design-issue в upstream НЕ открываем — отложено до фазы публикации, см. Post-Completion) — ничего не постилось, в upstream только read-only чтения

### Task 2: PR1 — контракт score_children + ModelConfig v2 + чекпойнт

**Files:**
- Modify: `cayleypy/predictor.py`, `cayleypy/predictor_test.py`
- Modify: `cayleypy/models/models.py`
- Create: `cayleypy/models/checkpoint.py`, `cayleypy/models/checkpoint_test.py`
- Create: `cayleypy/models/models_test.py`
- Modify: `cayleypy/__init__.py`, `cayleypy/models/__init__.py`, `docs/api.rst`

- [x] ветка `feat/score-children-contract`; `Predictor.score_children` с дефолтом через скалярный предикт детей; **учесть generator-major раскладку `get_neighbors` (транспонирование при reshape в `[B, n_gen]`)** — `predictor.py`: `reshape((n_gen, num_states)).transpose(0, 1).contiguous()`; число состояний берётся из `encode_states(...)`, поэтому 1-D вход (одно состояние) тоже корректен
- [x] исправить батчинг для 2-D выходов (`torch.cat(dim=0)` вместо `hstack`); guard в legacy `__call__`: 2-D выход модели → понятная ошибка — ➕ батчинг вынесен в публичный `Predictor.predict_batched` (возвращает сырой выход модели, 1-D или 2-D), а `__call__` = `predict_batched` + проверка `len(ans.shape) != 1`; так guard в одном месте, а PR3/PR6/PR7 получают готовую точку входа для многовыходных моделей
- [x] `ModelConfig`: плоские поля `n_outputs=1`, `tokenizer_groups: Optional=None`, `graph_hash: Optional[str]=None`; явно расширить `from_dict`; `Optional[...]`-синтаксис (3.9) — плюс `to_dict()` (через `dataclasses.asdict`, только примитивы) для чекпойнта; `tokenizer_groups` — `Optional[list[list[int]]]`, пары `[group_size, num_groups]` (мегаминкс = `[[3, 20], [2, 30]]`), семантика зафиксирована в докстринге для PR4; `MlpModel` при `n_outputs != 1` даёт понятную ошибку (многовыходной MLP — PR3)
- [x] `checkpoint.py`: save/load (config — примитивы; **`torch.load(..., weights_only=True)` явно**); `graph_hash(graph_def)` c поддержкой perm-списков и `MatrixGenerator` (`matrix.tolist()+modulo`) + `central_state` — формат `{format_version, config, state_dict}`, отказ на голом state_dict и на версии формата из будущего; `CayleyGraphDef` импортируется только под `TYPE_CHECKING` (ветвление через `is_permutation_group()`) — ноль рантайм-связи с родительским пакетом; ➕ **скоуп: `ModelConfig._build_model` переименован в публичный `build_model`** (нужен загрузчику чекпойнта; иначе protected-access-варнинг pylint, который по README чинят, а не отключают) — регистрация новых типов моделей в PR3/PR5/PR6 идёт в него, по-прежнему в `models.py`; `ModelConfig.load` тоже получил явный `weights_only=True`
- [x] экспорт новых символов в `__init__.py`, autosummary в `docs/api.rst` — `cayleypy/models/__init__.py`: `graph_hash`, `save_checkpoint`, `load_checkpoint`; `cayleypy/__init__.py`: `ModelConfig`, `save_checkpoint`, `load_checkpoint`; api.rst — 3 записи в «Beam search and ML»; `docs/build_docs.sh` собирается с `-W`
- [x] write tests (success): поколоночный тест score_children на `PermutationGroups.lrx(5)` (`device="cpu"`); round-trip чекпойнта c `weights_only=True`; `from_dict` со старым словарём; батч > `graph.batch_size` — `predictor_test.py` (+7 тестов: поколоночный на 4 состояниях со sanity-проверкой различности столбцов, одиночное состояние, дети не влезают в один батч, 2-D выход выживает батчинг), `models/models_test.py` (7), `models/checkpoint_test.py` (9, включая матричный граф и `to_dict`→JSON)
- [x] write tests (error/edge): отказ загрузки при неверном `graph_hash`; guard legacy-пути на 2-D модели; matrix-графы в `graph_hash` — плюс голый state_dict, версия формата из будущего, неизвестный `model_type`, `n_outputs != 1` для MLP, `graph_hash` не меняется от `with_name` и меняется от генераторов/central_state/modulo
- [x] run `./lint.sh && RUN_SLOW_TESTS=1 pytest` (и `uv run -p 3.9 pytest`) — must pass before next task — lint зелёный (black 62 файла, pylint 10.00/10, mypy 62 файла), `black --check .` по всему репо зелёный; `RUN_SLOW_TESTS=1 pytest` = **323 passed / 12 skipped / 3 xfailed** и на 3.12 (torch 2.13), и на 3.9 (`.venv39`, torch 2.8); `models_lib_test.test_loads_predictor_models` зелёный (совместимость Kaggle-весов)
- [x] открыть PR в форк (`gh pr create --repo stasdiener/cayleypy`) — [stasdiener/cayleypy#1](https://github.com/stasdiener/cayleypy/pull/1), base `main` форка, в upstream не отправлялось

### Task 3: PR2 — child-scored beam step (потребитель контракта)

**Files:**
- Modify: `cayleypy/algo/beam_search.py`, `cayleypy/algo/beam_search_test.py`

- [x] ветка `feat/child-scored-beam` от PR1; шаг луча: соседи в `[n_states, n_gen]`-раскладке, скоринг `score_children(parents)` до дедупа, unique/top-k через индексную карту; сохранить провенанс «ход, породивший слот» — новая приватная функция `_expand_layer(graph, states) -> _ExpandedLayer(states, hashes, moves, source_index)` в `beam_search.py`: дедуп идентичен `get_unique_states` (сорт по хэшу + первое вхождение), но дополнительно отдаёт `source_index` (индекс в generator-major выходе `get_neighbors` — по нему берётся скор) и `moves` (ид генератора = `source_index // n_states`); скоры считаются `_score_children` = `predictor.score_children(parents)` + транспонирование в порядок `get_neighbors`
- [x] определить, в какой из путей это входит (`search_simple` и/или `search_advanced`) с учётом судьбы #157 — зафиксировать в PR — **решение: только `search_simple`**; в `search_advanced` состояния дополнительно фильтруются по хэшам прошлых уровней (пришлось бы переиндексировать скоры), и именно этот цикл переписывает неслитый #157; к тому же провенанс нужен PR13/PR14 именно в `search_simple`. Зафиксировано в описании PR и в тексте ошибки
- [x] опция включения (дефолт — старое поведение), прокинуть через диспатч `search()` — `use_child_scores: bool = False` в `search()` и `search_simple()`; при `beam_mode="advanced"` — понятная ошибка; при `False` выполняется ровно старый код (ветка `expanded is None`)
- [x] write tests (success): паритет — Q-модель и скалярный эквивалент дают идентичный луч на `lrx(5)`; старый путь без опции не изменился (существующие тесты без правок) — паритет сделан на `lrx(8)` (на `lrx(5)` луч не обрезается — скоринг вообще не вызывается): Q-модель (хэмминг всех детей за один вызов) против `Predictor(graph, "hamming")` — совпадают путь и `debug_scores`; плюс тест «тот же предиктор с опцией и без» на успешном поиске и на 50-шаговом неуспешном (>40 сравниваемых шагов); плюс тест провенанса (применение `moves[i]` к родителю воспроизводит состояние, `states`/`hashes` равны `get_unique_states`); существующие тесты не правились
- [x] write tests (error/edge): модель c `n_outputs != n_gen` → понятная ошибка; пустой фронтир — плюс `use_child_scores` в режиме "advanced" → понятная ошибка. Пустой фронтир проверен на `_expand_layer`/`_score_children` (в самом `search_simple` он недостижим); ⚠️ с дефолтным `Predictor.score_children` пустой вход падает в `StringEncoder.encode` (`torch.min` по пустому тензору, `string_encoder.py:52`) — **предсуществующий баг upstream, не в охвате PR2** (скалярный путь падает там же), поэтому тест использует предиктор без переэнкодинга
- [x] run `./lint.sh && RUN_SLOW_TESTS=1 pytest` — must pass before next task — lint зелёный (black 62, pylint 10.00/10, mypy 62 файла), `black --check .` по всему репо зелёный, `docs/build_docs.sh` (`-W`) зелёный; `RUN_SLOW_TESTS=1 pytest` = **330 passed / 12 skipped / 3 xfailed** и на 3.12 (torch 2.13), и на 3.9 (`.venv39`, torch 2.8)
- [x] открыть PR (base = ветка PR1, Draft до мержа PR1) — [stasdiener/cayleypy#2](https://github.com/stasdiener/cayleypy/pull/2), Draft, base `feat/score-children-contract`, в upstream не отправлялось

### Task 4: PR3 — ResMLP и многовыходной MLP

**Files:**
- Modify: `cayleypy/models/models.py`, `cayleypy/models/models_test.py`
- Modify: `cayleypy/predictor.py`, `cayleypy/predictor_test.py` (быстрый путь живёт в `score_children`) ➕
- Modify: `cayleypy/models/__init__.py`, `docs/api.rst`

- [x] ветка `feat/resmlp-qmlp` от PR1; `ResMlpModel` (Linear+LN+ReLU+skip; hidden, n_blocks, n_outputs); `MlpModel` c `n_outputs>1` **без изменения ключей/форм state_dict при n_outputs=1** (совместимость претрейнов) — `n_blocks`/`hidden` берутся из существующего `layers_sizes` (новых полей конфига не нужно): блоков `len(layers_sizes)`, i-й шириной `layers_sizes[i]`; skip есть у всех блоков, кроме меняющих число фич (первый — проекция из one-hot), т.е. `[512, 512, 512]` = проекция + 2 residual-блока; у `MlpModel` последний слой стал `Linear(in, n_outputs)` — при `n_outputs=1` формы и ключи (`layers.N.*`) не изменились
- [x] регистрация в `ModelConfig._build_model` (модуль `models.py`, НЕ `models_lib.py`); типы согласовать с существующим `"MLP"` (uppercase) — тип `"RESMLP"` в `ModelConfig.build_model` (переименован в PR1); экспорт `MlpModel`/`ResMlpModel` из `cayleypy/models/__init__.py` + 2 записи в `docs/api.rst`
- [x] быстрый путь `score_children` при `n_outputs == n_gen` — диспетчеризация по атрибуту `n_outputs` модели (`Predictor.n_outputs = getattr(model, "n_outputs", 1)`; модели из фабрики выставляют его сами, конвенция задокументирована в докстринге `Predictor`): один forward по родителям вместо `n_gen` forward'ов по детям; выход валидируется по форме `[n_states, n_gen]`
- [x] write tests (success): формы выходов; поколоночная эквивалентность быстрого пути и дефолтного; чекпойнт round-trip; `models_lib_test.test_loads_predictor_models` остаётся зелёным — 12 новых тестов: формы для обеих архитектур при 1 и 3 выходах, совместимость state_dict (точный набор ключей + форма головы `[1, hidden]`), skip-связь (блок с занулёнными весами = тождественная функция), инвариантность к batch size, round-trip чекпойнта для `MLP` и `RESMLP`, поколоночное совпадение быстрого пути с дефолтным (хэмминг-Q-модель против эвристики `"hamming"`, плюс ассерт «модель вызвана 1 раз»), Q-модель из конфига; `models_lib_test.test_loads_predictor_models` зелёный (реальные Kaggle-веса)
- [x] write tests (error/edge): неверный n_outputs vs граф; неизвестный model_type — плюс `n_outputs <= 0`, пустой `layers_sizes` для RESMLP, форма выхода противоречит заявленному `n_outputs`; ➕ **существующий тест `test_score_children_rejects_2d_output` заменён** (`test_score_children_rejects_wrong_number_of_outputs`): он фиксировал ровно то ограничение, которое снимает PR3
- [x] run `./lint.sh && RUN_SLOW_TESTS=1 pytest` — must pass before next task — lint зелёный (black 62, pylint 10.00/10, mypy 62 файла), `black --check .` по всему репо зелёный, `docs/build_docs.sh` (`-W`) зелёный; `RUN_SLOW_TESTS=1 pytest` = **335 passed / 12 skipped / 3 xfailed** и на 3.12 (torch 2.13), и на 3.9 (`.venv39`, torch 2.8)
- [x] открыть PR — [stasdiener/cayleypy#3](https://github.com/stasdiener/cayleypy/pull/3), Draft, base `feat/score-children-contract`, в upstream не отправлялось

### Task 5: PR4 — GroupTokenizer

**Files:**
- Create: `cayleypy/models/tokenizer.py`, `cayleypy/models/tokenizer_test.py`
- Modify: `cayleypy/models/__init__.py`, `docs/api.rst`
- Modify: `cayleypy/__init__.py` (экспорт по конвенции Development Approach) ➕

- [x] ветка `feat/group-tokenizer` от PR1; `GroupTokenizer` по спеке `tokenizer_groups` (для мегаминкса: 20 углов×3 + 30 рёбер×2 → 50 токенов, словарь 60) — семантика: токен i = значение **первого элемента** группы i, отсчитанное от начала своего сегмента (сегмент = одна пара `[group_size, num_groups]`); значит токен кодирует «какая деталь в слоте + её ориентация», `vocab_size = max(group_size*num_groups)` (для мегаминкса 60 = 20×3 = 30×2), `n_tokens = 50`, плюс `token_type_ids` (сегмент каждого токена — модель PR5 сможет отличать углы от рёбер отдельным эмбеддингом); выход всегда int64 (нужен `nn.Embedding`), поддержан и батч `[B, state_size]`, и одиночное состояние; индексные тензоры переносятся на устройство состояний (класс — не `nn.Module`, чтобы не мусорить в state_dict); `from_config(config)` сверяет спеку с `input_size`
- [x] ➕ **вне первоначального охвата: `verify(graph_def)`** — проверка, что кодирование без потерь: у центрального состояния и у **каждого генератора** каждая группа = стикеры одной детали в том же циклическом порядке (индукция: если это верно для генератора-перестановки, свойство сохраняется на всех достижимых состояниях). Проверено по данным репо: мегаминкс `[[3,20],[2,30]]` и `mini_pyramorphix` `[[3,8]]` проходят, `rubik_cube(2,"QTM")` с `[[3,8]]` — нет (стикеры угла не в соседних позициях). Без этой проверки ошибочная спека даёт молча необратимую токенизацию
- [x] write tests (success): корректность на малой головоломке с известной раскладкой — 12 новых тестов: ручная раскладка `[[2,2],[1,3]]` (2 детали по 2 стикера + 3 детали по 1) с литеральными ожидаемыми токенами (включая случай «детали переставлены и повёрнуты»), размеры и токены центрального состояния мегаминкса, батч vs одиночное состояние (+dtype), **losslessness** (40 шагов по генераторам мегаминкса: число уникальных строк токенов == число уникальных состояний), `verify` на двух реальных головоломках, `from_config`
- [x] write tests (error/edge): несогласованная спека (сумма групп ≠ длине состояния) — плюс 0-мерный вход, пустой список групп, пара не из 2 чисел, неположительные размеры, конфиг без `tokenizer_groups`, спека vs `input_size`, `verify` на неверной группировке (сбитый циклический порядок / смешение деталей разных типов / цвета вместо ид стикеров), не-перестановочный граф (`MatrixGroups.heisenberg()`) и несовпадение `state_size`
- [x] run `./lint.sh && RUN_SLOW_TESTS=1 pytest` — must pass before next task — lint зелёный (black 64, pylint 10.00/10, mypy 64 файла), `black --check .` по всему репо зелёный, `docs/build_docs.sh` (`-W`) зелёный; `RUN_SLOW_TESTS=1 pytest` = **335 passed / 12 skipped / 3 xfailed** и на 3.12 (torch 2.13), и на 3.9 (`.venv39`, torch 2.8)
- [x] открыть PR — [stasdiener/cayleypy#4](https://github.com/stasdiener/cayleypy/pull/4), Draft, base `feat/score-children-contract`, в upstream не отправлялось

### Task 6: PR5 — Q-трансформер

**Files:**
- Create: `cayleypy/models/transformer.py`, `cayleypy/models/transformer_test.py`
- Modify: `cayleypy/models/models.py` (регистрация в `_build_model`), `cayleypy/models/__init__.py`, `docs/api.rst`
- Modify: `cayleypy/models/tokenizer.py` (импорт `ModelConfig` только под `TYPE_CHECKING`) ➕
- Create: `docs/plans/notes/20260804-task6-transformer-parity.md` (архитектура донора + статус паритета) ➕

- [x] ветка `feat/q-transformer` от PR4; embedding + learnable pos-encoding + `nn.TransformerEncoder` (SDPA) + голова `n_outputs`; конфиг-пример мегаминкса в докстринге — `TransformerModel` (`model_type="TRANSFORMER"`): эмбеддинг токенов `GroupTokenizer` + learnable позиционный эмбеддинг + эмбеддинг типа токена (углы/рёбра различимы) → `nn.TransformerEncoder` (pre-norm, gelu, dropout 0 → детерминизм в eval) → среднее по токенам → `Linear(d_model, n_outputs)`; ➕ **скоуп: 2 новых поля `ModelConfig`** — `n_heads: Optional[int]=None` (дефолт: одна голова на 64 фичи) и `dim_feedforward: Optional[int]=None` (дефолт: 4× ширина), иначе конфиг Влада (8 голов при d_model=256) невыразим, а чекпойнт обязан полностью описывать модель; `layers_sizes` = по записи на слой энкодера, все равны (у энкодера одна ширина во всех слоях), `num_classes_for_one_hot` не используется (словарь задаёт токенизатор); ➕ `tokenizer.py` переведён на `TYPE_CHECKING`-импорт `ModelConfig` — иначе `models.py → transformer.py → tokenizer.py → models.py` циклический импорт (pylint `cyclic-import`)
- [x] в описании PR: «веса появятся в PR11 (демонстратор)» — не мёртвый код — абзац «Weights» в описании [#5](https://github.com/stasdiener/cayleypy/pull/5)
- [x] write tests (success): форма выхода; чекпойнт round-trip; инвариантность к batch size — 8 тестов: формы для 1 выхода и Q-варианта (батч и одиночное состояние), конфиг мегаминкса (50 токенов, словарь 60, 2 типа токенов, число слоёв), дефолты голов/FF, независимость выхода от размера батча (полный батч vs по одному), round-trip чекпойнта (включая `n_heads`/`dim_feedforward` в конфиге и `graph_hash`), работа как `Predictor` (`__call__` + дефолтный `score_children`)
- [x] write tests (error/edge): вызов без `tokenizer_groups` → понятная ошибка — плюс спека групп vs `input_size`, пустой `layers_sizes`, слои разной ширины, ширина не делится на число голов, неположительные `n_heads`/`dim_feedforward`/`n_outputs`
- [x] ➕ (вне CI) скрипт паритета с Kaggle-весами Влада — публичные веса, по прецеденту `models_lib_test` без skipif; пометить slow — **не сделан, заблокирован внешними данными** (перенесён в Post-Completion → «Миграция весов»): в репозитории донора нет ни одного упоминания Kaggle (проверены `README.md`, `DESIGN.md`, `PROVENANCE.md`, `models.py`, `config.py`, `cli.py`, `configs/`) — id весов неизвестен; и его `PieceTransformer` параметризован иначе (эмбеддинг **каждого** стикера детали + `Linear`-проекция, CLS-пулинг, silu, самописные блоки), поэтому его state_dict не ложится на нашу модель без конверсии, согласованной с автором. Архитектура донора и план конверсии зафиксированы в `docs/plans/notes/20260804-task6-transformer-parity.md`
- [x] run `./lint.sh && RUN_SLOW_TESTS=1 pytest` — must pass before next task — lint зелёный (black 66, pylint 10.00/10 без сообщений, mypy 66 файлов), `black --check .` по всему репо зелёный, `docs/build_docs.sh` (`-W`) зелёный, докстринг-примеры проходят `pytest --doctest-modules`; `RUN_SLOW_TESTS=1 pytest` = **350 passed / 12 skipped / 3 xfailed** и на 3.12 (torch 2.13), и на 3.9 (`.venv39`, torch 2.8)
- [x] открыть PR — [stasdiener/cayleypy#5](https://github.com/stasdiener/cayleypy/pull/5), Draft, base `feat/group-tokenizer`, в upstream не отправлялось

### Task 7: PR6 — AZ-модель (q+v) и v-consistency

**Files:**
- Create: `cayleypy/models/qv_model.py`, `cayleypy/models/qv_model_test.py`
- Modify: `cayleypy/models/models.py` (регистрация), `cayleypy/models/__init__.py`, `docs/api.rst`
- Modify: `cayleypy/predictor.py` (делегирование `score_children` модели), `cayleypy/models/models_test.py` (новые поля конфига) ➕

- [x] ветка `feat/az-heads` от PR3; `QVModel`: бэкбон из фабрики + q-head (`n_gen`) + v-head (1) — `model_type="QV"`; обе головы считает **выходной слой бэкбона** (первые `n_outputs` значений = Q, последнее = V): это математически то же, что две отдельные линейные головы над фичами бэкбона, но без лишнего `Linear(w, w)` без нелинейности, который получился бы при «бэкбон как экстрактор фич + 2 головы»; `forward` возвращает Q (согласовано с конвенцией PR3 «`n_outputs` описывает форму `forward`»), V доступна через `v()`, обе — через `heads()`
- [x] ➕ **скоуп: бэкбон описан плоским полем `backbone_type: Optional[str]`, а не вложенным конфигом** (в чекбоксе тестов ниже был «вложенный конфиг бэкбона»): по решению #3 плана поля конфига плоские и аддитивные, а остальные поля конфига и так описывают бэкбон — так нет дублирования `input_size`/`num_classes_for_one_hot`, нет фиктивного `layers_sizes` у внешнего конфига и не нужно руками писать `n_gen+1`; чекпойнт по-прежнему полностью описывает модель. Второе новое поле — `v_consistency_weight: float = 0.0`. Бэкбон строится `replace(config, model_type=backbone_type, n_outputs=n_outputs+1).build_model()`, т.е. годится любая зарегистрированная архитектура (MLP/RESMLP сейчас, TRANSFORMER из PR5 после мержа)
- [x] v-consistency **внутри `QVModel.score_children`** (inference-time rescoring, не расширение API `Predictor`): штраф `weight * |Q_child − (V_parent − 1)|` — при `weight=0` (дефолт) скоры = Q как есть; ➕ **чтобы штраф доезжал до луча**, `Predictor.score_children` теперь делегирует модели, у которой есть свой `score_children` (проверка `getattr`, годится любая такая модель), с тем же батчингом, что и обычные предсказания (батчинг вынесен в `Predictor._apply_batched`, `predict_batched` = обёртка над ним); API `Predictor` не расширялся
- [x] write tests (success): формы голов; на ручном примере штраф понижает ранг ребёнка, чей Q противоречит V−1; чекпойнт с вложенным конфигом бэкбона — 17 тестов в `qv_model_test.py`: формы Q/V (батч и одиночное состояние, бэкбоны MLP и RESMLP), `forward == q`, `n_outputs` модели и `n_outputs+1` бэкбона, штраф на ручном примере (Q=[3,2], V=4 ⇒ argmin меняется с 1 на 0) и его пропорциональность весу, штраф выключен по умолчанию, round-trip чекпойнта (**конфиг с `backbone_type`/`v_consistency_weight`/`graph_hash` вместо вложенного конфига — см. пункт про скоуп**; скоры после загрузки идентичны), все веса под `backbone.*`, `Predictor` применяет `score_children` модели и батчит его; плюс 2 новых ассерта в `models_test.py` (новые поля в `from_dict`, дефолты для legacy-словаря)
- [x] write tests (error/edge): weight<0; бэкбон-конфиг неизвестного типа — плюс `backbone_type` не задан, `backbone_type="QV"` (сам себе бэкбон → рекурсия), `n_outputs <= 0`, `n_outputs` не совпадает с числом генераторов графа
- [x] run `./lint.sh && RUN_SLOW_TESTS=1 pytest` — must pass before next task — lint зелёный (black 64, pylint 10.00/10, mypy 64 файла), `black --check .` по всему репо зелёный, `docs/build_docs.sh` (`-W`) зелёный, докстринг-пример проходит `pytest --doctest-modules`; `RUN_SLOW_TESTS=1 pytest` = **352 passed / 12 skipped / 3 xfailed** и на 3.12 (torch 2.13), и на 3.9 (`.venv39`, torch 2.8)
- [x] открыть PR — [stasdiener/cayleypy#6](https://github.com/stasdiener/cayleypy/pull/6), Draft, base `feat/resmlp-qmlp`, в upstream не отправлялось

### Task 8: PR7 — EnsemblePredictor (плоский)

**Files:**
- Create: `cayleypy/ensemble.py`, `cayleypy/ensemble_test.py`
- Modify: `cayleypy/__init__.py`, `docs/api.rst`

- [x] ветка `feat/ensemble-predictor` от PR1; `EnsemblePredictor(members: list, weights: list)` — взвешенная сумма `score_children`; **без вложенных ансамблей** — класс наследует `Predictor` (значит, его можно передать в `beam_search` как есть) и ансамблирует **оба** метода скоринга: `__call__` (скоры самих состояний) и `score_children` (скоры детей — спрашивая у каждого члена его собственный `score_children`, поэтому члены с быстрым однопроходным `score_children` (Q-модели) продолжают им пользоваться); ➕ **веса опциональны**: дефолт `1/n` (среднее членов), заданные веса используются как есть и **не нормируются** (задокументировано); валидация — непустой список, число весов, члены суть `Predictor`-ы для одного графа (совпадают `n_generators` и `state_size`)
- [x] write tests (success): сумма 0.75/0.25 против ручного расчёта — 13 тестов в `ensemble_test.py`: литеральные ожидаемые числа для 0.75/0.25, поколоночная проверка ансамблированного `score_children`, ансамбль из одного члена == сам член, дефолтные веса дают среднее, делегирование в `score_children` члена (с ассертом «вызван ровно 1 раз»), батчинг обоих методов при состояниях, не влезающих в один батч, ансамбль как предиктор в beam search на `lrx(8)` (путь найден и проверен применением к старту)
- [x] write tests (error/edge): пустой список; разный `n_gen` у членов; веса не нормируются — плюс неверное число весов, разный `state_size` у членов, член не `Predictor` (понятная ошибка «оберните в Predictor»); «веса не нормируются» — отдельный тест: `[2.0, 3.0]` даёт `2a+3b`, а не `(2a+3b)/5`
- [x] run `./lint.sh && RUN_SLOW_TESTS=1 pytest` — must pass before next task — lint зелёный (black 64, pylint 10.00/10, mypy 64 файла), `black --check .` по всему репо зелёный, `docs/build_docs.sh` (`-W`) зелёный, докстринг-пример проходит `pytest --doctest-modules`; `RUN_SLOW_TESTS=1 pytest` = **336 passed / 12 skipped / 3 xfailed** и на 3.12 (torch 2.13), и на 3.9 (`.venv39`, torch 2.8)
- [x] открыть PR — [stasdiener/cayleypy#7](https://github.com/stasdiener/cayleypy/pull/7), Draft, base `feat/score-children-contract`, в upstream не отправлялось

### Task 9: PR8 — лоссы (независимый, можно параллельно с PR1)

**Files:**
- Create: `cayleypy/train/__init__.py`, `cayleypy/train/losses.py`, `cayleypy/train/losses_test.py`
- Modify: `cayleypy/__init__.py`, `docs/api.rst`

- [x] ветка `feat/train-losses` от `main`; MSE, pinball(τ), masked-sparse (маска неразмеченных выходов — фундамент PR10) — `Loss` (ABC: подклассы реализуют только `elementwise`, общий `__call__` делает валидацию и редукцию), `MseLoss`, `PinballLoss(tau)` (τ-квантиль: при τ<0.5 предсказания смещаются к нижней оценке, при τ=0.5 = ½·MAE), `make_loss(name, tau)` для конфига тренера; ➕ **скоуп: masked-sparse — не третий класс лосса, а параметр `mask` у обоих** (маскирование ортогонально поэлементной формуле: «masked sparse MSE» = `MseLoss()(pred, targets, mask=mask)`); ➕ **добавлен `weights`** (взвешенное среднее) — PR10 требует «верхние границы взвешивать», а `losses.py` в списке файлов PR10 нет, значит поддержка нужна здесь; лоссы **не** `nn.Module` (нет обучаемых параметров — не мусорить в state_dict, как `GroupTokenizer` в PR4); пустой mask → 0 и нулевой (не NaN) градиент через `clamp_min` знаменателя
- [x] write tests (success): формулы на синтетике (pinball при τ=0.5 = 0.5·MAE и т.п.); маска не пропускает градиент — 26 тестов (включая 3 doctest'а): формулы против ручного счёта и против `torch.nn.functional` (`mse_loss`, `0.5·l1_loss`), асимметрия при τ=0.9/0.1 с литеральными числами, минимум pinball ровно в квантиле целей, маска игнорирует неразмеченные элементы даже при целях `1e9`, градиент строго нулевой в маскированных позициях, полностью маскированный батч → 0 и нулевой градиент, bool- и числовая маски совпадают, веса (ручные числа, отсутствие нормировки, комбинация с маской), Q-форма `[B, n_gen]`, целочисленные цели (BFS), тренировочная санити с сидом (final loss < 20% initial для всех трёх лоссов; маскированная тренировка двигает только размеченный выход)
- [x] write tests (error/edge): τ вне (0,1); маска несовместимой формы — τ = 0, 1, −0.5, 1.5, NaN; неизвестное имя в `make_loss`; несовпадение формы targets/mask/weights с predictions (в т.ч. то же число элементов при другой форме — там бы бродкаст молча дал неверный ответ)
- [x] run `./lint.sh && RUN_SLOW_TESTS=1 pytest` — must pass before next task — lint зелёный (black 62, pylint 10.00/10, mypy 62 файла), `black --check .` по всему репо зелёный, `docs/build_docs.sh` (`-W`) зелёный; `RUN_SLOW_TESTS=1 pytest` = **322 passed / 12 skipped / 3 xfailed** и на 3.12 (torch 2.13), и на 3.9 (`.venv39`, torch 2.8)
- [x] открыть PR — [stasdiener/cayleypy#8](https://github.com/stasdiener/cayleypy/pull/8), base `main` форка (PR независим от PR1 — не Draft), в upstream не отправлялось

### Task 10: PR9 — тренер-ядро

**Files:**
- Create: `cayleypy/train/config.py`, `cayleypy/train/trainer.py`, `cayleypy/train/trainer_test.py`
- Modify: `cayleypy/train/__init__.py`, `docs/api.rst`

- [x] ветка `feat/trainer-core` от PR8 (+ использует checkpoint из PR1); `TrainConfig` (frozen, 3.9-синтаксис); `Trainer`: данные из `algo/random_walks.py` (nbt-волки), AdamW + CosineAnnealingLR, EMA-копия, сохранение через `checkpoint.py`, прогресс — `verbose` + `print` — `TrainConfig` (frozen, валидация всех полей в `__post_init__`): `n_epochs`, `n_walks`, `rw_length`, `rw_mode` (дефолт `"nbt"`), `nbt_history_depth`, `batch_size`, `lr`/`lr_min`/`weight_decay`, `ema_decay`, `loss`/`tau`, `seed`, `verbose`; `Trainer`: эпоха = свежие random walks → один проход батчами (данные не переиспользуются, поэтому лосс по эпохам читается как валидационная кривая), AdamW + `CosineAnnealingLR` (`lr`→`lr_min` за `n_epochs`), EMA-копия весов, `save`/`from_checkpoint` через `checkpoint.py` (чекпойнт хранит `graph_hash` → загрузка под чужой граф падает), прогресс — `print` при `verbose >= 1`; ➕ **скоуп: PR9 зависит от двух неслитых веток (PR8 + PR1)**, поэтому создана механическая merge-база `base/trainer-core` — она же base PR, чтобы диф показывал только тренер; ➕ публичные `train_step(states, targets, mask, weights)` и `train_on_data` — точки входа для PR10 (свои источники данных, sparse-маски) и PR16 (Bellman-батчи); `predictor()`/`save()`/`model_for_inference()` по умолчанию берут EMA-копию (она же — frozen target для PR16); модель с `n_outputs != 1` отклоняется с понятной ошибкой (у rw-данных нет таргета на каждый выход) — снимется в PR10
- [x] в описании PR: ссылка на #151, кредит vlzm, отличия (контракт PR1, лоссы PR8, EMA, ноль новых зависимостей — #151 тянул deps в pyproject) — раздел «Relation to upstream cayleypy/cayleypy#151» в описании [#9](https://github.com/stasdiener/cayleypy/pull/9) (ссылка полным URL, иначе `#151` в форке ведёт в никуда), плюс отличие «живёт в `cayleypy/train/`, не конфликтует с `cayleypy/trainers/` из #151»
- [x] write tests (success, RUN_SLOW_TESTS): `torch.manual_seed`, тренировка на `lrx(5)`: final loss < 20% initial; beam с обученной моделью решает все состояния (slow tier) — 2 slow-теста на `lrx(5)` (seed=42): (1) с **точными** метками (`rw_mode="bfs"` на таком графе обходит весь граф) final loss < 20% initial, MAE против точного BFS < 0.5 и beam находит **оптимальный** путь для всех 120 состояний при **beam_width=1** (хэмминг решает 10); (2) на nbt-волках модель решает **все 120 состояний при beam_width=5** (хэмминг — 23), лосс падает до ~40% initial и упирается в шум меток — поэтому строгий порог 20% проверяется на точных метках, а nbt-путь проверяется полезностью в луче
- [x] write tests (error/edge, быстрые): чекпойнт-резюме; EMA-обновление на 2 шагах; несовместимый конфиг — 20 тестов всего: резюме из чекпойнта (веса совпадают с сохранёнными, обучение продолжается), EMA против формулы `decay*ema + (1-decay)*w` на двух шагах подряд, EMA отключена (`ema_decay=0`), все невалидные поля `TrainConfig` по отдельности, модель под состояния другого размера, многовыходная модель, слишком мало one-hot классов, пустые данные, чекпойнт от другого графа (`lrx(5, k=2)`), плюс детерминизм по сиду, полностью маскированный батч не меняет веса, косинусный скедулер приходит в `lr_min`, `verbose` печатает прогресс
- [x] run `./lint.sh && RUN_SLOW_TESTS=1 pytest` — must pass before next task — lint зелёный (black 68, pylint 10.00/10, mypy 68 файлов), `black --check .` по всему репо зелёный, `docs/build_docs.sh` (`-W`) зелёный, докстринг-пример проходит `pytest --doctest-modules`; `RUN_SLOW_TESTS=1 pytest` = **366 passed / 12 skipped / 3 xfailed** и на 3.12 (torch 2.13), и на 3.9 (`.venv39`, torch 2.8)
- [x] открыть PR — [stasdiener/cayleypy#9](https://github.com/stasdiener/cayleypy/pull/9), Draft, base `base/trainer-core` (мерж `feat/train-losses` + `feat/score-children-contract`), в upstream не отправлялось

### Task 11: PR10 — источники данных: BFSAnchors + SparseQSampler

**Files:**
- Create: `cayleypy/train/data.py`, `cayleypy/train/data_test.py`
- Modify: `cayleypy/train/trainer.py`, `cayleypy/train/config.py`
- Modify: `cayleypy/train/trainer_test.py`, `cayleypy/train/__init__.py`, `cayleypy/__init__.py`, `docs/api.rst` ➕

- [x] ветка `feat/train-data-sources` от PR9; `BFSAnchors(graph, depth)`: семпл глубины ≤ d−1 → все `n_gen` детей в таблице → плотные точные метки; `SparseQSampler`: метки prev(p−1)/next(p+1), остальное маскируется — назван `BfsAnchors` (конвенция репо: `BfsResult`/`BfsAlgorithm`): BFS от центра гоняется **один раз в конструкторе**, `generate()` только семплит (`size` штук с возвращением, None = вся таблица); таблица хэш→точная дистанция + `searchsorted`; для Q-режима годятся состояния глубины ≤ max_depth−1, а если BFS обошёл весь граф (`bfs_completed`) — **все** состояния таблицы; ➕ **скоуп: `max_table_states` (дефолт 10⁷)** — BFS останавливается через `stop_condition`, и конструктор падает с понятной ошибкой (это и есть depth-guard по памяти). `SparseQSampler`: ходы находятся сравнением хэшей детей с prev/next состоянием прогулки (не нужны id генераторов, и если несколько генераторов дают то же состояние — размечаются все); при совпадении prev и next берётся меньшая метка; волки — **только "classic"** (единственный режим, чей выход — путь: nbt/bfs перемешивают состояния между шагами), поэтому `rw_mode` для Q-моделей игнорируется (задокументировано)
- [x] `PathDataSource`: пути решений (найденные beam search / загруженные, формат совместим с `cayleypy-beam-results` TSV) → hindsight-метки остаточной длины для всех состояний пути (верхние границы — флаг в семпле, чтобы лосс мог их взвешивать) — внутреннее представление — `list[CayleyPath]`; `weight` (дефолт 1.0) едет в `TrainingData.weights` — это и есть «флаг верхней границы» для лосса из PR8; для Q-моделей размечается ровно выход сделанного ходa (`edges[p]` → `L−p−1`), остальное маскируется, а центральное состояние из данных выпадает (из него ход не делается). ➕ `from_tsv(path, graph, ...)`: реальная схема из [TryDotAtwo/cayleypy-beam-results](https://github.com/TryDotAtwo/cayleypy-beam-results) проверена по репозиторию — широкая TSV с заголовком, колонка `solution` = имена генераторов через точку (формат `path_to_string`), значение с ведущим апострофом (spreadsheet-safe); **решённое состояние в файле не хранится**, поэтому решение проигрывается назад от центрального состояния через `generators_inverse_map` (не inverse-closed → понятная ошибка); фильтр `puzzle_id`, остальные колонки игнорируются
- [x] пропорции смеси в `TrainConfig` (дефолт anchors 1–2% — 10% вредит, эмпирика Стаса) — `anchors_depth: int = 0` (0 = без anchors) + `anchors_fraction: float = 0.02`, в докстринге зафиксирована эмпирика («несколько процентов помогают, 10%+ вредят»); ➕ `MixtureDataSource(sources, fractions)`: размеры выбираются максимальными, при которых пропорции держатся **без повторов** состояний (`total = min_i len(data_i)/fraction_i`), и ни один непустой источник не выбрасывается округлением; трейнер выставляет anchors `size = f/(1−f) × n_walks·rw_length`, поэтому смесь почти ничего не отбрасывает
- [x] ➕ **скоуп: `TrainingData(states, targets, mask, weights)`** — контракт «данные ↔ трейнер» (frozen dataclass, `select`/`concat`/`n_outputs`); `Trainer.generate_data()` теперь возвращает его, `train_on_data(data)` принимает его (маски и веса доезжают до лосса) — сигнатуры PR9 обновлены вместе с его тестами; `Trainer(..., data_source=...)` — точка входа для `PathDataSource` и Bellman-батчей PR16; **снято ограничение PR9 «только n_outputs == 1»**: допускается 1 или `n_generators`, источник по умолчанию выбирается по этому числу (Q → `SparseQSampler` + anchors на все выходы)
- [x] write tests (success): метки anchors == точный BFS на `lrx(5)`; ровно 2 размеченных выхода у sparse; распределение смеси в батче; hindsight-метки пути == длина хвоста пути — 31 тест в `data_test.py` (+5 доктестов) и 5 новых в `trainer_test.py`: anchors против полного BFS (скалярно и **поколоночно** для Q — ловит транспонирование), метки sparse проверяются ход за ходом против самой прогулки (плюс отдельный тест «1 размеченный выход на концах прогулки, 2 внутри»), пропорции смеси точны (900/100), hindsight-метки == хвост пути + восстановление решённого состояния из TSV-решения, `puzzle_id`-фильтр, смесь sparse+anchors (10 полностью размеченных строк против 40 разреженных), Q-модель обучается на sparse-метках и на смеси с anchors, `data_source=` переопределяет конфиг
- [x] write tests (error/edge): depth-guard по памяти; глубина 0; граф без inverse-closed генераторов для nbt-волков — `max_table_states=5` → ошибка с числом найденных состояний; `depth=0`; `lx(5)` (не inverse-closed) → ошибка и в `SparseQSampler`, и в `from_tsv`; плюс неверный `n_outputs` (не 1 и не `n_gen`), `size<=0`, `rw_length<2`, путь не заканчивается в центральном состоянии, генератор вне диапазона, `weight<=0`, неизвестное имя хода в TSV, отсутствующая колонка, пустой файл/пустой фильтр, все несогласованные формы `TrainingData`, `concat` разных `n_outputs`, невалидные `fractions`, новые поля `TrainConfig`
- [x] интеграционный slow-тест: тренировка с anchors даёт точные предсказания на глубинах ≤ d — `test_anchors_make_predictions_near_central_state_exact` на `lrx(5)` (seed=42, depth=3): MAE на состояниях глубины ≤ 3 = 0.24 (< 0.5) и **в 14 раз** меньше, чем без anchors (3.49; ассерт — «> 3×»). Доля anchors в тесте намеренно завышена (0.9): при 1–2% на бюджете теста эффект не отделим от шума, а при 0.9 виден и его минус — ошибка по всему графу не улучшается (2.09 против 2.56), что и есть эмпирика «10% вредит»
- [x] run `./lint.sh && RUN_SLOW_TESTS=1 pytest` — must pass before next task — lint зелёный (black 72 файла, pylint 10.00/10, mypy 70 файлов), `black --check .` по всему репо зелёный, `docs/build_docs.sh` (`-W`) зелёный, доктесты `cayleypy/train/` зелёные; `RUN_SLOW_TESTS=1 pytest` = **402 passed / 12 skipped / 3 xfailed** и на 3.12 (torch 2.13), и на 3.9 (`.venv39`, torch 2.8)
- [x] открыть PR — [stasdiener/cayleypy#10](https://github.com/stasdiener/cayleypy/pull/10), Draft, base `feat/trainer-core`, в upstream не отправлялось

### Task 12: PR11 — демонстрационный чекпойнт в PREDICTOR_MODELS

**Files:**
- Modify: `cayleypy/models/models_lib.py`, `cayleypy/models/models_lib_test.py`
- Modify: `README.md` (форк; при мерже — upstream)
- Modify: `cayleypy/models/models.py`, `cayleypy/models/models_test.py` (`ModelConfig.load` — форматы весов + обёртка ошибок Kaggle) ➕
- Modify: `cayleypy/predictor.py` (сверка `graph_hash` в `Predictor.pretrained`) ➕
- Create: `docs/plans/notes/20260804-task12-demo-checkpoint.md` (рецепт, замеры, пределы рецепта) ➕

- [x] обучить тренером (PR9+PR10) небольшую Q-модель (ResMLP или трансформер) на графе типа `lrx(N)`/малой головоломке; проверить по гайду upstream: «reliably finds the paths» в beam search — **`lrx-14`, `RESMLP [512,512,512]`, `n_outputs=3`** (631k параметров), 200 эпох × 1024 classic-волка длины 92 + 2% BfsAnchors (depth 6), EMA 0.999, seed 42 — 167 с на CPU, лосс 902 → 248. Критерий upstream проверен на **50 равномерно случайных перестановках** (`torch.randperm`, seed 11) при луче 1000: **50/50 решено**, средняя длина пути 61.2, макс 86 (диаметр графа 91); хэмминг на тех же 50 состояниях — **0/50**. При луче 100 — 17/20. ➕ **скоуп: понадобилась merge-база `base/demo-checkpoint`** (мерж PR10 + PR3 + PR2): Q-модель невозможно ни построить (многовыходная архитектура — PR3), ни применить в луче (`use_child_scores` — PR2) ни в одной ветке по отдельности. ⚠️ **`lrx-20` намеренно НЕ выбран**: тот же рецепт с бюджетом ×4 (400 эпох × 2048 волков, 1463 с) даёт лишь 8/10 при луче 3000–10000, причём лосс встаёт на плато к 5-й эпохе — предел разметки (индекс шага classic-волка), а не бюджета; это ровно задача PR16 (Bellman) и `PathDataSource`. Цифры и диагноз — в заметках
- [x] выгрузить веса на **свой** Kaggle-аккаунт; добавить запись в `PREDICTOR_MODELS` — [rokham/lrx-14-q](https://www.kaggle.com/models/rokham/lrx-14-q) (`pyTorch/resmlp-512x3`, MIT, публичная; `kagglehub.model_upload` создаёт приватную — публичность выставлена `ApiUpdateModelRequest` + `FieldMask`), файл `lrx_14_q_resmlp.pt` = самоописывающий чекпойнт PR1 (веса + `ModelConfig` + `graph_hash`); анонимное скачивание проверено без `KAGGLE_*` и без `~/.kaggle` → CI не нужны креды. Запись `"lrx-14"` в `PREDICTOR_MODELS` с `n_outputs=3` и `graph_hash`; `prepare_graph("lrx-14")` уже работает (новых графов не добавлялось — правило README)
- [x] сослаться на этот PR из описаний PR3/PR5/PR6 («веса здесь») — закрыть вопрос «мёртвого кода» — описания [#3](https://github.com/stasdiener/cayleypy/pull/3), [#5](https://github.com/stasdiener/cayleypy/pull/5), [#6](https://github.com/stasdiener/cayleypy/pull/6) обновлены ссылкой на #11 с цифрами 50/50 против 0/50; для #5 и #6 формулировка честная: демо — `RESMLP`, поэтому веса **для трансформера** (нужна головоломка с токенайзер-спекой или конверсия весов Влада) и **для `QV`** (прогон с `v_consistency_weight > 0`) остаются в серии, а #11 доказывает работоспособность всего пути «обучение → чекпойнт → реестр → луч по детям»
- [x] write tests (success): загрузка новой записи в `models_lib_test` (по прецеденту — публичные веса без skipif) — `test_loads_predictor_models` расширен ветвью для моделей с выходом на генератор (проверяется форма `score_children`), реальные веса с Kaggle, без skipif; плюс slow-тест `test_lrx_14_model_finds_paths` (10 случайных состояний, луч 1000: путь найден, воспроизведён `apply_path` до центрального состояния, длина ≤ диаметра); плюс в `models_test.py` — `ModelConfig.load` из обоих форматов весов даёт идентичные предсказания
- [x] write tests (error/edge): понятная ошибка при недоступности kagglehub (обёртка, если её нет) — ➕ обёртка добавлена (`RuntimeError` с именем модели и подсказкой про сеть/креды; типы ошибок перечислены списком, т.к. `broad-exception-caught` в этом репо не отключён), тест через monkeypatch `kagglehub.model_download`; плюс `Predictor.pretrained` без модели для графа и ➕ **сверка `graph_hash`**: модели ищутся по имени графа, а Q-модели достаточно переставленных генераторов, чтобы её выходы стали бессмысленными
- [x] run `./lint.sh && RUN_SLOW_TESTS=1 pytest` — must pass before next task — lint зелёный (black 70 файлов, pylint 10.00/10, mypy 70 файлов), `black --check .` по всему репо зелёный (72 файла), `docs/build_docs.sh` (`-W`) зелёный, доктесты `cayleypy/models` и `cayleypy/train` зелёные; `RUN_SLOW_TESTS=1 pytest` = **426 passed / 12 skipped / 3 xfailed** и на 3.12 (torch 2.13), и на 3.9 (`.venv39`, torch 2.8)
- [x] открыть PR — [stasdiener/cayleypy#11](https://github.com/stasdiener/cayleypy/pull/11), Draft, base `base/demo-checkpoint`, в upstream не отправлялось

### Task 13: PR12 — SymmetryGroup + TTA

**Files:**
- Create: `cayleypy/symmetries.py`, `cayleypy/symmetries_test.py`
- Modify: `cayleypy/puzzles/moves.py` (захардкоженные списки симметрий — конвенция README)
- Modify: `cayleypy/__init__.py`, `docs/api.rst`

- [x] ветка `feat/symmetry-group` от PR1; `SymmetryGroup(symmetries, graph_def)`: `apply`, `transport_actions` (sigma_inv), `verify()` (сопряжение генератора симметрией — снова генератор, индексы согласованы) — симметрия = перестановка позиций `sigma`, которая сохраняет центральное состояние и сопряжением переводит каждый генератор в генератор; действие на состояния `T(s)[i] = element_map[s[sigma[i]]]`, где **`element_map` (переобозначение элементов) выводится**, а не задаётся: из условия сохранения центрального состояния `element_map[central_state[sigma[i]]] = central_state[i]` он определён однозначно. Это даёт одно определение на оба случая: для графа с центральным состоянием-перестановкой (`lrx-n`) `element_map == sigma_inv` (то есть `T` — сопряжение), а для головоломок с цветами — перестановку цветов (поворот всего куба 2×2×2 переставляет его 6 граней). ➕ `transport_scores(child_scores, i)` — перенос скоров детей образа обратно в порядок генераторов исходных состояний (то, что нужно TTA и PR13); ➕ в `verify()` добавлена проверка **групповости** (есть тождественная, замкнуто по композиции) — она нужна PR13: канон-форма не должна зависеть от того, с какого элемента орбиты стартуем
- [x] **источник симметрий**: захардкодить списки для 2×2×2 (+ LRX-отражение) в `puzzles/moves.py`; ➕ опциональный helper-деривация перебором для малых групп (под RUN_SLOW_TESTS) — `CUBE_222_ROTATIONS` в `puzzles/moves.py` (24 поворота всего куба как перестановки 24 стикеров; выведены как произведения ходов параллельных слоёв, например «f0, затем f1» = поворот вокруг оси f) + конструктор `SymmetryGroup.rubik_cube_rotations`; отражения — `SymmetryGroup.reflections` (перебирает `i -> (m-i) mod n` и берёт те, что являются симметриями, затем замыкает по композиции), что автоматически покрывает и `lrx(n, k)` при `k != 1` (там отражение `i -> (k-i) mod n`); `SymmetryGroup.derive(graph_def, max_state_size=8)` — полный перебор `state_size!` (для `lrx(5)` находит ровно {тождественная, отражение}). Все 24 поворота проверены `verify()` на метриках QTM/QSTM/HTM/ATM
- [x] TTA-обёртка `SymmetrizedPredictor(base, sym_group)`: среднее `score_children` по образам с обратным транспортом индексов — подкласс `Predictor`, усредняет и `score_children` (с `transport_scores` перед усреднением), и скалярный путь `__call__`/`predict`; проверяет, что группа построена для того же графа (те же генераторы **в том же порядке** и то же центральное состояние — иначе транспорт индексов молча неверен)
- [x] write tests (success): `verify()` на 2×2×2; транспорт индексов против точного BFS; TTA не меняет предсказания симметричной модели — 30 тестов: `verify()` на 2×2×2 во всех 4 метриках; **против точного BFS** на `lrx(6)` (полный BFS, 720 состояний) — каждая симметрия сохраняет дистанцию каждого состояния и `dist(s*g_i) == dist(T(s)*g_j)` для всех состояний/генераторов/симметрий; **на реальном кубе** — образы каждого слоя BFS (глубина ≤ 4) под всеми 24 поворотами совпадают с самим слоем (по хэшам), плюс тождество транспорта `T(s*g_i) == T(s)*g_j` на случайных состояниях; TTA от хэмминга (симметричен) не меняет ни `__call__`, ни `score_children`; TTA от асимметричной модели = ручное среднее; предсказания симметризованного предиктора равны по всей орбите; для «точный BFS + асимметричная ошибка» TTA снижает MAE больше чем на 20%; симметризованный предиктор работает как `predictor=` в beam search
- [x] write tests (error/edge): не-симметрия (не сохраняет набор генераторов) ловится `verify()` — плюс перестановка, не сохраняющая центральное состояние coset-графа; отсутствие тождественной и незамкнутость по композиции; повороты куба для 3×3×3 и для не-куба того же размера состояния; `reflections` на `lx(5)` (там нет отражения-симметрии); матричные графы, не-перестановки, дубликаты, неверный `symmetry_id`, неверная форма состояний и скоров детей
- [x] run `./lint.sh && RUN_SLOW_TESTS=1 pytest` — must pass before next task — lint зелёный (black 64 файла, pylint 10.00/10, mypy 64 файла), `black --check .` по всему репо зелёный, `docs/build_docs.sh` (`-W`) зелёный; `RUN_SLOW_TESTS=1 pytest` = **353 passed / 12 skipped / 3 xfailed** и на 3.12 (torch 2.13), и на 3.9 (`.venv39`, torch 2.8). Под `RUN_SLOW_TESTS` — только `derive` для состояний размера 8 (`lrx(8)` и `cyclic_coxeter(8)`, где находятся все 16 симметрий цикла); `derive` для `lrx(5)` быстрый и остался в быстром слое
- [x] открыть PR — [stasdiener/cayleypy#12](https://github.com/stasdiener/cayleypy/pull/12), Draft, base `feat/score-children-contract`, в upstream не отправлялось

### Task 14: PR13 — канонический дедуп в луче

**Files:**
- Modify: `cayleypy/algo/beam_search.py`, `cayleypy/algo/beam_search_test.py`
- Modify: `cayleypy/symmetries.py` (метод `canonical`)

- [x] ветка `feat/canonical-dedup` **от ветки PR2** (нужен провенанс/раскладка) + использует PR12; `canonical(states)`: лексикографический минимум орбиты (батчево, прямой перебор — группы симметрий малы) — ➕ **скоуп: понадобилась merge-база `base/canonical-dedup`** (мерж `feat/child-scored-beam` + `feat/symmetry-group`), как в PR9/PR11: нужны и провенанс луча из PR2, и `SymmetryGroup` из PR12, а они в разных ветках. `SymmetryGroup.canonical` материализует все образы и сводит их поэлементным лексикографическим минимумом (`_is_lexicographically_smaller` — первая различающаяся колонка через `cumsum`, без опоры на неопределённый тай-брейк `argmax`); 1-D и 2-D вход, как у `apply`
- [x] опция `canonical_dedup: Optional[SymmetryGroup]` — дедуп фронтира по хэшу канонической формы (через `hasher.py`), прокинуть через `search()` — только `search_simple` (как `use_child_scores` из PR2; в "advanced" — понятная ошибка); дедуп идёт **после** проверки «центральное состояние достигнуто» (решение не может быть потеряно до того, как его заметили) и **до** скоринга/top-k (в этом и смысл: луч держит `beam_width` орбит, а не копий одной позиции); при `use_child_scores=True` вместе с состояниями фильтруются `moves`/`source_index`, иначе скоры детей разъехались бы со своими состояниями; ➕ проверка «группа построена для этого графа» (как в `SymmetrizedPredictor`); ➕ рефакторинг: «первое состояние с каждым хэшем» вынесено в `_unique_index` (общее с `_expand_layer`)
- [x] write tests (success): фронтир сжимается на состояниях-орбитах (малый граф, ручной подсчёт); решение не теряется — 5 тестов на `canonical` в `symmetries_test.py` (против ручного перебора на кубе, одинаковость по орбите и «канон-форма — один из образов» на всех 720 состояниях `lrx(6)`, число канон-форм == числу орбит: 64 из 120 состояний `lrx(5)` — посчитано руками через централизатор отражения, ручной пример на одиночном состоянии и батче) + 6 в `beam_search_test.py`: ручной пример «5 состояний = 3 орбиты», сжатие целого слоя BFS куба 2×2×2 (6539 → 294, т.е. ×22 из максимальных ×24), **луч находит путь там, где обычный не находит** (`lrx(6)`, луч 200: в обычном луче ~половина состояний эквивалентна другим, он обрезается и путь теряется; с дедупом — путь длины 13, стабильно на 20 прогонах), «в луче нет эквивалентных состояний» (`lrx(8)`, луч 300: с дедупом 100% попарно неэквивалентных, без — все слои с дубликатами), паритет Q-модели и скалярного эквивалента под дедупом (ловит рассинхрон `source_index`), MITM, slow-тест на кубе 2×2×2 (луч 10⁴: без дедупа только 48–66% луча — попарно неэквивалентные состояния)
- [x] write tests (error/edge): дефолт None — поведение не изменилось (существующие тесты) — плюс явный тест «тривиальная группа (одна тождественная) не меняет ни путь, ни `debug_scores`», симметрии другого графа → понятная ошибка, `canonical_dedup` в режиме "advanced" → понятная ошибка, невалидные формы состояний в `canonical`
- [x] run `./lint.sh && RUN_SLOW_TESTS=1 pytest` — must pass before next task — lint зелёный (black 64 файла, pylint 10.00/10, mypy 64 файла), `black --check .` по всему репо зелёный (66 файлов), `docs/build_docs.sh` (`-W`) зелёный; `RUN_SLOW_TESTS=1 pytest` = **375 passed / 12 skipped / 3 xfailed** и на 3.12 (torch 2.13), и на 3.9 (`.venv39`, torch 2.8)
- [x] открыть PR (Draft до мержа PR2/PR12) — [stasdiener/cayleypy#13](https://github.com/stasdiener/cayleypy/pull/13), Draft, base `base/canonical-dedup`, в upstream не отправлялось. Цена опции — `n_symmetries` перестановок состояний + одно хэширование на слой (2.2 с на слой из 828k состояний куба на CPU); замер на мегаминксе с лучом 2^16 остаётся в Post-Completion

### Task 15: PR14 — non-backtracking для search_simple (кандидат на выброс)

**Files:**
- Modify: `cayleypy/algo/beam_search.py`, `cayleypy/algo/beam_search_test.py`

- [x] ⚠️ вопрос «нужен ли, учитывая `history_depth` в `search_advanced`?» отложен до фазы публикации; в форке реализуем (ценность: бан для `search_simple`, где его нет, + O(1)-память vs `history_depth=1` на широких лучах) — подавать ли в upstream, решится по замерам — замеры сделаны (см. последний пункт): на узких лучах nbt сильно лучше и плоского `search_simple`, и `history_depth=1`; на широких — качество как у `history_depth=1`, но дешевле по времени и памяти; при этом на очень широком луче бан может и повредить (задокументировано в докстринге). Итоговое решение по upstream — за пользователем, ⚠️ остаётся до фазы публикации
- [x] ветка `feat/nbt-simple-beam` **от ветки PR13**; реюз `graph_def.generators_inverse_map` (None = не inverse-closed → опция недоступна, понятная ошибка); бан ребёнка с `move == inv_map[parent_move]` по провенансу из PR2 — `banned_moves` (по одному ид генератора на состояние луча) прокинут в `_expand_layer`, где банящиеся слоты `(состояние, генератор)` выкидываются **до дедупа**: если то же состояние породил разрешённый ход (от другого родителя), оно остаётся. Так бан строго слабее, чем выкидывание состояний по хэшу, и никогда не теряет состояние, недостижимое иначе. Ид обратного генератора — через новый хелпер `_inverse_generators` (он же даёт понятную ошибку при не-inverse-closed наборе); ➕ бан прокидывается через фильтры слоя (канон-дедуп `keep` и top-k `idx`), иначе `moves` разъехались бы со своими состояниями
- [x] параметр `non_backtracking: bool = False` — в `search_simple` и `search()`; при `beam_mode="advanced"` — понятная ошибка со ссылкой на `history_depth`; при `False` выполняется ровно старый код; ➕ добавлен выход «слой опустел» (возможен только при nbt: все ходы из всех состояний луча забанены) — иначе поиск крутился бы до `max_steps` на пустом фронтире
- [x] write tests (success): nbt-луч не посещает отменяющие пары (малый граф); результат не хуже на фикс-наборе `lrx(5)` — 15 новых тестов. Бан на уровне расширения слоя: ни одно состояние не пришло из забаненной пары (ожидаемое множество посчитано через `apply_path`), состояние с разрешённым ходом выживает (ручной пример: `[1,0,2,3,4]` даёт X от одного родителя и L от другого — остаётся с `moves=L`), **удалённые состояния = ровно предыдущий слой** (`plain - nbt == {start}`); фикс-набор — все 120 состояний `lrx(5)` при `beam_width=1`: плоский решает 10, nbt — 44, и ни одно решённое плоским поиском не решено хуже (одинаково на torch 2.13 и 2.8); валидный путь на `lrx(8)`; `non_backtracking=False` даёт ровно старый луч (путь и `debug_scores`); комбинации с `use_child_scores` (паритет Q-модели и скалярного эквивалента), с канон-дедупом (тот же паритет — ловит рассинхрон `moves`/`source_index`) и с MITM
- [x] write tests (error/edge): генераторы-инволюции (inv == сам ход); не-inverse-closed набор — инволюции на `cyclic_coxeter(5)` (все 5 генераторов — свои обратные: слой 16 → 15, уходит ровно старт); не-inverse-closed — `lx(5)` (`generators_inverse_map is None`) → понятная ошибка «inverse-closed»; плюс «все ходы забанены» (граф с единственным генератором-инволюцией: nbt останавливается после 1 шага, плоский крутится все 10) и `non_backtracking` в режиме "advanced" → понятная ошибка
- [x] run `./lint.sh && RUN_SLOW_TESTS=1 pytest` — must pass before next task — lint зелёный (black 64 файла, pylint 10.00/10, mypy 64 файла), `black --check .` по всему репо зелёный (66 файлов), `docs/build_docs.sh` (`-W`) зелёный; `RUN_SLOW_TESTS=1 pytest` = **388 passed / 12 skipped / 3 xfailed** и на 3.12 (torch 2.13), и на 3.9 (`.venv39`, torch 2.8); ➕ **починена флакующая от сида часть PR13** (`test_beam_search_simple_cube222_canonical_dedup`: случайный скрэмбл иногда попадал в 10 ходов от центра, слоёв для сравнения не хватало — ~1 падение на 40 прогонов); фикс отдельным коммитом в ветку PR13 (`feat/canonical-dedup`), чтобы диф PR14 остался чистым
- [x] открыть PR + числа сравнения с `history_depth=1` в описании — [stasdiener/cayleypy#14](https://github.com/stasdiener/cayleypy/pull/14), Draft, base `feat/canonical-dedup`, в upstream не отправлялось. Замеры (хэмминг, CPU, все состояния графа как старты, `max_steps=50`, решено стартов): `lrx(5)`, ширины 1/2/5/10 — плоский 10/13/23/39, nbt **44/62/113/116**, `history_depth=1` 10/31/27/50; `lrx(6)` (720 состояний) — плоский 12/14/29/54, nbt **165/45/87/107**, `history_depth=1` 12/22/32/54. На широком луче (куб 2×2×2 QTM, ширина 10⁴, 20 случайных скрэмблов) nbt и `history_depth=1` решают одинаково (8/20 против 10/20 у плоского — бан там только убирает состояния), но nbt 9.0 с против 11.3 с (`history_depth` дополнительно хэширует слой и пересекает с хранилищем) и O(1) память вместо хранения хэшей слоёв

### Task 16: PR15 — LowerBound-отсечение в луче

**Files:**
- Create: `cayleypy/lower_bound.py`, `cayleypy/lower_bound_test.py`
- Modify: `cayleypy/algo/beam_search.py`, `cayleypy/algo/beam_search_test.py`
- Modify: `cayleypy/__init__.py`, `docs/api.rst`

- [x] ветка `feat/lower-bound-pruning` (от PR2, либо от `main` если PR2 застрял — логика не зависит от child-scoring); протокол `LowerBound.lb(states)` (докстринг: требование допустимости) — ➕ **скоуп: ветка от PR14 (`feat/nbt-simple-beam`), а не от PR2**: PR15 тоже правит `beam_search.py`, а по Development Approach такие PR стекаются строго последовательно (PR2 → PR13 → PR14 → PR15), иначе текстовые конфликты между собственными ветками. `LowerBound` — `typing.Protocol` (`@runtime_checkable`), чтобы годился любой объект с методом `lb`, без наследования; в докстринге зафиксировано требование допустимости (`lb ≤ истина`) и последствие его нарушения (поиск теряет пути)
- [x] `BfsLowerBound` поверх `BfsResult`/`bfs_bitmask`: в таблице → точно, вне → `depth+1` — ➕ **`bfs_bitmask` не подошёл и в охват не вошёл**: он возвращает только функцию роста (`list[int]`), сопоставить состояние с дистанцией по нему нельзя (проверено по `algo/bfs_bitmask.py`). `BfsLowerBound` — поверх `BfsResult`: все слои хэшей склеиваются в одну сортированную таблицу (`searchsorted`, один int64 на состояние шара), в таблице → точная дистанция, вне → `radius+1`. Валидация: генераторы inverse-closed (иначе BFS от центра меряет дистанцию в другую сторону), `bfs_result.graph == graph.definition`, хэши всех слоёв на месте (`return_all_hashes=True`) и `layers_hashes[0]` — ровно хэш центрального состояния (одной проверкой ловятся и BFS не от центра, и BFS другим объектом `CayleyGraph`, у которого другой сид хэшера)
- [x] опции `lower_bound: Optional[LowerBound]`, `prune_above: Optional[int]` (семантика в докстринге: известная верхняя оценка длины; None = выкл), прокинуть через `search()` — только `search_simple` (как `use_child_scores`/`canonical_dedup`/`non_backtracking`; в "advanced" — понятная ошибка). Состояние, достигнутое за `k` шагов, выкидывается при `k + lb > prune_above`; отсечение идёт **после** проверки «центральное состояние достигнуто» (найденный путь не теряется) и **после** канон-дедупа (меньше вызовов `lb`), но **до** скоринга и top-k — в этом и смысл: освободить слоты луча. Пустой слой после отсечения = честный «пути такой длины нет». ➕ **скоуп: опции обязательны только вместе** (порознь каждая — молчаливый no-op, поэтому понятная ошибка), `prune_above < 0` отвергается, а «путь длиннее бюджета всё же может быть возвращён» задокументировано (проверка достижения центра идёт раньше отсечения, плюс MITM); ➕ рефакторинг: фильтрация слоя вместе с провенансом (`moves`/`source_index`) вынесена в общий `_filter_layer` (используют канон-дедуп и отсечение)
- [x] write tests (success): на графе с полным BFS фильтр не отсекает состояния оптимального пути (допустимость); max-эффект при заниженном prune_above — 25 новых тестов. Допустимость: в `lower_bound_test.py` — точность при полном BFS, точность внутри шара и `radius+1` снаружи на **всех** состояниях `lrx(6)`, головоломка с цветами (куб 2×2×2), одиночное состояние vs батч, slow-тест против точных дистанций куба 2×2×2. В луче: при **точной** нижней оценке и `prune_above` = истинной дистанции в луче остаются только состояния на оптимальных путях, поэтому beam находит **оптимальный** путь из всех 119 нецентральных состояний `lrx(5)` даже при `beam_width=1` (плоский поиск решает 10) — это и есть «оптимальный путь не отсекается». Max-эффект: slow-тест «чем больше шар, тем сильнее отсечение» на `lrx(6)` при `beam_width=3` (радиус 0/3/9 → 22/130/703 решённых, все найденные пути оптимальны); плюс отсутствие эффекта при `prune_above=1000` (путь и `debug_scores` совпадают с обычным поиском) и комбинации с child-scoring + канон-дедупом + nbt (паритет Q-модели и скалярного эквивалента — ловит рассинхрон провенанса) и с MITM
- [x] write tests (error/edge): `lower_bound=None` — ноль оверхеда/старое поведение; prune_above < длины решения → честный fail поиска — дефолты дают ровно старый луч (путь и `debug_scores`); **все** бюджеты от 0 до истинной дистанции−1 на `lrx(6)` дают `path_found=False`, причём поиск обрывается ровно тогда, когда бюджета уже не хватает (`len(debug_scores) <= prune_above`); отдельный тест «найденный путь не отвергается за превышение бюджета»; плюс одна опция без другой, `prune_above < 0`, `lb` неверной формы, обе опции в режиме "advanced", а в `lower_bound_test.py` — BFS без `return_all_hashes`, BFS другого графа, BFS не от центрального состояния, BFS другим объектом графа (другое хэширование), не inverse-closed генераторы
- [x] run `./lint.sh && RUN_SLOW_TESTS=1 pytest` — must pass before next task — lint зелёный (black 66 файлов, pylint 10.00/10, mypy 66 файлов), `black --check .` по всему репо зелёный (68 файлов), `docs/build_docs.sh` (`-W`) зелёный (страницы `cayleypy.LowerBound` и `cayleypy.BfsLowerBound` генерируются), доктест `lower_bound.py` зелёный; `RUN_SLOW_TESTS=1 pytest` = **413 passed / 12 skipped / 3 xfailed** и на 3.12 (torch 2.13), и на 3.9 (`.venv39`, torch 2.8)
- [x] открыть PR — [stasdiener/cayleypy#15](https://github.com/stasdiener/cayleypy/pull/15), Draft, base `feat/nbt-simple-beam`, в upstream не отправлялось

### Task 17: PR16 — Bellman/DAVI-дообучение

**Files:**
- Create: `cayleypy/train/bellman.py`, `cayleypy/train/bellman_test.py`
- Modify: `cayleypy/train/config.py`, `cayleypy/train/__init__.py`, `docs/api.rst`

- [ ] ветка `feat/bellman-finetune` от PR10; режим Bellman в тренере: таргет `y(s) = 1 + min_a V_target(child_a(s))` — векторизованно через `score_children` **замороженной target-копии** (EMA-копия из PR9 переиспользуется); периодическое обновление target
- [ ] подмешивание anchors в Bellman-батч обязательно (solved таргет 0 + соседи — иначе шкала уплывает; это твоё улучшение 154→139 из ноутбука) — дефолт из `TrainConfig`
- [ ] режим «дообучение»: старт с RW-претрейна (чекпойнт PR1), пониженный lr — сценарий из версий ноутбука `rw-modelbaselines`
- [ ] write tests (success): fixed-point — точная V на крошечном графе является неподвижной точкой оператора Bellman; (slow) дообучение RW-претрейна на `lrx(5)` приближает предсказания к точному BFS
- [ ] write tests (error/edge): без anchors и с нулевым lr шкала не обновляется (санити); несовместимый чекпойнт → понятная ошибка
- [ ] run `./lint.sh && RUN_SLOW_TESTS=1 pytest` — must pass before next task
- [ ] открыть PR

### Task 18: Verify acceptance criteria (только подконтрольное автору)

**Files:**
- Create: ветка `integration/all-features` в форке

- [ ] собрать ветку `integration/all-features` (мерж всех фиче-веток); `./lint.sh && RUN_SLOW_TESTS=1 pytest` зелёные
- [ ] сквозной сценарий на ней: обучить QVModel на `lrx(5)` (PR9+10), дообучить Bellman-режимом (PR16), чекпойнт (PR1), ансамбль (PR7), beam child-scored (PR2) + канон-дедуп (PR12/13) + LowerBound (PR15) — решает все состояния оптимально (сверка с точным BFS)
- [ ] существующие тесты репо проходят без правок на каждой фиче-ветке
- [ ] замечания самопроверки/агент-ревью по каждому PR закрыты или явно отложены (список в Post-Completion)
- [ ] все 16 PR открыты; таблица статусов заведена в Post-Completion

### Task 19: [Final] Update documentation

**Files:**
- Modify: `docs/api.rst` (форк — добивка секций, если где-то не добавлено в PR)
- Modify: `README.md` форка (сниппет «конфиг → обучение → чекпойнт → ансамбль → beam»)
- Modify: `CLAUDE.md` этого репо (создать, если нет — паттерны работы с upstream-PR)
- Move: этот план → `docs/plans/completed/`

- [ ] `docs/api.rst`: все новые публичные символы в autosummary, job `build-docs` зелёный
- [ ] README-сниппет одним блоком
- [ ] обновить/создать CLAUDE.md с выработанными паттернами (stacked-PR протокол, 3.9-ловушки)
- [ ] перенести план в `docs/plans/completed/`

## Post-Completion

*Внешние события и чужие решения — без чекбоксов*

**Таблица статусов PR** (вести здесь; мерж требует 2 апрувов и не подконтролен автору):

| PR | Ветка | Статус |
|---|---|---|
| PR1 | `feat/score-children-contract` | open в форке — [#1](https://github.com/stasdiener/cayleypy/pull/1) (base = `main` форка) |
| PR2 | `feat/child-scored-beam` | draft в форке — [#2](https://github.com/stasdiener/cayleypy/pull/2) (base = `feat/score-children-contract`; Draft до мержа PR1) |
| PR3 | `feat/resmlp-qmlp` | draft в форке — [#3](https://github.com/stasdiener/cayleypy/pull/3) (base = `feat/score-children-contract`; Draft до мержа PR1) |
| PR4 | `feat/group-tokenizer` | draft в форке — [#4](https://github.com/stasdiener/cayleypy/pull/4) (base = `feat/score-children-contract`; Draft до мержа PR1) |
| PR5 | `feat/q-transformer` | draft в форке — [#5](https://github.com/stasdiener/cayleypy/pull/5) (base = `feat/group-tokenizer`; Draft до мержа PR4) |
| PR6 | `feat/az-heads` | draft в форке — [#6](https://github.com/stasdiener/cayleypy/pull/6) (base = `feat/resmlp-qmlp`; Draft до мержа PR3) |
| PR7 | `feat/ensemble-predictor` | draft в форке — [#7](https://github.com/stasdiener/cayleypy/pull/7) (base = `feat/score-children-contract`; Draft до мержа PR1) |
| PR8 | `feat/train-losses` | open в форке — [#8](https://github.com/stasdiener/cayleypy/pull/8) (base = `main` форка; независим от PR1) |
| PR9 | `feat/trainer-core` | draft в форке — [#9](https://github.com/stasdiener/cayleypy/pull/9) (base = `base/trainer-core` = мерж `feat/train-losses` + `feat/score-children-contract`; Draft до мержа PR8 и PR1) |
| PR10 | `feat/train-data-sources` | draft в форке — [#10](https://github.com/stasdiener/cayleypy/pull/10) (base = `feat/trainer-core`; Draft до мержа PR9) |
| PR11 | `feat/demo-checkpoint` | draft в форке — [#11](https://github.com/stasdiener/cayleypy/pull/11) (base = `base/demo-checkpoint` = мерж `feat/train-data-sources` + `feat/resmlp-qmlp` + `feat/child-scored-beam`; Draft до мержа PR10, PR3 и PR2) |
| PR12 | `feat/symmetry-group` | draft в форке — [#12](https://github.com/stasdiener/cayleypy/pull/12) (base = `feat/score-children-contract`; Draft до мержа PR1) |
| PR13 | `feat/canonical-dedup` | draft в форке — [#13](https://github.com/stasdiener/cayleypy/pull/13) (base = `base/canonical-dedup` = мерж `feat/child-scored-beam` + `feat/symmetry-group`; Draft до мержа PR2 и PR12) |
| PR14 | `feat/nbt-simple-beam` | draft в форке — [#14](https://github.com/stasdiener/cayleypy/pull/14) (base = `feat/canonical-dedup`; Draft до мержа PR13) |
| PR15 | `feat/lower-bound-pruning` | draft в форке — [#15](https://github.com/stasdiener/cayleypy/pull/15) (base = `feat/nbt-simple-beam`; Draft до мержа PR14) |
| PR16 | … | not started / draft / open / approved(1/2) / merged / blocked |

**Фаза публикации в upstream (после ручной проверки пользователем; порядок и сроки — его решение):**
- открыть design-issue в upstream: роадмап, ссылки на #151/#188, вопрос о судьбе #157/#175/#177/#170
- подавать смерженные в форке PR в upstream по одному (ветки уже готовы; base upstream-PR — только `main` upstream, поэтому строго после мержа родителя)
- правила upstream: ДВА апрува команды ревьюеров; пуш после апрува сбрасывает апрув — не ребейзить одобренное без необходимости
- stall-политика: 3 недели без ревью → пинг в TG-чате CayleyPy; 6 недель → группа продолжает жить с форка, серия догонит

**Координация с сообществом:**
- анонс серии в TG-чате CayleyPy; Влад/Андрей ревьюят свои куски (трансформер, AZ, anchors) — их апрувы приближают правило двух
- судьба #151 после подачи PR9 в upstream: предложить vlzm закрыть со ссылкой; судьба #157/#175/#177/#170 — выяснить в design-issue на фазе публикации
- stall-эскалация по протоколу из Development Approach (3 недели / 6 недель)

**Миграция весов (нужны авторы):**
- конвертация в формат PR1: трансформер Влада (Kaggle Models), MLP Люды (2 датасета), AZ Андрея, веса Кирилла — токенизатор и порядок генераторов восстанавливать с авторами; `infer_config_from_state_dict` для размеров слоёв
- ⚠️ **паритет Q-трансформера (перенесено из Task 6)**: id весов Влада на Kaggle нигде не опубликован, а его `PieceTransformer` параметризован иначе, чем `TransformerModel` (PR5) — нужен либо переобученный чекпойнт (PR11), либо опция «эмбеддить все стикеры детали» в конфиге; детали и план конверсии — `docs/plans/notes/20260804-task6-transformer-parity.md`
- выложить мигрированные чекпойнты на общий хаб

**Фаза 2 — бэкенды (вне охвата):**
- адаптер CUDA-луча Ивана (precompiled) — вместе с Иваном (#188 уже его)
- JAX/TPU-паритет beam-фич (nbt, канон-дедуп, LowerBound)
- MITM-beam поверх `algo/bfs_mitm.py`
- список `LowerBound`-ов c max-комбинацией — когда появится вторая реализация (PDB)

**Ручная верификация:**
- ➕ после PR16 повторить рецепт демо-чекпойнта на `lrx-18`/`lrx-20` (сейчас 19/20 и 8/10 при луче 1000–10000 — предел sparse-Q-разметки, см. `docs/plans/notes/20260804-task12-demo-checkpoint.md`); если Bellman-дообучение даёт 50/50, зарегистрировать вторую запись и обновить README
- паритет Q-трансформера с оригинальными весами Влада (скрипт из Task 6)
- бенчмарк канон-дедупа и nbt на мегаминксе, луч 2^16, GPU — сравнение средней длины с бейзлайном
