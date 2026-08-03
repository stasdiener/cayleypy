# Task 1 — инвентаризация окружения и кода (фактические сигнатуры)

Снято с `main` фoрка `stasdiener/cayleypy` на коммите `fbdde24` (= upstream `main`, последний мерж 2026‑05‑24).
Дата: 2026‑08‑04. Всё ниже — факты из кода, а не из плана; при расхождении с планом верить этому файлу.

## 1. Окружение

| Что | Значение |
|---|---|
| origin | `https://github.com/stasdiener/cayleypy.git` (форк) |
| upstream | `https://github.com/cayleypy/cayleypy.git` |
| `gh repo set-default` | `stasdiener/cayleypy` (защита от случайного PR в upstream) |
| GitHub Actions в форке | включены (`actions/permissions` → `enabled:true`); зарегистрированы 2 workflow: `Continuous integration` (active), `Deploy docs` (active) |
| основной venv | `.venv` — CPython 3.12.12, `uv pip install -e ".[lint,test,dev]"`, torch 2.13.0 |
| venv для 3.9-проверки | `.venv39` — CPython 3.9, `-e ".[test]"` |
| `./lint.sh` на `main` | зелёный: black 59 файлов unchanged, pylint 10.00/10, mypy — no issues in 59 files |
| `RUN_SLOW_TESTS=1 pytest` на `main` (3.12) | зелёный: **299 passed, 12 skipped, 3 xfailed**, 71.6 s |
| `RUN_SLOW_TESTS=1 pytest` на `main` (3.9) | см. раздел 6 |

`.venv`/`.venv39` не попадают в git — `.gitignore` содержит `.venv/` и `venv/` (но не `.venv39/`; папку не коммитить,
либо добавить в `.git/info/exclude` локально).

CI-матрица (`.github/workflows/ci.yaml`): format-check (black 25.1.0, `black --check --diff .` — **весь репозиторий**, не
только `./cayleypy`), lint (`./lint.sh`), tests Ubuntu × Python 3.9/3.10/3.11/3.12/3.13 c `RUN_SLOW_TESTS=1`,
tests Windows (3.12, **bare `pytest`** — без slow), tests macOS (3.12, `RUN_SLOW_TESTS=1`), build-docs (3.13, pandoc).

## 2. Правила upstream (README)

- «How to contribute» — фактически внутри `## How to add a new Cayley graph`, README.md:162‑178:
  форк → ветка → PR → **имя PR должно отражать изменения** («Update graphs_lib» — плохо) → все автопроверки зелёные →
  **два апрува от reviewers team** → **пуш после апрува сбрасывает апрув** → мержит автор.
- README.md:129‑135 (Style): Google Python Style Guide; **точка в конце предложений в комментариях**;
  pylint‑варнинги чинить, а не отключать.
- README.md:153: «Do not add new graphs to `prepare_graph`» — upstream сознательно избегает конфликтных файлов.
- «How to add a new predictor model» (README.md:194‑216): обучить → **проверить, что с beam search надёжно находит пути**
  → `torch.save(model.state_dict(), path)` → веса на Kaggle публично (MIT) → у графа должно быть уникальное
  `CayleyGraphDef.name`, и `prepare_graph(name)` должен возвращать этот граф (нужно для тестов) → `ModelConfig`
  (`weights_kaggle_id`, `weights_path`) → запись в `PREDICTOR_MODELS` по имени графа → `pytest
  cayleypy/models/models_lib_test.py`.
  **Замечание к PR1/PR11:** текущий формат весов — «голый state_dict», описанный внешним `ModelConfig` в коде
  библиотеки. Самоописывающий чекпойнт PR1 — надстройка; обратная совместимость с этим форматом обязательна.

## 3. Фактические сигнатуры (что трогают PR1/PR2)

### `cayleypy/predictor.py`

```python
class Predictor:
    def __init__(self, graph: "CayleyGraph", models_or_heuristics)      # "zero" | "hamming" | nn.Module | .predict | callable
    @staticmethod
    def pretrained(graph: "CayleyGraph")                                # KeyError, если graph.definition.name нет в PREDICTOR_MODELS
    def __call__(self, states: torch.Tensor) -> torch.Tensor
```

- `self.predict: Callable[[torch.Tensor], torch.Tensor]` — атрибут, а не метод (присваивается в `__init__`).
- `__call__` (predictor.py:58‑66) батчит при `num_batches > 1` через **`torch.hstack(ans)`**. Для 1‑D выходов hstack
  склеивает по dim 0 — корректно; для 2‑D склеит по **dim 1** — молча испортит `[B, n_gen]`. Это и есть баг, который
  правит PR1 (`torch.cat(ans, dim=0)`).
- `predict` вызывается на **декодированных** состояниях: в обоих путях луча передаётся `graph.decode_states(...)`.
- «zero»-предиктор возвращает тензор **на CPU** (`torch.zeros((x.shape[0],))` без `device`) — при работе на GPU это
  расхождение устройств уже существует в `main`; не трогать в PR1 без нужды.

### `cayleypy/models/models.py`

```python
@dataclass(frozen=True)
class ModelConfig:
    model_type: str                     # существующее значение — "MLP" (uppercase)
    input_size: int
    num_classes_for_one_hot: int
    layers_sizes: list[int]
    weights_kaggle_id: Optional[str] = None
    weights_path: Optional[str] = None

    @staticmethod
    def from_dict(cfg: dict[str, Any]) -> ModelConfig      # перечисляет поля явно, .get(...) для опциональных
    def _build_model(self) -> nn.Module                    # ФАБРИКА здесь: if model_type == "MLP" -> MlpModel(self)
    def load(self, device="cpu") -> nn.Module              # torch.load(path, map_location=device) — БЕЗ weights_only

class MlpModel(nn.Module):
    def __init__(self, config)                             # assert config.model_type == "MLP"
    def forward(self, x) -> torch.Tensor                   # one_hot -> flatten(start_dim=-2) -> Sequential -> squeeze(-1)
```

- state_dict-ключи `MlpModel`: `layers.{0,1,3,4,...}.{weight,bias}` (плоский `nn.Sequential`:
  Linear/LayerNorm/ReLU × len(layers_sizes), затем финальный `Linear(in, 1)`). **Любая правка `MlpModel` в PR3 не должна
  менять ни ключи, ни формы при `n_outputs == 1`** — иначе ломаются веса на Kaggle.
- `forward` делает `squeeze(-1)`: при `n_outputs == 1` выход 1‑D `[B]`, что и ожидает `__call__`/`argsort`.
- `load()` вызывает `torch.load` **без** `weights_only`; torch ≥ 2.6 по умолчанию `weights_only=True`, так что голые
  state_dict грузятся, а dict-чекпойнт PR1 придётся грузить своим кодом (`checkpoint.py`), явно указывая `weights_only`.
- `# pylint: disable=not-callable` в шапке файла (models.py:1) — уже есть, наследуется на новый код в этом файле.

### `cayleypy/models/models_lib.py`

`PREDICTOR_MODELS: dict[str, ModelConfig]` — только 2 записи: `"lrx-16"` (fedimser/lrx-16/pyTorch/ep60/1,
`model_ep60.pth`), `"lrx-32"` (fedimser/lrx-32-by-mrnnnn/PyTorch/model_final/1, `model_final.pth`). Никакой логики,
только данные — сюда идёт PR11.

### `cayleypy/cayley_graph.py`

```python
class CayleyGraph:
    batch_size: int = 2**20                                            # ctor kwarg, cayley_graph.py:58
    central_state_hash                                                  # = hasher.make_hashes(encode_states(central_state))
    def get_unique_states(self, states, hashes=None) -> tuple[Tensor, Tensor]
    def encode_states(self, states: AnyStateType) -> Tensor
    def decode_states(self, states: Tensor) -> Tensor
    def apply_generator_batched(self, i: int, src: Tensor, dst: Tensor)
    def apply_path(self, states, generator_ids: list[int]) -> Tensor
    def get_neighbors(self, states: Tensor) -> Tensor                    # ВАЖНО: generator-major
    def get_neighbors_decoded(self, states: Tensor) -> Tensor
    def restore_path(self, hashes: list[Tensor], to_state) -> list[int]
```

- **`get_neighbors` (cayley_graph.py:192‑203) — generator-major**: результат имеет форму
  `[n_gen * n_states, encoded_size]`, блок `neighbors[i*n_states : (i+1)*n_states]` = генератор `i`, применённый ко всем
  состояниям. Значит `score_children` → `[B, n_gen]` требует `reshape((n_gen, B, ...))` и **транспонирования**, а не
  `reshape((B, n_gen))`. Это ровно та ловушка, под которую в плане прописан поколоночный тест.
- **`get_unique_states` при `hasher.is_identity`** (cayley_graph.py:124‑126) возвращает `unique_hashes.reshape((-1, 1))`,
  т.е. состояния схлопываются в **один столбец = сам хэш** (случай `encoded_state_size == 1`, малые графы!).
  Для PR2 (индексная карта родитель→ребёнок) это отдельная ветка: форма states меняется, порядок — сортировка по хэшу.
- `get_unique_states` **сортирует по хэшу** и возвращает `(states, hashes)`; в обоих путях луча дедуп идёт по всему
  плоскому набору соседей → связь «родитель→ход» теряется до скоринга (подтверждает мотивацию PR2).

### `cayleypy/cayley_graph_def.py`

```python
@dataclass(frozen=True)
class CayleyGraphDef:
    generators_type: GeneratorType            # PERMUTATION | MATRIX
    generators_permutations: list[list[int]]
    generators_matrices: list[MatrixGenerator]
    generator_names: list[str]
    central_state: list[int]                  # всегда ПЛОСКИЙ list[int] (normalize_central_state)
    name: str

    @cached_property generators -> Union[list[list[int]], list[MatrixGenerator]]
    @cached_property n_generators -> int
    @cached_property state_size -> int                       # == len(central_state)
    @cached_property generators_inverse_map -> Optional[list[int]]   # None, если не inverse-closed
    @cached_property generators_inverse_closed -> bool
    @cached_property decoded_state_shape -> tuple[int, ...]

@dataclass(frozen=True)
class MatrixGenerator:
    matrix: np.ndarray        # dtype int64, (n, n)
    modulo: int               # 0 = int64 с переполнением, иначе 2..2^31
```

- `generators_inverse_map` (cayley_graph_def.py:210‑231) — `cached_property`, `Optional[list[int]]`; для инволюции
  `inv_map[i] == i`. Матричная ветка — O(n²) через `is_inverse_to`.
- Для `graph_hash` в PR1: перестановки — `generators_permutations` (list[list[int]]), матрицы —
  `matrix.tolist()` + `modulo`; `central_state` уже плоский `list[int]`; плюс `generators_type` и (осознанно) **не**
  `generator_names`/`name` (имена не влияют на математику).

### `cayleypy/hasher.py`

```python
class StateHasher:
    def __init__(self, graph: "CayleyGraph", random_seed: Optional[int], chunk_size=2**18)
    make_hashes: Callable[[Tensor], Tensor]     # атрибут; одна из 4 реализаций
    is_identity: bool                            # True, если encoded_state_size == 1 (хэш = само состояние)
```

Реализации: identity (`x.reshape(-1)`), `_hash_splitmix64` (для bit-encoded, когда есть `string_encoder`),
`_make_hashes_cpu_and_modern_gpu` (матричное произведение на `vec_hasher`), `_make_hashes_older_gpu` (fallback).
Все возвращают 1‑D `int64` длины `n_states`. Для PR13 (канон-дедуп по хэшу канонической формы) переиспользуется как есть.

### `cayleypy/algo/beam_search.py`

```python
class BeamSearchAlgorithm:
    def __init__(self, graph: "CayleyGraph")
    def search(self, *, start_state, destination_state=None, beam_mode="simple", predictor=None,
               beam_width=1000, max_steps=1000, history_depth=0, return_path=False,
               bfs_result_for_mitm=None, verbose=0) -> BeamSearchResult
    def search_simple(self, *, start_state, predictor=None, beam_width=1000, max_steps=1000,
                      return_path=False, bfs_result_for_mitm=None) -> BeamSearchResult
    def search_advanced(self, start_state, destination_state=None, *, beam_width=1000, max_steps=1000,
                        history_depth=0, predictor=None, verbose=0) -> BeamSearchResult
```

- Два независимых пути подтверждены. `search()` — диспатч по `beam_mode in {"simple", "advanced"}`, иначе
  `ValueError("Unknown beam_mode:", beam_mode)`.
- `search_simple` (beam_search.py:159‑186): `layer2, layer2_hashes = get_unique_states(get_neighbors(layer1))` →
  проверка mitm → **скоринг только если `len(layer2) >= beam_width`** → `argsort(scores)[:beam_width]`.
  Банов (nbt/history) нет вообще — подтверждает мотивацию PR14. `verbose` в этом пути читается из `graph.verbose`
  (не из аргумента!), порог `>= 2`.
- `search_advanced` (beam_search.py:248‑339): `get_neighbors` → `get_unique_states` → проверка destination →
  hash-бан по `history_depth` (кольцевой буфер `[beam_width * n_gen, history_depth]`) → скоринг при
  `shape[0] > beam_width`. Возвращает `BeamSearchResult(..., None, ...)` — путь не восстанавливает.
- Докстринги — reST `:param x:`; в `search_advanced` есть «мёртвый» `:param batch_size:` (в сигнатуре его нет).
- Оба пути скорят `predictor(graph.decode_states(...))` уже **после** дедупа → провенанс хода потерян (PR2).

### Экспорты и документация

- `cayleypy/__init__.py` — 9 строк импортов; `ModelConfig` в топ-левел **не** экспортирован.
- `cayleypy/models/__init__.py` — только `from .models import ModelConfig`.
- `docs/api.rst` — autosummary по полным именам; секция «Beam search and ML» уже содержит `cayleypy.Predictor`,
  `cayleypy.algo.BeamSearchAlgorithm`, `cayleypy.models.ModelConfig`. Новые символы PR1 добавлять туда же.

## 4. Открытые PR upstream, конфликтные с планом (read-only, 2026‑08‑04)

| PR | Автор | Состояние | base ← head | Файлы | Что важно для нас |
|---|---|---|---|---|---|
| #157 | iKolt | OPEN, не draft, REVIEW_REQUIRED, MERGEABLE, обновлён 2025‑11‑23 | `main` ← `feature/beam-search-unified` | `algo/beam_search.py` (+181/−86), `beam_search_test.py` (+182/−4) | «unified, added mitm and return path to advanced»: `_check_path_found`/`_restore_path` появляются и в `search_advanced`. Переписывает те же тела циклов, что и PR2 → **основной конфликт для PR2/PR13/PR14** |
| #175 | iKolt | OPEN **draft**, base — не main | `feature/dtype-and-bit_encoding_width-optimization` ← `feature/beam-search-iterated-and-bugfix` | `algo/beam_search.py` (+280/−9), `cayley_graph.py` (+9), `beam_search_test.py`, `.gitignore` | iterated-режим + фикс потери пути. В `main` этого нет (плановая заметка подтверждена) |
| #177 | iKolt | OPEN **draft** | `feature/beam-search-iterated-and-bugfix` ← `feature/beam-search-beam-to-cpu` | `algo/beam_search.py` (+132/−118), `cayley_graph.py` (+11/−5) | beam на CPU; стоит третьим в стеке iKolt |
| #170 | iKolt | OPEN **draft** | `feature/beam-search-unified` ← `feature/small-ai-optimization-hints` | 13 файлов, в т.ч. `predictor.py` (+9/−8), `algo/beam_search.py` (+15/−12) | **единственный PR, трогающий `predictor.py`**: оборачивает всё тело `__call__` в `with torch.inference_mode():`, `hstack` оставляет. Текстовый конфликт с правкой батчинга в PR1 — тривиально разрешим (обе правки совместимы по смыслу). Часть с `pyproject.toml` (перенос torch в dependencies) в `main` **уже** применена — PR устарел |
| #151 | vlzm | OPEN, **APPROVED**, **CONFLICTING** (stale с 2025‑10‑07) | `main` ← `add_nn_training` | `cayleypy/trainers/**` (+523), `docs/api.rst`, `pyproject.toml` (+2/−1) | тренер в `cayleypy/trainers/` + новые зависимости в pyproject. Наш тренер — `cayleypy/train/` (не конфликтует по путям); в описании PR9 кредитовать vlzm и сослаться на #151 |
| #188 | TryDotAtwo | OPEN, REVIEW_REQUIRED, MERGEABLE | `main` ← `feature/bfs-torchrun-distributed` | `algo/bfs_distributed.py`, `cayley_graph.py` | распределённый BFS; с планом не пересекается (Фаза 2 — бэкенды) |

Вывод для порядка работ: `predictor.py` и `models/**` практически свободны (только draft #170 задевает `predictor.py`),
а `algo/beam_search.py` — самый горячий файл репозитория (4 открытых PR). Это дополнительный аргумент держать
PR2/PR13/PR14 тонкими и последовательными.

## 5. Ловушки Python 3.9 (CI floor)

- `list[int]` / `dict[str, Any]` / `tuple[...]` в аннотациях **работают** в 3.9 (PEP 585) и уже используются в
  `models.py`, `cayley_graph_def.py`. Не переписывать на `typing.List`.
- **Нельзя**: `X | None` (3.10+), `match`, `dataclasses.KW_ONLY` (3.10+), `itertools.pairwise` (3.10+),
  `zip(strict=True)` (3.10+), `functools.cache` — есть с 3.9, ок; `typing.Self` (3.11+), `typing.TypeAlias` (3.10+).
- В frozen dataclass для изменяемых дефолтов — `field(default_factory=...)`.

## 6. Прогон под Python 3.9

Вместо `uv run -p 3.9 pytest` (пересоздал бы основной `.venv` под 3.9) сделан отдельный интерпретатор:

```
uv venv --python 3.9 .venv39
uv pip install --python .venv39 -e ".[test]"
RUN_SLOW_TESTS=1 .venv39/bin/python -m pytest -q
```

Результат: **299 passed, 12 skipped, 3 xfailed** за 68.6 s — идентично прогону на 3.12.
Окружение: CPython **3.9.6**, torch **2.8.0**, numpy **2.0.2** (на 3.12 — torch 2.13.0). То есть 3.9-ветка CI живёт на
более старом torch: новые API torch проверять на совместимость с 2.8, а не только с 2.13.
Единственный варнинг — `NotOpenSSLWarning` из urllib3 (LibreSSL системного Python 3.9), к коду репозитория не относится.

`.venv39/` нет в `.gitignore` upstream — добавлен в локальный `.git/info/exclude`, чтобы не мусорить в `git status`
на фиче-ветках и не менять отслеживаемый `.gitignore`.
