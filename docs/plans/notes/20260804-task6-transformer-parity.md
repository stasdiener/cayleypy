# Task 6 (PR5): паритет Q-трансформера с весами Влада — статус и данные

Дата: 2026-08-04. Ветка кода: `feat/q-transformer` ([stasdiener/cayleypy#5](https://github.com/stasdiener/cayleypy/pull/5)).

## Итог

Скрипт паритета с Kaggle-весами Влада **в этом PR не написан** — он невозможен без двух вещей, которых
нет: (1) id модели на Kaggle нигде не опубликован, (2) архитектура донора параметризована иначе, чем
`TransformerModel`, поэтому «загрузить его state_dict в нашу модель» нельзя в принципе — нужна конверсия,
согласованная с автором. Пункт перенесён в Post-Completion → «Миграция весов» / «Ручная верификация».

## Что проверено (2026-08-04)

- Репозиторий донора [AnanasClassic/cayleypy-training-core](https://github.com/AnanasClassic/cayleypy-training-core)
  доступен; прочитаны `models.py`, `README.md`, `DESIGN.md`, `PROVENANCE.md`, `config.py`, `cli.py`, `configs/`.
- **Слова «kaggle» нет ни в одном из этих файлов.** README упоминает только экспорт «bare `.pth`»-весов рядом с
  полным чекпойнтом (модель + оптимизатор + RNG-состояния) для мегаминкс/IHES-конфигов. Публичного id модели,
  который можно было бы подставить в `weights_kaggle_id` (как `fedimser/lrx-16/pyTorch/ep60/1` в `PREDICTOR_MODELS`),
  найти не удалось → нужен вопрос автору на фазе публикации.

## Архитектура донора: `PieceTransformer` (`models.py`)

Спека для будущего конвертера (ключи state_dict — как в его коде):

- `local_value_embedding: Embedding(max_piece_size * num_classes, d_model)` — **эмбеддинг каждого стикера детали
  отдельно**, с оффсетом `arange(max_piece_size) * num_classes`;
- `piece_projection: Linear(max_piece_size * d_model, d_model)` — конкатенация стикеров детали → один токен;
- `piece_position_embedding: Embedding(num_pieces, d_model)`, `piece_type_embedding: Embedding(num_piece_types, d_model)`;
- `cls_token` (при `pooling="cls"`, дефолт) + `input_norm`, блоки `MegaminxEncoderBlock`/`IHESEncoderBlock`
  (pre-norm: `norm1` → `MultiheadAttention` → `norm2` → FF `Linear→act→Dropout→Linear`, активация silu),
  `output_norm`, `output_layer: Linear(d_model, output_dim)`; `output_dim = num_actions` (Q-модель),
  squeeze при `output_dim == 1` — как у нас;
- раскладки деталей захардкожены (`_p900_layout` для мегаминкса `state_size=120`, `_ihes_layout` для 72);
- дефолты (`build_model`): `d_model=256`, `nhead=8`, `num_layers=4`, `ff_dim=1024`, `dropout=0.0`, `pooling="cls"`.

## Отличия от `TransformerModel` (PR5) — почему веса не переносятся один-в-один

| | донор | PR5 |
|---|---|---|
| токен детали | эмбеддинги всех стикеров + `Linear`-проекция | один эмбеддинг: токен = первый стикер (`GroupTokenizer`) |
| пулинг | CLS-токен | среднее по токенам |
| активация | silu | gelu |
| блоки | самописные (`norm1/attn/norm2/ff`) | `nn.TransformerEncoderLayer` (ключи `self_attn.*`, `linear1/2`) |

Информационно оба кодирования эквивалентны (см. `GroupTokenizer.verify`), но число и форма параметров разные.

## Что делать на фазе публикации

1. Спросить у Влада id весов на Kaggle (или попросить выложить) + конфиг обучения (`configs/megaminx_p900_t000_piece_transformer.json`).
2. Решить с ним: (а) конвертировать веса в нашу параметризацию невозможно → переобучить нашей архитектурой
   (PR11 и так даёт демо-чекпойнт), либо (б) добавить его `piece_projection`-вариант эмбеддинга как опцию конфига
   (`tokenizer_groups` + флаг «эмбеддить все стикеры детали») — тогда паритет становится проверяемым.
3. Скрипт паритета писать после (2): загрузить его веса, прогнать на общем наборе состояний, сверить скоры
  (по прецеденту `models_lib_test` — публичные веса, без `skipif`, пометить slow).
