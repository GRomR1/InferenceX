# Локальный запуск vLLM-бенчмарка InferenceX на MetaX (русское руководство)

Пошаговая инструкция для полностью локального (air-gapped) бенчмаркинга
инференса по методологии InferenceX: поднятие vLLM-сервера, прогон sweep по
конкурентности, генерация канонических JSON-результатов, проверка их
корректности и офлайн-сравнение с другими движками и GPU.

Проверено на: 1x MetaX C500 (65 ГБ), образ
`cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.23.0-maca.ai3.8.0.103-torch2.10-py312-ubuntu22.04-amd64`,
MACA 3.8.1.3, `mx-smi` 2.3.4.

Связанные файлы:

| Файл | Роль |
| --- | --- |
| `benchmarks/local/run_local_sweep.sh` | Оркестратор: sweep + агрегация + сравнение |
| `benchmarks/single_node/fixed_seq_len/qwen3.8_int8_c500.sh` | Launcher: поднимает vLLM в docker (или бьёт по живому серверу) и запускает клиент |
| `benchmarks/benchmark_lib.sh` | Общая библиотека: GPU-мониторинг (`metax`/`mx-smi`), запуск клиента, порт-поллинг |
| `infx/results/fixed_sequence.py` | Нормализация сырого результата в `agg_*.json` (канонический формат) |
| `infx/results/collect_results.py` | Сбор всех точек в один `agg_<run>.json` (список) |
| `infx/results/local_compare.py` | Офлайн-сравнение run против baseline-файлов |
| `infx/results/fetch_baselines.py` | Экспорт опубликованных результатов InferenceX из публичного API в baseline-JSON |

## 1. Подготовка окружения (одноразово)

### 1.1. Python-клиент на хосте

Бенчмарк-клиент (`python3 -m infx.bench_serving.benchmark_serving`) и все
обработчики результатов исполняются **на хосте**, а не в контейнере. Создайте
виртуальное окружение:

```bash
python3 -m venv ~/.venvs/inferencex-bench
~/.venvs/inferencex-bench/bin/pip install \
    "huggingface_hub[cli]" transformers aiohttp tqdm numpy requests pytest pyyaml
```

Дальше везде `python3` — из этого venv (первым на `PATH`).

### 1.2. Веса модели

Локальная абсолютная пути в `MODEL` обходят HuggingFace полностью
(`[[ "$MODEL" != /* ]]` в launcher'е). Скачайте веса любым удобным способом,
например:

```bash
~/.venvs/inferencex-bench/bin/hf download Qwen/Qwen3-0.6B \
    --local-dir /home/$USER/metax-vllm/models/Qwen3-0.6B
```

### 1.3. Baseline'ы для сравнения (нужен интернет)

Один раз на любой машине с доступом в интернет экспортируйте опубликованные
результаты InferenceX в локальные JSON-файлы:

```bash
cd InferenceX
python3 -m infx.results.fetch_baselines \
    --model "Qwen-3.5-397B-A17B" \   # frontend-имя модели, смотреть /availability
    --isl 8192 --osl 1024 \
    --out-dir ~/inferencex-local-bench/baselines
```

Скрипт выгружает все точки (конкурентности) для комбинаций
(hardware, framework, precision) в отдельные файлы вида
`<model>_<isl>x<osl>_<hw>_<fw>_<prec>.json` — списки точек в каноническом
формате. Для выбора подмножества есть `--hardware <hw>` и `--framework <fw>`
(повторяемые). Сами API-имена моделей смотрите по
`https://inferencex.semianalysis.com/api/v1/availability`.

## 2. Запуск sweep'а

### 2.1. Команда

```bash
export PATH=~/.venvs/inferencex-bench/bin:$PATH
cd InferenceX

MODEL=/home/$USER/metax-vllm/models/Qwen3.8-27B-Uncensored-INT8 \
RUNNER_TYPE=metax-c500 MODEL_PREFIX=qwen3.8 FRAMEWORK=vllm PRECISION=int8 \
TP=1 ISL=8192 OSL=1024 RANDOM_RANGE_RATIO=0.0 CONC_LIST="4 8 16 32 64" \
IMAGE=cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.23.0-maca.ai3.8.0.103-torch2.10-py312-ubuntu22.04-amd64 \
BASELINES_DIR=~/inferencex-local-bench/baselines \
bash benchmarks/local/run_local_sweep.sh
```

Для каждого значения в `CONC_LIST` оркестратор:
1. запускает launcher' (тот поднимает docker-контейнер vLLM с нужными
   флагами, ждёт `/health`, гоняет бенчмарк-клиент, убирает контейнер);
2. нормализует точку: `python3 -m infx.results.fixed_sequence`;
3. в конце собирает `agg_*.json` в `agg_<RUN_NAME>.json`
   (`infx.results.collect_results`) и, задан `BASELINES_DIR`, — пишет
   `comparison.md` (`infx.results.local_compare`).

Всё лежит в `~/inferencex-local-bench/results/<RUN_NAME>/` (переменная
`RESULTS_DIR`/`LOCAL_BENCH_HOME` позволяют поменять место).

### 2.2. Параметры оркестратора (обязательные)

| Переменная | Описание |
| --- | --- |
| `MODEL` | Абсолютный хостовый путь к весам (внутрь контейнера маунтится read-only в тот же путь) |
| `RUNNER_TYPE` | Имя железа для результата, например `metax-c500` (попадает в поле `hw` agg-JSON) |
| `MODEL_PREFIX` | Префикс модели для сравнения с baseline (должен совпадать с `infmax_model_prefix` в baseline; например `qwen3.5`) |
| `FRAMEWORK` | Движок: `vllm` / `sglang` / `trt` |
| `PRECISION` | Точность: `int8` / `fp8` / `bf16` / `fp4` … |
| `TP` | Tensor parallelism (число акселераторов на ноду) |
| `ISL`, `OSL` | Фиксированные входная/выходная длины последовательности (синтетическая нагрузка, `RANDOM_RANGE_RATIO=0.0` — без вариации) |
| `IMAGE` | Тег vendor-образа с движком (попадает в поле `image` результата) |

### 2.3. Параметры оркестратора (опциональные)

| Переменная | По умолчанию | Описание |
| --- | --- | --- |
| `CONC_LIST` | `1 2 4 8 16 32 64` | Список точек конкурентности (каждая точка = отдельный запуск сервера) |
| `RANDOM_RANGE_RATIO` | `0.0` | Доля случайной вариации длин (0 — строго фиксированные) |
| `SPEC_DECODING` | `none` | Метод спекулятивного декодинга (поле результата; для честного apples-to-apples сравнивайте одинаковые значения) |
| `DISAGG` | `false` | Disaggregated prefill/decode |
| `EP_SIZE`, `DP_ATTENTION`, `PP_SIZE`, `DCP_SIZE`, `PCP_SIZE` | `1/false/1/1/1` | Топология параллелизма (попадают в agg-JSON и участвуют в валидации) |
| `RUN_NAME` | `local_<prefix>_<prec>_<runner>_<stamp>` | Имя run'а (имена файлов и каталога) |
| `RESULTS_DIR` | `~/inferencex-local-bench/results/$RUN_NAME` | Куда писать результаты |
| `BASELINES_DIR` | (пусто) | Каталог baseline-JSON для `comparison.md` |
| `LAUNCHER` | `.../fixed_seq_len/qwen3.8_int8_c500.sh` | Путь к launcher'у (можно свой) |
| `EXISTING_SERVER_PORT`, `OPENAI_API_KEY` | (пусто) | Режим «сервер уже работает» — см. §2.5 |
| `PORT` | `8888` | Порт сервера (маппинг контейнера) |

### 2.4. Параметры launcher'а (прокидываются как есть)

| Переменная | По умолчанию | Описание |
| --- | --- | --- |
| `METAX_VLLM_IMAGE` | vllm-metax 0.23.0 | Тег образа для docker-запуска |
| `METAX_GPU_MEMORY_UTILIZATION` | `0.92` | `--gpu-memory-utilization`. Если на GPU уже работает другой сервер, снизьте до доли *свободной* памяти (проверка vLLM: свободная >= GMU×всего) |
| `METAX_MAX_NUM_SEQS` | (нет) | `--max-num-seqs` (полезно ограничивать, если общая ёмкость KV мала) |
| `METAX_EXTRA_VLLM_ARGS` | (пусто) | Произвольные флаги vLLM, например `--enforce-eager` (на metax без него EngineCore может зависать на capture CUDA-графов — см. §5) |
| `METAX_CONTAINER_NAME` | `inferencex-bench-<pid>` | Имя контейнера |
| `SERVER_LOG` | `<RESULTS_DIR>/server_c<conc>.log` | Лог сервера |
| `RUN_EVAL` | (не true) | Запустить lm-eval после бенчмарка |

Фиксированные флаги сервера в launcher'е: `--host 0.0.0.0 --port 8888
--tensor-parallel-size $TP --max-model-len $((ISL+OSL+20)) --dtype bfloat16
--trust-remote-code --no-enable-prefix-caching`. Клиент: `--request-rate inf`,
`--ignore-eos`, `--num-warmups 2×CONC`, `--num-prompts 10×CONC`,
`--percentile-metrics ttft,tpot,itl,e2el`.

### 2.5. Режим «бенчмарк уже запущенного сервера»

Если vLLM уже обслуживает модель (например, ваш рабочий контейнер),
не поднимая нового:

```bash
EXISTING_SERVER_PORT=8011 \
OPENAI_API_KEY=opencode-key \   # если сервер поднят с --api-key
MODEL=<путь к весам для токенизатора> ... bash benchmarks/local/run_local_sweep.sh
```

Внимание: серверные флаги (prefix caching, MTP, max-model-len, max-num-seqs)
влияют на цифры — для сравнения с InferenceX-рецептами такой режим годится,
только если конфигурация сервера совпадает с эталонной.

## 3. Генерируемые результаты

Структура `~/inferencex-local-bench/results/<RUN_NAME>/`:

```
bmk_<prefix>_<prec>_<runner>_c<conc>_gpus_<tp>.json   # сырой результат клиента (per point)
agg_bmk_...c<conc>_gpus_<tp>.json                     # каноническая точка (per point)
agg_local_....json                                     # список всех точек (collect_results)
gpu_metrics.csv                                        # телеметрия mx-smi (timestamp,index,power_w)
gpu_metrics_identity.txt                               # снимок mx-smi (версии/устройств)
power_validation_...c<conc>_gpus_<tp>.json             # аудит энергометрик per point
server_c<conc>.log                                     # лог сервера per point
agg/                                                   # копии agg_bmk_*.json (для collect)
comparison.md                                          # если задан BASELINES_DIR
```

### 3.1. Сырой JSON (`bmk_*.json`)

Что пишет клиент: `duration`, `benchmark_start_time_unix` /
`benchmark_end_time_unix` (окно для интеграции мощности), `completed`,
`total_input_tokens`, `total_output_tokens`, `request_throughput`,
`output_throughput`, `total_token_throughput`, `mean/median/std_ttft_ms`,
`p90/p99/p99.9_ttft_ms`, аналогично для `tpot`, `itl`, `e2el`,
`max_concurrency`, `model_id`, `tokenizer_id`, `benchmark_outcome`
(`status: passed|failed`, `requested`, `completed`, `failed`,
`max_failure_rate`).

### 3.2. Канонический JSON (`agg_bmk_*.json`) — главный артефакт

Поля конфигурации (с ними сопоставляются результаты):

| Поле | Пример | Смысл |
| --- | --- | --- |
| `hw` | `metax-c500` | Железо (`RUNNER_TYPE`) |
| `framework`, `precision`, `spec_decoding`, `disagg` | `vllm`, `int8`, `none`, `false` | Конфигурация |
| `isl`, `osl`, `conc` | `8192`, `1024`, `16` | Точка нагрузки |
| `tp`, `ep`, `pp`, `dcp_size`, `pcp_size`, `dp_attention` | `1`, `1`, … | Топология |
| `image` | `cr.metax-tech.com/...` | Образ |
| `infmax_model_prefix` | `qwen3.8` | Модель для сопоставления |
| `benchmark_outcome.status` | `passed` | Ворат по доле упавших запросов (5%) |

Метрики (все латентности — в **секундах**; токены/с — на 1 GPU):

| Поле | Смысл | Лучше |
| --- | --- | --- |
| `tput_per_gpu` | (вход+выход) токены/с на GPU | выше |
| `output_tput_per_gpu` | сгенерированные токены/с на GPU | выше |
| `input_tput_per_gpu` | входные токены/с на GPU | выше |
| `median_ttft`, `p99_ttft` | Time-to-First-Token, медиана/p99 | ниже |
| `median_tpot`, `p99_tpot` | Time-Per-Output-Token | ниже |
| `median_intvty`, `p99_intvty` | 1000/TPOT — токены/с на пользователя (инверсия TPOT) | выше |
| `mean_e2el` и прочие | End-to-end латентность | ниже |
| `power_valid` | 1 — телеметрия мощности прошла валидацию, 0 — нет (см. sidecar) | 1 |
| `avg_power_w`, `avg_total_gpu_power_w` | Средняя мощность 1 GPU / всей ноды, Вт | ниже |
| `p75_power_w`, `p90_power_w`, `p75/p90_total_gpu_power_w` | Перцентили мощности | ниже |
| `total_gpu_energy_j` | Интеграл энергии за окно бенчмарка, Дж | ниже |
| `joules_per_input_token`, `joules_per_output_token`, `joules_per_total_token`, `joules_per_successful_query` | Энергозатраты на токен/запрос, Дж | ниже |

`power_validation_*.json` — полный аудит: окно бенчмарка, число образцов на
GPU, max gap выборки, причины невалидности (пустой список = валидно),
интеграл на GPU.

### 3.3. `agg_<RUN_NAME>.json`

Массив всех точек run'а (выдача `collect_results`). Это и есть формат,
с которым работает `local_compare` и который можно загрузить в локальную БД
по схеме `InferenceX-app/packages/db`, если нужна историческая аналитика.

## 4. Как смотреть результаты

Быстрый осмотр одной точки:

```bash
cd ~/inferencex-local-bench/results/<RUN_NAME>
python3 - <<'PY'
import json, glob
for f in sorted(glob.glob("agg_bmk_*.json")):
    d = json.load(open(f))
    print(d["conc"], d["hw"], d["tput_per_gpu"], d["output_tput_per_gpu"],
          d["median_ttft"], d["median_tpot"], d["power_valid"])
PY
```

На что смотреть:
- `benchmark_outcome.status == "passed"` во всех точках;
- `power_valid == 1` (иначе — причины в `power_validation_*.json`);
- `tput_per_gpu` монотонно растёт с `conc` до saturation, `median_tpot`
  медленно растёт — нормальная форма кривой;
- в `server_c*.log` — отсутствие ошибок vLLM, корректные `non-default args`;
- `gpu_metrics.csv` — непрерывность (max gap < 3 с, иначе мощность
  отметится `sampling_gap_exceeded`).

## 5. Сравнение с другими замерами

### 5.1. Офлайн-сравнение (без сети, без БД)

```bash
cd InferenceX
python3 -m infx.results.local_compare \
    --run ~/inferencex-local-bench/results/<RUN_NAME>/agg_<RUN_NAME>.json \
    --baselines ~/inferencex-local-bench/baselines \
    --out ~/inferencex-local-bench/results/<RUN_NAME>/comparison.md
```

Отчёт `comparison.md` — markdown-таблица на каждую точку: значения run'а,
значения каждого baseline'а, `Δ` (абсолютный) и `Δ%` (относительно baseline'а)
по метрикам §3.2. Точки сопоставляются по кортежу
`(model prefix, framework, precision, spec_decoding, disagg, isl, osl,
conc)`. Коды выхода: `0` — есть совпадения, `1` — совпадений нет
(например, модель/точка отличаются), `2` — ошибка аргументов.

Чтобы сравнение «зацепилось», у run'а должны совпадать с baseline:
`MODEL_PREFIX` (= frontend-имя модели, например `qwen3.5`), `FRAMEWORK`,
`PRECISION`, `ISL`/`OSL` (поэтому в §2.1 ISL=8192/OSL=1024 — под
опубликованные точки), `SPEC_DECODING=none`.

### 5.2. Сравнение «run против самого себя» / двух локальных run'ов

Любые два файла в том же формате работают как baseline:

```bash
mkdir -p ~/inferencex-local-bench/baselines_self
cp <старый_run>/agg_bmk_*.json ~/inferencex-local-bench/baselines_self/
python3 -m infx.results.local_compare \
    --run <новый_run>/agg_<run>.json \
    --baselines ~/inferencex-local-bench/baselines_self \
    --name "новый-run" --out <новый_run>/comparison.md
```

Типичные сценарии: A/B флаг сервера (`--enforce-eager` vs графы), смена
образа, смена `max-num-seqs`.

### 5.3. Альтернатива: официальное сравнение через публичную БД

Если позже понадобится официальный публичный дашборд: результаты попадают в
`InferenceX-app` (PostgreSQL) через CI — это уже требует PR'а (recipe в
`configs/*-master.yaml`, `runners.yaml`, запись в `perf-changelog.yaml`),
см. `docs/results-and-ingestion.md`. Локальный `local_compare` и
`fetch_baselines` дают тот же сравнительный эффект без CI.

## 6. Известные грабли (MetaX / vllm-metax 0.23.0)

1. **`--enforce-eager`**: без него EngineCore может зависать (99% CPU,
   никаких логов) после `Maximum concurrency for ...`. Проверяйте `health`
   и при зависании добавляйте
   `METAX_EXTRA_VLLM_ARGS="--enforce-eager"`. Для продакшен-контейнера с MTP
   графы работают — грабли проявились на коротких max-model-len, но
   перепроверяйте на каждой конфигурации.
2. **`--disable-log-requests`** форк не принимает — не добавлять.
3. **Память**: vLLM требует `свободно >= GMU × всего`. Если на GPU живёт
   другой сервер (напр. рабочий 27B занимает ~59 ГБ из 65), ставьте
   `METAX_GPU_MEMORY_UTILIZATION` под свободную долю (в валидации — 0.07).
4. **ENTRYPOINT образа** — `/opt/conda/bin/vllm`; launcher передаёт
   `--entrypoint ""` и полный `vllm serve ...`, не «serve» голый.
5. **GPU-устройства**: контейнеру нужны `--device /dev/mxcd --device /dev/dri`
   (runc, без nvidia-рантайма).
6. **Клиентские зависимости** живут на хосте, не в контейнере — venv из §1.1
   должен быть первым на `PATH` (оркестратор сам падает с подсказкой, если
   импортов не хватает).
7. **`mx-smi`** пишет CSV только в файл (`-o`), `-t` с `-o` несовместим;
   интервал — миллисекунды (`-l <ms>`). Нормализация raw-CSV в
   `timestamp,index,power_w` — в `stop_gpu_monitor` (ветка `metax`).
8. Один `nodes:N`-label-инвариант CI для локальных run'ов не нужен (Slurm
   не используется), но при переносе точки в CI-рецепт — см. AGENTS.md.

## 7. Быстрая проверочная команда (smoke)

Минимальный smoke-запуск на простой модели (1–2 минуты на точку):

```bash
export PATH=~/.venvs/inferencex-bench/bin:$PATH
cd InferenceX
MODEL=$HOME/metax-vllm/models/Qwen3-0.6B \
RUNNER_TYPE=metax-c500 MODEL_PREFIX=qwen3 FRAMEWORK=vllm PRECISION=bfloat16 \
TP=1 ISL=1024 OSL=512 RANDOM_RANGE_RATIO=0.0 CONC_LIST="1 2" \
IMAGE=cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.23.0-maca.ai3.8.0.103-torch2.10-py312-ubuntu22.04-amd64 \
METAX_GPU_MEMORY_UTILIZATION=0.07 METAX_MAX_NUM_SEQS=8 \
METAX_EXTRA_VLLM_ARGS="--enforce-eager" \
bash benchmarks/local/run_local_sweep.sh
```

Ожидаемый исход (референс, C500 с занятым GPU): 10/10 и 20/20 запросов,
`power_valid=1`, в точке c1 ~184 total tok/s на GPU, c2 ~281; avg power
85–90 Вт.
