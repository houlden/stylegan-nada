# StyleGAN-NADA: CLIP-Guided Domain Adaptation of Image Generators

> Данный репозиторий представляет собой независимую реализацию метода StyleGAN-NADA на основе [оригинальной статьи](https://arxiv.org/abs/2108.00946) для предобученного на датасете FFHQ генератора StyleGAN2. Метод позволяет решать задачу адаптации домена в zero-shot формате. То есть для переноса модели в новый домен не требуются дополнительные графические данные — обучение происходит исключительно по текстовому описанию целевого стиля с помощью Directional CLIP Loss. Такой подход отлично зарекомендовал себя для решения задач художественной стилизации изображений.

![](.github/assets/grid.png)

Также данным методом можно осуществлять **Fine-Tuning** — обучать генератор в несколько этапов. Ниже приведен пример, где сначала базовый генератор обучается менять текстуру лица, потом уже этот генератор используется в качестве базового и задает какое-то свойство (цвет волос), далее уже новый генератор используется в качестве базового и текстура лица ещё дополнительно усиливается. Таким образом можно формировать сложные стили, которые трудно получить одним промптом, а для инференса у нас будет все также одна модель, просто веса её собрались не за один раз.

<p style="text-align: center;">White Walker — Pale gray hair — White Walker</p>

![](.github/assets/white_walker_staging.png)

## Особенности реализации

В проекте реализован следующий функционал:

1. Обучающий пайплайн.
    - Предусмотрена возможность удобного обучения моделей в различных разрешениях: 256x256, 512x512 и 1024x1024.
    - Реализованы следующие механизмы выбора обучающих слоев:
        + `static` — список замороженных слоев передается напрямую в пайплайн обучения и не меняется в процессе.
        + `once` — выбор слоев производится один раз до обучающего цикла в автоматическом формате (под воздействием Global CLIP Loss оцениваются изменения тензора латентов для каждого слоя).
        + `adaptive` — выбор наиболее значимых слоев осуществляется во время обучающего цикла каждые N-шагов.
    - Предусмотрена возможность Fine-Tuning. Можно передать в `train` веса стилизованного генератора `init_generator_weights_path` и дообучить с другим промптом.
2. Инференс.
3. Инверсия.
    - Реализованы два пайплайна для инверсии реальных изображений в латентный код — пошаговая латентная оптимизация (для разрешений 256x256, 512x512 и 1024x1024) и быстрая инверсия с помощью сети-энкодера `e4e` (для разрешения 1024x1024).
    - Для разрешения 1024x1024 также доступен гибридный инвертор. Полученный с помощью `e4e` латент является стартовой точкой латентной оптимизации, что повышает качество результата и ускоряет сходимость.

## Структура проекта

```plaintext
.
├── 📂 data                    # Константные данные (валидационные латенты, средние латенты)
│   ├── 📂 research            # Медиа для отчета
│   ├── 💾 fixed_val_set.pt
│   ├── 💾 w_avg_256.pt
│   ├── 💾 w_avg_512.pt
│   └── 💾 w_avg_1024.pt
├── 📂 notebooks
│   ├── 💻 playground.ipynb    # Ноутбук-песочница (обучение, инференс, инверсия)
│   └── 💻 research.ipynb      # Ноутбук с исследованием
├── 📂 output                                         # Результаты работы скриптов
│   ├── 📂 styles                                     # Результаты работы обучающих скриптов
│   │   └── 📂 *experiment_name*
│   │       ├── 📂 images
│   │       │   ├── 🖼️ comparison_*step*.png          # Сравнение сеток изображений Source-Target
│   │       │   ├── 📷 source.png                     # Сетка Source
│   │       │   └── 🎨 target_*step*.png              # Сетка Target
│   │       ├── 📂 weights
│   │       │   ├── 🧠 *experiment_name*_*step*.pt    # Веса модели каждые save_weights_every_n шагов
│   │       │   └── 🧠 *experiment_name*.pt           # Веса модели на последнем шаге num_steps
│   │       └── ⚙️ config.json                        # Параметры, с которыми запускалось обучение
│   └── 📂 inversion_bank                             # Результаты работы скриптов инверсии
│       └── 📂 *person_name*
│           ├── 🖼️ image_*step*.png                   # Результат инверсии на шаге step
│           ├── 🖼️ image.png                          # Результат инверсии на последнем шаге
│           ├── 💾 w_*step*.pt                        # Инвертированный латент на шаге step
│           └── 💾 w.pt                               # Инвертированный латент на последнем шаге
├── 📂 src                              # Основной исходный код
│   ├── 📂 losses
│   │   ├── 📜 __init__.py
│   │   ├── 📜 clip_losses.py           # CLIP-based лоссы
│   │   ├── 📜 id_loss.py               # ID-Loss (ArcFace)
│   │   └── 📜 w_reg_loss.py            # Reg-Loss (MSE(w_opt, w_avg))
│   ├── 📂 models
│   │   ├── 📜 __init__.py
│   │   ├── 📜 e4e_invertor.py          # E4E-Invertor
│   │   └── 📜 stylegan_nada.py         # StyleGANNADA model
│   ├── 📂 pipelines
│   │   ├── 📜 __init__.py
│   │   ├── 📜 e4e_inversion.py         # Пайплайн E4E-инверсии
│   │   └── 📜 latent_optimization.py   # Пайплайн латентной оптимизации
│   ├── 📂 utils
│   │   ├── 🛠️ __init__.py
│   │   ├── 🛠️ alignment.py             # Выравнивание изображений для инверсии
│   │   ├── 🛠️ blocks_selection.py      # Выбор обучаемых блоков
│   │   ├── 🛠️ clip_utils.py
│   │   ├── 🛠️ fix_random.py
│   │   ├── 🛠️ image.py                 # Утилиты для работы с картинками
│   │   ├── 🛠️ inversion.py             # Вычисление w_avg, утилиты для инверсии
│   │   ├── 🛠️ training_utils.py
│   │   ├── 🛠️ validation.py
│   │   └── 🛠️ weights.py               # Загрузка весов и генераторов
│   ├── 📜 __init__.py
│   └── 📜 train.py                     # Основной скрипт обучения
├── 📂 third_party             # Код из сторонних репозиториев
│   ├── 📂 arcface             # Модель ir_se от TreB1eN
│   ├── 📂 e4e                 # Ядро e4e энкодера от omertov
│   └── 📂 stylegan2           # Ядро StyleGAN2-ADA-PyTorch от NVlabs
├── 📂 weights        # Веса используемых предобученных моделей
├── 🔒 pixi.lock      # Lock-файл зависимостей
└── 🤖 pixi.toml      # Конфигурационный файл зависимостей
```

## Быстрый старт

### Подготовка окружения

Установите пакетный менеджер [Pixi](https://pixi.prefix.dev/latest/) в систему:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

### Установка зависимостей

```bash
git clone https://github.com/houlden/stylegan-nada
cd stylegan-nada
pixi install --locked
```

Если вы вообще не планируете использовать Jupyter Notebook, и вас интересуют только `.py` скрипты или вы не хотите ставить `ipykernel`, `ipywidgets`, `jupyter` в изолированное окружение, то последняя команда может выглядеть так:

```bash
pixi install --locked --environment default
```

### Активация среды

Поскольку проект содержит компилируемые CUDA-файлы из кода StyleGAN2, критически важно правильно активировать окружение.

1. Если вы запускаете скрипт из терминала, то `pixi` сам активирует окружение. Например, следующая команда запустит обучение с параметрами по умолчанию:

```bash
cd stylegan-nada
pixi run train
```

2. Если вы запускаете `notebooks/playground.ipynb` в стандартном Jupyter Notebook, среда активируется автоматически командой:

```bash
cd stylegan-nada
pixi run notebook
```

3. Если вы работаете с `.ipynb` в VS Code, то автоматически он не активирует полноценное `pixi`-окружение, а лишь позволяет выбрать нужный интерпретатор. Чтобы всё работало "из коробки" нужно заранее активировать окружение в терминале и запустить из него VS Code:

```bash
cd stylegan-nada
pixi shell --environment dev
code .
```

Также для VS Code есть расширение `Pixi`, которое активирует `pixi`-окружения, но оно совершенно не обязательно.

### Запуск обучения

Ниже представлен вариант запуска обучающего скрипта с передачей всех возможных параметров (в качестве значений указаны значения по умолчанию). Подробное описание всех параметров представлено в `notebooks/playground.ipynb`.

```bash
pixi run train \
    --source_text "a photo of a person" \
    --target_text "a sketch of a person"\
    --experiment_name "sketch" \
    --resolution 256 \
    --device "cuda" \
    --seed 101 \
    --num_steps 300 \
    --batch_size 4 \
    --lr 0.002 \
    --truncation_psi 0.7 \
    --clip_model_name "ViT-B/32" \
    --blocks_selection_mode "static" \
    --blocks_to_freeze b4 b8 b16 b32 \
    --k_trainable_blocks 3 \
    --select_batch_size 16 \
    --select_num_steps 50 \
    --select_lr 0.01 \
    --select_criterion "absolute" \
    --select_norm "l2" \
    --adaptive_selection_every_n 50 \
    --weights_dir "weights" \
    --output_dir "output/styles" \
    --init_generator_weights_path "your/path/default/None" \
    --save_changed_only \
    --logging_every_n 10 \
    --save_weights_every_n 50 \
    --validate_every_n 50 \
    --val_set_path "data/fixed_val_set.pt" \
    --verbose 2
```

Вы также можете передать только те параметры, которые отличаются от стандартных:

```bash
pixi run train --target_text "a vampire" --experiment_name "vampire"
```

Также, запустить обучение, инференс и инверсию можно в ноутбуке-песочнице `notebooks/playground.ipynb`.

## Загрузка предобученных весов

Загружать веса вручную нет необходимости, при первом выполнении соответствующие скрипты их сами загрузят, но ссылки на используемые веса все-таки будут ниже.

| Модель              | Ссылка                                                                                                               | Путь в проекте                                  |
|---------------------|----------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| StyleGAN2 FFHQ 256  | [link](https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-256x256.pkl)   | `weights/stylegan2-ffhq-256x256.pkl`            |
| StyleGAN2 FFHQ 512  | [link](https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-512x512.pkl)   | `weights/stylegan2-ffhq-512x512.pkl`            |
| StyleGAN2 FFHQ 1024 | [link](https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-1024x1024.pkl) | `weights/stylegan2-ffhq-1024x1024.pkl`          |
| E4E FFHQ 1024       | [link](https://drive.google.com/file/d/1cUv_reLE6k3604or78EranS7XzuVMWeO/view?usp=sharing)                           | `weights/e4e_ffhq_encode.pt`                    |
| Shape Predictor     | [link](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)                                              | `weights/shape_predictor_68_face_landmarks.dat` |
| IR_SE50             | [link](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view?usp=sharing)                           | `weights/model_ir_se50.pth`                     |

## Благодарности

- Код `StyleGAN2` взят из официальной PyTorch реализации от NVlabs: [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch).
- Код `e4e` энкодера взят из оригинального репозитория: [encoder4editing](https://github.com/omertov/encoder4editing).
- Код `ArcFace` взят из репозитория: [ArcFace](https://github.com/TreB1eN/InsightFace_Pytorch).

Официальная реализация `StyleGAN-NADA` от авторов: [StyleGAN-NADA](https://github.com/rinongal/StyleGAN-nada).