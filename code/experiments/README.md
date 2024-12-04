# Руководство

Данная директория содержит результаты проведенных экспериментов с кодом из статьи [Optimal Brain Compression](https://arxiv.org/pdf/2208.11580). Основная задача: повторить результаты авторов статьи и убедиться в их достоверности и актуальности.

# Настройка окружения

## Conda

Для настройки окружения для запуска экспериментов используйте следующие команды:

```bash
# Create the environment from the environment.yml file
conda env create -f environment.yml

# Activate the new environment
conda activate experiment
```

Конфигурацию окружения можно найти в файле `environment.yml`.

## Калибровочный датасет

Данные о содержимом калибровочного датасета для ResNet можно найти в папке `imagenet_setup`.

# Эксперименты

## 1. Квантизация

Проверка результатов статьи была проведена на моделях ResNet18 и ResNet50. Логи записаны в соответствующие `.txt` файлы.

### ResNet18
- **4 бита:**

  ```bash
  CUDA_VISIBLE_DEVICES=6 python main_trueobs.py rn18 imagenet quant --wbits 4 --wasym --save rn18_4quant.pth > rn_quant/resnet18_4_quant.txt
  ```
  _Значение accuracy: 69.16_

  Теперь сделаем batchnorm tuning полученной модели:

  ```bash
  CUDA_VISIBLE_DEVICES=4 python postproc.py rn18 imagenet rn18_4quant.pth --bnt > rn_quant/resnet18_4_quant_tuning.txt
  ```
  _Значение accuracy: 69.44_

- **3 бита:**

  ```bash
  CUDA_VISIBLE_DEVICES=5 python main_trueobs.py rn18 imagenet quant --wbits 3 --wasym --save rn18_3quant.pth > rn_quant/resnet18_3_quant.txt
  ```
  _Значение accuracy: 67.81_

  Теперь сделаем batchnorm tuning полученной модели:

  ```bash
  CUDA_VISIBLE_DEVICES=4 python postproc.py rn18 imagenet rn18_3quant.pth --bnt > rn_quant/resnet18_3_quant_tuning.txt
  ```
  _Значение accuracy: 68.21_

- **2 бита:**

  ```bash
  CUDA_VISIBLE_DEVICES=5 python main_trueobs.py rn18 imagenet quant --wbits 2 --wasym --save rn18_2quant.pth > rn_quant/resnet18_2_quant.txt
  ```
  _Значение accuracy: 56.82_

  Теперь сделаем batchnorm tuning полученной модели:

  ```bash
  CUDA_VISIBLE_DEVICES=4 python postproc.py rn18 imagenet rn18_2quant.pth --bnt > rn_quant/resnet18_2_quant_tuning.txt
  ```
  _Значение accuracy: 63.65_

### ResNet50
- **4 бита:**

  ```bash
  CUDA_VISIBLE_DEVICES=6 python main_trueobs.py rn50 imagenet quant --wbits 4 --wasym --save rn50_4quant.pth > rn_quant/resnet50_4_quant.txt
  ```
  _Значение accuracy: 75.58_

  Теперь сделаем batchnorm tuning полученной модели:

  ```bash
  CUDA_VISIBLE_DEVICES=4 python postproc.py rn50 imagenet rn50_4quant.pth --bnt > rn_quant/resnet50_4_quant_tuning.txt
  ```
  _Значение accuracy: 75.10_


- **3 бита:**

  ```bash
  CUDA_VISIBLE_DEVICES=5 python main_trueobs.py rn50 imagenet quant --wbits 3 --wasym --save rn50_3quant.pth > rn_quant/resnet50_3_quant.txt
  ```
  _Значение accuracy: 74.13_

  Теперь сделаем batchnorm tuning полученной модели:

  ```bash
  CUDA_VISIBLE_DEVICES=4 python postproc.py rn50 imagenet rn50_3quant.pth --bnt > rn_quant/resnet50_3_quant_tuning.txt
  ```
  _Значение accuracy: 74.51_

- **2 бита:**

  ```bash
  CUDA_VISIBLE_DEVICES=7 python main_trueobs.py rn50 imagenet quant --wbits 2 --wasym --save rn50_2quant.pth > rn_quant/resnet50_2_quant.txt
  ```
  _Значение accuracy: 61.59_

  Теперь сделаем batchnorm tuning полученной модели:

  ```bash
  CUDA_VISIBLE_DEVICES=4 python postproc.py rn50 imagenet rn50_2quant.pth --bnt > rn_quant/resnet50_2_quant_tuning.txt
  ```
  _Значение accuracy: 70.25_

## 2. Неструктурированный прунинг ResNet50 с использованием SPDY

Эксперименты по неструктурированному прунингу модели ResNet50 с использованием алгоритма динамического программирования SPDY. Результаты записаны в соответствующие `.txt` файлы.

### Подготовка данных
```bash
# Setup database
mkdir models_unstr
CUDA_VISIBLE_DEVICES=2,3 python main_trueobs.py rn50 imagenet unstr --sparse-dir model_unstr

# Compute corresponding losses
CUDA_VISIBLE_DEVICES=2,3 python database.py rn50 imagenet unstr loss > rn_unstr_prune/resnet50_db_log.txt
```

### Обрезание с коэффициентом x2
```bash
# Run DP algorithm to determine 2x compression target
CUDA_VISIBLE_DEVICES=2,3 python spdy.py rn50 imagenet 2 unstr --dp > rn_unstr_prune/resnet50_spdy_x2_log.txt
```
- Объединяем слои и считаем точность:

  ```bash
  CUDA_VISIBLE_DEVICES=2,3 python postproc.py rn50 imagenet rn50_unstr_200x_dp.txt --database unstr > rn_unstr_prune/resnet50_postproc_x2_log.txt
  ```
  _Значение accuracy: 74.13_

- Объединяем слои, делаем batchnorm tuning и считаем точность:

  ```bash
  CUDA_VISIBLE_DEVICES=2,3 python postproc.py rn50 imagenet rn50_unstr_200x_dp.txt --database unstr --bnt > rn_unstr_prune/resnet50_postproc_bnt_x2_log.txt
  ```
  _Значение accuracy: 74.20_

### Обрезание с коэффициентом x3
```bash
# Run DP algorithm to determine 3x compression target
CUDA_VISIBLE_DEVICES=2,3 python spdy.py rn50 imagenet 3 unstr --dp > rn_unstr_prune/resnet50_spdy_x3_log.txt
```
- Объединяем слои и считаем точность:

  ```bash
  CUDA_VISIBLE_DEVICES=2,3 python postproc.py rn50 imagenet rn50_unstr_300x_dp.txt --database unstr > rn_unstr_prune/resnet50_postproc_x3_log.txt
  ```
  _Значение accuracy: 73.46_

- Объединяем слои, делаем batchnorm tuning и считаем точность:

  ```bash
  CUDA_VISIBLE_DEVICES=2,3 python postproc.py rn50 imagenet rn50_unstr_300x_dp.txt --database unstr --bnt > rn_unstr_prune/resnet50_postproc_bnt_x3_log.txt
  ```
  _Значение accuracy: 74.33_

### Обрезание с коэффициентом x4
```bash
# Run DP algorithm to determine 4x compression target
CUDA_VISIBLE_DEVICES=2,3 python spdy.py rn50 imagenet 4 unstr --dp > rn_unstr_prune/resnet50_spdy_x4_log.txt
```
- Объединяем слои и считаем точность:

  ```bash
  CUDA_VISIBLE_DEVICES=2,3 python postproc.py rn50 imagenet rn50_unstr_400x_dp.txt --database unstr > rn_unstr_prune/resnet50_postproc_x4_log.txt
  ```
  _Значение accuracy: 71.48_

- Объединяем слои, делаем batchnorm tuning и считаем точность:

  ```bash
  CUDA_VISIBLE_DEVICES=2,3 python postproc.py rn50 imagenet rn50_unstr_400x_dp.txt --database unstr --bnt > rn_unstr_prune/resnet50_postproc_bnt_x4_log.txt
  ```
  _Значение accuracy: 73.39_

### 3. N:M прунинг ResNet

Эксперименты по обрезанию моделей ResNet с использованием паттерна n:m для прунинга. Результаты записаны в соответствующие `.txt` файлы.

#### ResNet18
- **n=2, m=4:**

  ```bash
  CUDA_VISIBLE_DEVICES=2 python main_trueobs.py rn18 imagenet nmprune --prunen 2 --prunem 4 --save rn18_2_4prune.pth > rn_nm_prune/resnet18_2_4_prune_log.txt
  ```
  _Значение accuracy: 68.16_

  Теперь сделаем batchnorm tuning полученной модели:

  ```bash
  CUDA_VISIBLE_DEVICES=3 python postproc.py rn18 imagenet rn18_2_4prune.pth --bnt > rn_nm_prune/resnet18_2_4_prune_tuning_log.txt
  ```
  _Значение accuracy: 68.70_

- **n=4, m=8:**

  ```bash
  CUDA_VISIBLE_DEVICES=2 python main_trueobs.py rn18 imagenet nmprune --prunen 4 --prunem 8 --save rn18_4_8prune.pth > rn_nm_prune/resnet18_4_8_prune_log.txt -->
  ```
  _Значение accuracy: 69.62_

  Теперь сделаем batchnorm tuning полученной модели:

  ```bash
  CUDA_VISIBLE_DEVICES=3 python postproc.py rn18 imagenet rn18_4_8prune.pth --bnt > rn_nm_prune/resnet18_4_8_prune_tuning_log.txt
  ```
  _Значение accuracy: 69.36_

#### ResNet34
- **n=2, m=4:**

  ```bash
  CUDA_VISIBLE_DEVICES=1 python main_trueobs.py rn34 imagenet nmprune --prunen 2 --prunem 4 --save rn34_2_4prune.pth > rn_nm_prune/resnet34_2_4_prune_log.txt
  ```
  _Значение accuracy: 70.61_

  Теперь сделаем batchnorm tuning полученной модели:

  ```bash
  CUDA_VISIBLE_DEVICES=3 python postproc.py rn34 imagenet rn34_2_4prune.pth --bnt > rn_nm_prune/resnet34_2_4_prune_tuning_log.txt
  ```
  _Значение accuracy: 70.97_

- **n=4, m=8:**

  ```bash
  CUDA_VISIBLE_DEVICES=1 python main_trueobs.py rn34 imagenet nmprune --prunen 4 --prunem 8 --save rn34_4_8prune.pth > rn_nm_prune/resnet34_4_8_prune_log.txt
  ```
  _Значение accuracy: 71.12_

  Теперь сделаем batchnorm tuning полученной модели:

  ```bash
  CUDA_VISIBLE_DEVICES=3 python postproc.py rn34 imagenet rn34_4_8prune.pth --bnt > rn_nm_prune/resnet34_4_8_prune_tuning_log.txt
  ```
  _Значение accuracy: 71.37_

#### ResNet50
- **n=2, m=4:**

  ```bash
  CUDA_VISIBLE_DEVICES=4 python main_trueobs.py rn50 imagenet nmprune --prunen 2 --prunem 4 --save rn50_2_4prune.pth > rn_nm_prune/resnet50_2_4_prune_log.txt
  ```
  _Значение accuracy: 73.21_

  Теперь сделаем batchnorm tuning полученной модели:

  ```bash
  CUDA_VISIBLE_DEVICES=3 python postproc.py rn50 imagenet rn50_2_4prune.pth --bnt > rn_nm_prune/resnet50_2_4_prune_tuning_log.txt
  ```
  _Значение accuracy: 74.28_

- **n=4, m=8:**

  ```bash
  CUDA_VISIBLE_DEVICES=3 python main_trueobs.py rn50 imagenet nmprune --prunen 4 --prunem 8 --save rn50_4_8prune.pth > rn_nm_prune/resnet50_4_8_prune_log.txt
  ```
  _Значение accuracy: 74.23_

  Теперь сделаем batchnorm tuning полученной модели:

  ```bash
  CUDA_VISIBLE_DEVICES=3 python postproc.py rn50 imagenet rn50_4_8prune.pth --bnt > rn_nm_prune/resnet50_4_8_prune_tuning_log.txt
  ```
  _Значение accuracy: 75.04_

### 4. N:M прунинг BERT

Эксперименты по обрезанию моделей BERT с использованием nmprune. Результаты записаны в соответствующие `.txt` файлы.

#### BERT-squad3
- **n=2, m=4:**

  ```bash
  CUDA_VISIBLE_DEVICES=5 python main_trueobs.py bertsquad3 squad nmprune --prunen 2 --prunem 4 --save bert3_2_4prune.pth > bert_nm_prune/bert3_2_4_prune_log.txt
  ```
  _Значение accuracy: 83.44_

#### BERT-squad6
- **n=2, m=4:**

  ```bash
  CUDA_VISIBLE_DEVICES=5 python main_trueobs.py bertsquad6 squad nmprune --prunen 2 --prunem 4 --save bert6_2_4prune.pth > bert_nm_prune/bert6_2_4_prune_log.txt
  ```
  _Значение accuracy: 86.94_

#### BERT-squad
- **n=2, m=4:**

  ```bash
  CUDA_VISIBLE_DEVICES=5 python main_trueobs.py bertsquad squad nmprune --prunen 2 --prunem 4 --save bert_2_4prune.pth > bert_nm_prune/bert_2_4_prune_log.txt
  ```
  _Значение accuracy: 86.73_
