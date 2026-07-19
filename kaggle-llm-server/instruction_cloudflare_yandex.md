# Инструкция по настройке кластера 24/7 (Cloudflare Tunnels + Yandex Disk API)
## Специфический гайд для домена `youngfreak.click`

Данное руководство составлено под ваш стек с использованием постоянного **Cloudflare Tunnel** (на домене `youngfreak.click`) и бесплатного **Yandex Disk API** (через сырой OAuth-токен).

---

## 1. Текущий статус сбора данных

### ✅ УЖЕ СОБРАНО:
* **Telegram Bot Token** (`TELEGRAM_BOT_TOKEN`): `7932915257:AAFboG4...` *(ваш токен бота)*
* **Cloudflare Tunnel Token** (`CLOUDFLARE_TUNNEL_TOKEN`): `eyJhIjoi...` *(ваш токен туннеля)*
* **HuggingFace Token** (`HF_TOKEN`): `hf_uXao...` *(ваш токен HuggingFace)*
* **Rclone Provider** (`RCLONE_PROVIDER`): `yandex`
* **Yandex Disk API Token** (`YANDEX_TOKEN`): `y0__wg...` *(ваш токен API Яндекс Диска)*
* **Rotation Time Seconds** (`ROTATION_TIME_SECONDS`): `600` (для теста на 10 минут)

### 🔍 ОСТАЛОСЬ ДОБЫТЬ:
1. **`HANDOVER_SECRET`**: Придумайте любое секретное слово безопасности (например, `youngfreak_secret_123`). Оно должно быть одинаковым на обоих аккаунтах Kaggle.
2. **Kaggle API реквизиты соседа**:
   * Для **Account A**: вам нужно получить юзернейм и API-ключ от **Account B**.
   * Для **Account B**: вам нужно получить юзернейм и API-ключ от **Account A**.
   *(Они находятся в скачиваемом файле `kaggle.json` в настройках каждого аккаунта в разделе **API > Create New Token**)*.

---

## 2. Настройка Cloudflare Zero Trust

Поскольку вы привязали свой домен `youngfreak.click` к вашему Cloudflare, убедитесь, что в панели управления туннелем (`kegla-tunnel`) во вкладке **Public Hostname** созданы следующие два правила:

1. **Интерфейс API и Web UI**:
   * Subdomain: `api`
   * Domain: `youngfreak.click`
   * Type: `HTTP` *(указывать именно http, не https!)*
   * URL: `localhost:8080`

2. **Медиа-сервер (FLUX / Wan)**:
   * Subdomain: `media`
   * Domain: `youngfreak.click`
   * Type: `HTTP` *(указывать именно http, не https!)*
   * URL: `localhost:8081`

---

## 3. Настройка секретов в Kaggle (Circular Chain)

Для создания циклической цепочки 24/7 у вас должно быть два Kaggle-аккаунта (обозначим их как **Account A** и **Account B**). Ноутбук на обоих аккаунтах должен иметь имя `keglaai`.

Перейдите в редактор ноутбука на каждом аккаунте в меню **Add-ons > Secrets** и введите следующие секреты:

### Таблица секретов для Account A:

| Название секрета (Secret Key) | Значение (Value) | Описание |
| :--- | :--- | :--- |
| `TELEGRAM_BOT_TOKEN` | *Ваш TG_API_KEY* | Токен бота Telegram |
| `CLOUDFLARE_TUNNEL_TOKEN` | *Ваш CLOUDFLARE_TUNNEL_TOKEN* | Токен туннеля Cloudflare |
| `HF_TOKEN` | *Ваш HF_TOKEN* | Токен HuggingFace |
| `RCLONE_PROVIDER` | `yandex` | Провайдер бэкапа |
| `YANDEX_TOKEN` | *Ваш YANDEX_TOKEN* | Токен API Яндекс Диска |
| `HANDOVER_SECRET` | `youngfreak_secret_123` | Ваше придуманное секретное слово |
| `ROTATION_TIME_SECONDS` | `600` | Время работы ноды (10 минут) |
| `NEXT_KAGGLE_USERNAME` | *Юзернейм от **Account B*** | Имя целевого аккаунта |
| `NEXT_KAGGLE_KEY` | *API Ключ от **Account B*** | API Ключ целевого аккаунта |
| `NEXT_KAGGLE_SLUG` | `keglaai` | Имя ноутбука (всегда `keglaai`) |

### Таблица секретов для Account B:

| Название секрета (Secret Key) | Значение (Value) | Описание |
| :--- | :--- | :--- |
| `TELEGRAM_BOT_TOKEN` | *Ваш TG_API_KEY* | Токен бота Telegram |
| `CLOUDFLARE_TUNNEL_TOKEN` | *Ваш CLOUDFLARE_TUNNEL_TOKEN* | Токен туннеля Cloudflare |
| `HF_TOKEN` | *Ваш HF_TOKEN* | Токен HuggingFace |
| `RCLONE_PROVIDER` | `yandex` | Провайдер бэкапа |
| `YANDEX_TOKEN` | *Ваш YANDEX_TOKEN* | Токен API Яндекс Диска |
| `HANDOVER_SECRET` | `youngfreak_secret_123` | Ваше придуманное секретное слово |
| `ROTATION_TIME_SECONDS` | `600` | Время работы ноды (10 минут) |
| `NEXT_KAGGLE_USERNAME` | *Юзернейм от **Account A*** | Имя целевого аккаунта |
| `NEXT_KAGGLE_KEY` | *API Ключ от **Account A*** | API Ключ целевого аккаунта |
| `NEXT_KAGGLE_SLUG` | `keglaai` | Имя ноутбука (всегда `keglaai`) |

---

## 4. Сценарий проверочного 10-минутного тестирования

1. **Старт на Account A**: Запустите ячейку ноутбука на **Account A** вручную.
2. **Ожидание готовности**: Дождитесь, пока в логах появится сообщение о готовности туннеля. Проверьте работоспособность бота в Telegram с помощью команды `/status`. Трафик пойдет на домены `api.youngfreak.click` и `media.youngfreak.click`.
3. **Запуск ротации**: Через 10 минут таймер на Account A выполнит `kaggle kernels push` для запуска ноутбука на **Account B**.
4. **Смена ролей**: 
   * Account B запускается в облаке, скачивает `agent.db` с вашего Яндекс Диска (база автоматически загружается с помощью нативного бесплатного API).
   * Подключается к тому же туннелю Cloudflare. Трафик доменов перехватывается на Account B.
   * Account B загружает файл-маркер `handover.signal` на ваш Яндекс Диск.
5. **Авто-выключение Account A**:
   * Account A видит маркер в облаке, делает финальную выгрузку `agent.db` в облако, стирает маркер с Яндекс Диска, мягко выгружает все процессы и гасит сессию Kaggle. 
   * Вы великолепны! Кластер переключился без ручного вмешательства.
