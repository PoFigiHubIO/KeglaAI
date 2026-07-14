# GitHub Copilot и OpenAI-compatible endpoint

GitHub Copilot Chat в обычной версии **не позволяет** подставить произвольный
OpenAI-compatible endpoint — он жёстко привязан к бэкенду GitHub/OpenAI.
Исключение: **Copilot Enterprise / Copilot для бизнеса** с функцией
"Bring Your Own Model" (BYOM), доступной части корпоративных клиентов через
Azure AI Foundry — но она не поддерживает произвольные self-hosted
llama.cpp-сервера напрямую.

## Как всё-таки использовать модель из этого проекта в связке с Copilot-подобным UX

1. **Рекомендуемый путь** — используйте Continue, Cline или Roo Code
   (см. `continue_config.json`, `cline_config.json`, `roo_code_config.json`
   в этой папке). Они дают тот же UX (inline chat, автодополнение, агентные
   правки файлов), но честно поддерживают произвольный OpenAI-compatible
   `base_url`.

2. Если по корпоративным причинам нужен именно интерфейс Copilot Chat —
   поднимите локальный прокси, транслирующий запросы Copilot в формат
   вашего сервера (нестандартное и хрупкое решение, не рекомендуется для
   продакшена).

3. Для чистого API-доступа из VS Code без специального расширения
   используйте REST Client / Thunder Client с примерами запросов ниже,
   либо стандартный `openai` SDK (см. `vscode/example_openai_python.py`
   и `vscode/example_openai_js.js`).
