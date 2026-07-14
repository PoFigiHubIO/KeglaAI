// Этап 10. Пример подключения к серверу через OpenAI SDK (JavaScript / Node.js)
//
// npm install openai
//
// Замените PUBLIC_URL на ссылку, которую выводит start.py после запуска
// Cloudflare Tunnel.

import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://PUBLIC_URL/v1",
  apiKey: "sk-no-key-required",
});

async function main() {
  // --- Простой запрос ---
  const completion = await client.chat.completions.create({
    model: "local-model",
    messages: [{ role: "user", content: "Привет! Напиши функцию сортировки на JS." }],
    temperature: 0.7,
    max_tokens: 512,
  });
  console.log(completion.choices[0].message.content);

  // --- Streaming ---
  const stream = await client.chat.completions.create({
    model: "local-model",
    messages: [{ role: "user", content: "Расскажи короткую историю про робота." }],
    stream: true,
  });
  for await (const chunk of stream) {
    const delta = chunk.choices[0]?.delta?.content;
    if (delta) process.stdout.write(delta);
  }
  console.log();

  // --- Function calling ---
  const toolResp = await client.chat.completions.create({
    model: "local-model",
    messages: [{ role: "user", content: "Какая погода в Париже?" }],
    tools: [
      {
        type: "function",
        function: {
          name: "get_weather",
          description: "Получить текущую погоду в указанном городе",
          parameters: {
            type: "object",
            properties: { city: { type: "string" } },
            required: ["city"],
          },
        },
      },
    ],
  });
  console.log(JSON.stringify(toolResp.choices[0].message, null, 2));
}

main().catch(console.error);
