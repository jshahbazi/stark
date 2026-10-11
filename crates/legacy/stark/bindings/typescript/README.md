# Stark

Stark is an independent fork of the Uzu inference engine with Rust, Metal, and CPU
implementation paths. This directory contains the inherited high-level API and
language bindings. The root README documents the scoring scope and current
validation evidence.

Stark binding packages have not been published. Build them from the repository
before using these examples. Upstream Mirai model identifiers and provider
endpoints remain in the inherited registry integration for compatibility.

## Quick Start



From the Stark repository root, build the local TypeScript binding:

```sh
cargo tools install typescript
cargo tools build typescript
```

The package name `@jshahbazi/stark` identifies the local binding. The fork has no
published npm or JSR package.

Run the code below:

```ts
import { ChatConfig, ChatMessage, ChatReplyConfig, Engine, EngineConfig } from '@jshahbazi/stark';

async function main() {
    let engineConfig = EngineConfig.create();
    let engine = await Engine.create(engineConfig);

    let model = await engine.model('alibaba:qwen3.5:0.8b:mirai:mirai-m:4');
    if (!model) {
        throw new Error('Model not found');
    }

    for await (const update of await engine.download(model)) {
        process.stdout.write(`\rDownload progress: ${(update.progress * 100).toFixed(2)}%`);
    }
    console.log();

    let session = await engine.chat(model, ChatConfig.create());

    let messages = [
        ChatMessage.system().withText('You are a helpful assistant'),
        ChatMessage.user().withText('Tell me a short, funny story about a robot')
    ];

    let reply = await session.reply(messages, ChatReplyConfig.create());
    let message = reply[0]?.message;

    if (message) {
        console.log('Reasoning: ', message.reasoning);
        console.log('Text: ', message.text);
    }
}

main().catch((error) => {
    console.error(error);
});
```


<br>

These examples use the inherited high-level registry and inference API. See the [repository README](https://github.com/jshahbazi/stark) for the supported scoring scope and validation evidence.

## Examples

You can run any example via `cargo tools example` \<**typescript**\> \<**chat** | **chat-cloud** | **chat-shared-instance** | **chat-structured-output** | **quick-start** | **tool-calls**\>:

### Chat

In this example, we will download a model and get a reply to a specific list of messages:

```ts
import {
    ChatConfig,
    ChatMessage,
    ChatReplyConfig,
    ChatSessionStreamChunkError,
    ChatSessionStreamChunkReplies,
    Engine,
    EngineConfig
} from '@jshahbazi/stark';

async function main() {
    let engineConfig = EngineConfig.create();
    let engine = await Engine.create(engineConfig);

    let model = await engine.model('alibaba:qwen3.5:0.8b:mirai:mirai-m:4');
    if (!model) {
        throw new Error('Model not found');
    }
    for await (const update of await engine.download(model)) {
        process.stdout.write(`\rDownload progress: ${(update.progress * 100).toFixed(2)}%`);
    }
    console.log();

    let messages = [
        ChatMessage.system().withText('You are a helpful assistant'),
        ChatMessage.user().withText('Tell me a short, funny story about a robot')
    ];
    let session = await engine.chat(model, ChatConfig.create());
    let stream = await session.replyWithStream(messages, ChatReplyConfig.create());
    let message: ChatMessage | undefined;
    for await (const chunk of stream) {
        if (chunk instanceof ChatSessionStreamChunkReplies) {
            message = chunk.replies[0]?.message;
            console.log('Generated tokens: ', chunk.replies[0]?.stats.tokensCountOutput);
        } else if (chunk instanceof ChatSessionStreamChunkError) {
            console.error('Error: ', chunk.error);
        }
    }
    console.log('Reasoning: ', message?.reasoning);
    console.log('Text: ', message?.text);
}

main().catch((error) => {
    console.error(error);
});
```

<br>Once loaded, the same `ChatSession` can be reused for multiple requests until you drop it. Each model may consume a significant amount of RAM, so it's important to keep only one session loaded at a time. For iOS apps, we recommend adding the [Increased Memory Capability](https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.kernel.increased-memory-limit) entitlement to ensure your app can allocate the required memory.

### Chat with the cloud model

In this example, we will get a reply to a specific list of messages from a cloud model:

```ts
import { ChatConfig, ChatMessage, ChatReplyConfig, Engine, EngineConfig, ReasoningEffort } from '@jshahbazi/stark';

async function main() {
    let engineConfig = EngineConfig.create().withOpenaiApiKey('OPENAI_API_KEY');
    let engine = await Engine.create(engineConfig);

    let model = await engine.model('gpt-5');
    if (!model) {
        throw new Error('Model not found');
    }

    let messages = [
        ChatMessage.system().withReasoningEffort("Low" as ReasoningEffort),
        ChatMessage.user().withText('How LLMs work')
    ];

    let session = await engine.chat(model, ChatConfig.create());
    let reply = await session.reply(messages, ChatReplyConfig.create());
    let message = reply[0]?.message;
    if (message) {
        console.log('Reasoning: ', message.reasoning);
        console.log('Text: ', message.text);
    }
}

main().catch((error) => {
    console.error(error);
});
```

### Chat with shared instance

This example shows how to reuse chat instance without reloading model into memory:

```ts
import { ChatConfig, ChatMessage, ChatReplyConfig, Engine, EngineConfig } from '@jshahbazi/stark';

async function main() {
    let engineConfig = EngineConfig.create();
    let engine = await Engine.create(engineConfig);

    let model = await engine.model('alibaba:qwen3.5:0.8b:mirai:mirai-m:4');
    if (!model) {
        throw new Error('Model not found');
    }
    for await (const update of await engine.download(model)) {
        process.stdout.write(`\rDownload progress: ${(update.progress * 100).toFixed(2)}%`);
    }
    console.log();

    // The chat instance owns the loaded model and can be shared between sessions.
    let chatInstance = await engine.chatInstance(model, ChatConfig.create());

    let firstSession = await engine.chatWithInstance(chatInstance);
    let replies = await firstSession.reply(
        [ChatMessage.user().withText('Tell me a short, funny story about a robot')],
        ChatReplyConfig.create(),
    );
    let reply = replies[replies.length - 1];
    if (reply) {
        console.log('First session reasoning: ', reply.message.reasoning);
        console.log('First session text: ', reply.message.text);
    }

    // The second session reuses the already-loaded weights instead of loading the model again.
    let secondSession = await engine.chatWithInstance(chatInstance);
    replies = await secondSession.reply(
        [ChatMessage.user().withText('What is the capital of France?')],
        ChatReplyConfig.create(),
    );
    reply = replies[replies.length - 1];
    if (reply) {
        console.log('\nSecond session reasoning: ', reply.message.reasoning);
        console.log('Second session text: ', reply.message.text);
    }
}

main().catch((error) => {
    console.error(error);
});
```

### Chat with structured output

Sometimes you want the generated output to be valid JSON with predefined fields. You can use `Grammar` to manually specify a JSON schema for the response you want to receive:

```ts
import { ChatConfig, ChatMessage, ChatReplyConfig, Engine, EngineConfig, GrammarJsonSchema, ReasoningEffort } from '@jshahbazi/stark';
import * as z from "zod";

const CountryType = z.object({
    name: z.string(),
    capital: z.string(),
});
const CountryListType = z.array(CountryType);

function structuredResponse<T extends z.ZodType>(response: string | null | undefined, type: T): z.infer<T> | undefined {
    if (!response) {
        return undefined;
    }
    const data = JSON.parse(response);
    const result = type.parse(data);
    return result;
}

async function main() {
    let engineConfig = EngineConfig.create();
    let engine = await Engine.create(engineConfig);

    let model = await engine.model('alibaba:qwen3.5:0.8b:mirai:mirai-m:4');
    if (!model) {
        throw new Error('Model not found');
    }
    for await (const update of await engine.download(model)) {
        process.stdout.write(`\rDownload progress: ${(update.progress * 100).toFixed(2)}%`);
    }
    console.log();

    let schema = z.toJSONSchema(CountryListType);
    let schemaString = JSON.stringify(schema);
    let messages = [
        ChatMessage.system().withReasoningEffort("Disabled" as ReasoningEffort),
        ChatMessage.user().withText('Give me a JSON object containing a list of 3 countries, where each country has name and capital fields')
    ];

    let session = await engine.chat(model, ChatConfig.create());
    let reply = await session.reply(messages, ChatReplyConfig.create().withGrammar(new GrammarJsonSchema(schemaString)));
    let message = reply[0]?.message;
    let countries = structuredResponse(message?.text, CountryListType);
    console.log(countries);
}

main().catch((error) => {
    console.error(error);
});
```

### Tool calls

This example shows how to use external tools:

```ts
import {
    ChatConfig,
    ChatMessage,
    ChatReplyConfig,
    Engine,
    EngineConfig,
    SamplingMethodGreedy,
    SamplingPolicyCustom,
    starkToolFunction,
} from '@jshahbazi/stark';
import * as z from 'zod';


const Coordinate = z.object({
    latitude: z.number().describe('Latitude in decimal degrees.'),
    longitude: z.number().describe('Longitude in decimal degrees.'),
});

type Coordinate = z.infer<typeof Coordinate>;


const getCurrentLocation = starkToolFunction({
    name: 'get_location',
    description: 'Return the current location in coordinates',
    parameters: z.object({}),
    returns: Coordinate,
    handler: (): Coordinate => ({
        latitude: 51.5074,
        longitude: -0.1278,
    }),
});


async function calculateCurrentTemperature({latitude, longitude}: Coordinate): Promise<number> {
    if (!Number.isFinite(Math.hypot(latitude, longitude))) {
        throw new RangeError('Coordinates must be finite');
    }
    return 25;
}

const getCurrentTemperature = starkToolFunction({
    name: 'get_current_temperature',
    description: 'Return the temperature at the provided coordinates',
    parameters: Coordinate,
    returns: z.number(),
    handler: calculateCurrentTemperature,
});


async function main() {
    const engine = await Engine.create(EngineConfig.create());
    const model = await engine.model('alibaba:qwen3.5:0.8b:mirai:mirai-m:4');
    if (!model) {
        throw new Error('Model not found');
    }
    for await (const update of await engine.download(model)) {
        process.stdout.write(`\rDownload progress: ${(update.progress * 100).toFixed(2)}%`);
    }
    process.stdout.write('\n');

    const session = await engine.chat(model, ChatConfig.create());
    await session.addTool(getCurrentLocation);
    await session.addTool(getCurrentTemperature);

    const messages = [
        ChatMessage.system().withText('You are a helpful assistant'),
        ChatMessage.user().withText('What temperature is it now at my location?'),
    ];
    const config = ChatReplyConfig.create().withSamplingPolicy(
        new SamplingPolicyCustom(new SamplingMethodGreedy()),
    );
    const replies = await session.reply(messages, config);
    const message = replies[replies.length - 1]?.message;
    if (message) {
        console.log('Reasoning:', message.reasoning ?? '');
        console.log('Text:', message.text ?? '');
    }
}

main().catch((error: unknown) => {
    console.error(error);
});
```



## Troubleshooting

Report Stark problems in the [issue tracker](https://github.com/jshahbazi/stark/issues).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
