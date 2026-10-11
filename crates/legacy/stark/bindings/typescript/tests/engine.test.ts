import { expect, test } from 'vitest';

const nativeIntegration = test.runIf(process.env['STARK_NATIVE_INTEGRATION'] === '1');

nativeIntegration('chat session exposes tool registration methods', async () => {
    const { ChatSession } = await import('../src/index');
    expect(typeof ChatSession.prototype.addTool).toBe('function');
    expect(typeof ChatSession.prototype.addTools).toBe('function');
});

nativeIntegration(
    'chat reply produces text',
    async () => {
        const { ChatConfig, ChatMessage, ChatReplyConfig, Engine, EngineConfig } =
            await import('../src/index');
        const engine = await Engine.create(EngineConfig.create());

        const model = await engine.model('alibaba:qwen3.5:0.8b:mirai:mirai-m:4');
        expect(model).toBeTruthy();

        for await (const update of await engine.download(model!)) {
            console.log('Download progress:', update.progress);
        }

        const session = await engine.chat(model!, ChatConfig.create());

        const messages = [
            ChatMessage.system().withText('You are a helpful assistant'),
            ChatMessage.user().withText('Hi'),
        ];

        const reply = await session.reply(messages, ChatReplyConfig.create());
        const message = reply[reply.length - 1]?.message;

        expect(message).toBeDefined();
        expect(message!.reasoning).not.toBeNull();
        expect(message!.text).not.toBeNull();
    },
    600_000,
);
