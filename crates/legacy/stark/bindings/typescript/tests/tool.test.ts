import { expect, test, vi } from 'vitest';
import * as z from 'zod';

vi.mock('../src/napi/index', () => ({
    NativeTool: class {},
    ToolFunction: class {},
    Value: class {},
}));

import { StarkToolFunction, starkToolFunction } from '../src/tool';

test('tool factory builds schemas and invokes a typed handler', async () => {
    const tool = starkToolFunction({
        name: 'add',
        description: 'Add two numbers',
        parameters: z.object({
            left: z.number().describe('Left operand.'),
            right: z.number().describe('Right operand.'),
        }),
        returns: z.number(),
        handler: ({ left, right }) => Promise.resolve(left + right),
    });

    expect(tool).toBeInstanceOf(StarkToolFunction);
    expect(tool.name).toBe('add');
    expect(tool.description).toBe('Add two numbers');
    expect(tool.parametersSchema).toMatchObject({
        type: 'object',
        properties: {
            left: {
                type: 'number',
                description: 'Left operand.',
            },
            right: {
                type: 'number',
                description: 'Right operand.',
            },
        },
        required: ['left', 'right'],
        additionalProperties: false,
    });
    expect(tool.returnSchema).toMatchObject({
        type: 'number',
    });
    await expect(tool.invoke({ left: 2, right: 3 })).resolves.toBe(5);
});

test('tool invocation validates arguments and results', async () => {
    const tool = starkToolFunction({
        name: 'length',
        parameters: z.object({
            value: z.string(),
        }),
        returns: z.number(),
        handler: ({ value }) => value.length,
    });

    await expect(tool.invoke({ value: 'app' })).resolves.toBe(3);
    await expect(tool.invoke({ value: 3 } as never)).rejects.toBeInstanceOf(z.ZodError);

    const invalidResult = starkToolFunction({
        name: 'invalid_result',
        parameters: z.object({}),
        returns: z.number(),
        handler: () => 'not a number' as never,
    });
    await expect(invalidResult.invoke({})).rejects.toBeInstanceOf(z.ZodError);
});

test('tool handler receives the invocation abort signal', async () => {
    const controller = new AbortController();
    const tool = starkToolFunction({
        name: 'is_cancelled',
        parameters: z.object({}),
        returns: z.boolean(),
        handler: (_parameters, context) => context.signal.aborted,
    });

    controller.abort();
    await expect(tool.invoke({}, { signal: controller.signal })).resolves.toBe(true);
});
