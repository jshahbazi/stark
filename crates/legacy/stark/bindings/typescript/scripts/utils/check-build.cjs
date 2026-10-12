const assert = require('node:assert/strict');
const fs = require('node:fs');
const Module = require('node:module');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const ts = require('typescript');

// Validate the JS package contract without loading a binary or a model.
const nativePath = path.resolve(__dirname, '../../dist/napi/index.js');
const originalLoad = Module._load;
const native = { NativeTool: class {}, ToolFunction: class {}, Value: class {} };
Module._load = function (request, parent, isMain) {
    if (Module._resolveFilename(request, parent, isMain) === nativePath) return native;
    return originalLoad.call(this, request, parent, isMain);
};

async function main() {
    const root = path.resolve(__dirname, '../..');
    const dist = path.resolve(__dirname, '../../dist');
    const pkg = JSON.parse(fs.readFileSync(path.join(dist, 'package.json'), 'utf8'));
    const distRequire = Module.createRequire(path.join(dist, 'package.json'));
    assert.equal(require.resolve(pkg.name), path.join(dist, 'index.js'));
    assert.equal(distRequire.resolve(pkg.name), path.join(dist, 'index.js'));
    const common = require(pkg.name);
    assert.equal(distRequire(pkg.name), common);
    const esm = await import(pkg.name);
    const fixture = path.join(dist, '.stark-package-contract.mjs');
    try {
        fs.writeFileSync(fixture, `export * from ${JSON.stringify(pkg.name)};\n`);
        const distEsm = await import(pathToFileURL(fixture).href);
        assert.equal(distEsm.starkToolFunction, esm.starkToolFunction);
    } finally {
        fs.rmSync(fixture, { force: true });
    }
    assert.notEqual(common.starkToolFunction, esm.starkToolFunction);
    assert.equal(common.NativeTool, native.NativeTool);
    assert.equal(esm.NativeTool, native.NativeTool);
    assert.equal(typeof common.starkToolFunction, 'function');
    assert.equal(typeof esm.starkToolFunction, 'function');
    const z = require('zod');
    for (const entry of [common, esm]) {
        const tool = entry.starkToolFunction({
            name: 'increment',
            parameters: z.object({ value: z.number() }),
            returns: z.number(),
            handler: ({ value }) => value + 1,
        });
        assert.equal(await tool.invoke({ value: 2 }), 3);
    }
    for (const file of ['index', 'tool', 'version']) {
        for (const extension of ['.js', '.mjs', '.d.ts', '.d.mts']) {
            assert.ok(fs.existsSync(path.join(dist, file + extension)), `${file}${extension} is missing`);
            const map = JSON.parse(fs.readFileSync(path.join(dist, file + extension + '.map'), 'utf8'));
            assert.equal(map.file, file + extension);
            assert.ok(map.sources.includes(`src/${file}.ts`));
        }
    }
    assert.equal(pkg.exports['.'].require.types, './index.d.ts');
    assert.equal(pkg.exports['.'].types, './index.d.mts');
    for (const scope of [root, dist]) {
        for (const [sourceExtension, declarationExtension] of [
            ['.cts', '.d.ts'],
            ['.mts', '.d.mts'],
        ]) {
            const filename = path.join(scope, `contract${sourceExtension}`);
            const source = `import { starkToolFunction, type StarkToolContext } from ${JSON.stringify(pkg.name)};
import * as z from 'zod';
const tool = starkToolFunction({
    name: 'increment', parameters: z.object({ value: z.number() }), returns: z.number(),
    handler: ({ value }, context: StarkToolContext) => context.signal.aborted ? value : value + 1,
});
const result: Promise<number> = tool.invoke({ value: 2 });
void result;`;
            const options = {
                module: ts.ModuleKind.NodeNext,
                moduleResolution: ts.ModuleResolutionKind.NodeNext,
                target: ts.ScriptTarget.ES2020,
                strict: true,
                noEmit: true,
                skipLibCheck: true,
            };
            const host = ts.createCompilerHost(options);
            const getSourceFile = host.getSourceFile.bind(host);
            host.getSourceFile = (file, languageVersion, onError) =>
                file === filename ?
                    ts.createSourceFile(file, source, languageVersion)
                :   getSourceFile(file, languageVersion, onError);
            const program = ts.createProgram([filename], options, host);
            const diagnostics = ts.getPreEmitDiagnostics(program);
            assert.equal(
                diagnostics.length,
                0,
                ts.formatDiagnostics(diagnostics, {
                    getCanonicalFileName: (file) => file,
                    getCurrentDirectory: () => process.cwd(),
                    getNewLine: () => '\n',
                }),
            );
            assert.ok(program.getSourceFile(path.join(dist, `index${declarationExtension}`)));
        }
    }
    console.log('CJS, ESM, declarations, source maps, and tool invocation checks passed');
}

main()
    .catch((error) => {
        console.error(
            'TypeScript package checks failed:',
            error instanceof Error ? error.message : 'unknown error',
        );
        process.exitCode = 1;
    })
    .finally(() => {
        Module._load = originalLoad;
    });
