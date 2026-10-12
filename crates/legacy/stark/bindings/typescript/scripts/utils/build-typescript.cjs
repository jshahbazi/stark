const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');

function outputPath(file, extension) {
    return file
        .replace(/\.d\.ts(?=\.map$|$)/, `.d.${extension === '.mjs' ? 'mts' : 'ts'}`)
        .replace(/(?<!\.d)\.js(?=\.map$|$)/, extension);
}

function modulePaths(extension) {
    return (context) => {
        const literal = (specifier, source) => {
            if (!ts.isStringLiteral(specifier) || !specifier.text.startsWith('.')) return specifier;
            const base = specifier.text.replace(/\.[cm]?js$/, '');
            const resolved = path.resolve(path.dirname(source.fileName), base);
            let suffix = '';
            if (fs.existsSync(`${resolved}.ts`) || fs.existsSync(`${resolved}.d.ts`)) {
                suffix = extension;
            } else if (fs.existsSync(path.join(resolved, 'index.ts'))) {
                suffix = `/index${extension}`;
            }
            return suffix ? ts.factory.createStringLiteral(`${base}${suffix}`) : specifier;
        };
        return (source) => {
            const visit = (node) => {
                if (ts.isImportDeclaration(node)) {
                    return ts.factory.updateImportDeclaration(
                        node,
                        node.modifiers,
                        node.importClause,
                        literal(node.moduleSpecifier, source),
                        node.attributes,
                    );
                }
                if (ts.isExportDeclaration(node) && node.moduleSpecifier) {
                    return ts.factory.updateExportDeclaration(
                        node,
                        node.modifiers,
                        node.isTypeOnly,
                        node.exportClause,
                        literal(node.moduleSpecifier, source),
                        node.attributes,
                    );
                }
                if (ts.isCallExpression(node) && node.expression.kind === ts.SyntaxKind.ImportKeyword) {
                    return ts.factory.updateCallExpression(
                        node,
                        node.expression,
                        node.typeArguments,
                        node.arguments.map((argument) => literal(argument, source)),
                    );
                }
                if (ts.isImportTypeNode(node) && ts.isLiteralTypeNode(node.argument)) {
                    return ts.factory.updateImportTypeNode(
                        node,
                        ts.factory.createLiteralTypeNode(literal(node.argument.literal, source)),
                        node.attributes,
                        node.qualifier,
                        node.typeArguments,
                        node.isTypeOf,
                    );
                }
                return ts.visitEachChild(node, visit, context);
            };
            return ts.visitNode(source, visit);
        };
    };
}

function build(extension, module) {
    const configPath = path.resolve('tsconfig.build.json');
    const config = ts.readConfigFile(configPath, ts.sys.readFile);
    if (config.error) throw new Error(ts.flattenDiagnosticMessageText(config.error.messageText, '\n'));
    const parsed = ts.parseJsonConfigFileContent(config.config, ts.sys, path.dirname(configPath));
    const program = ts.createProgram(parsed.fileNames, { ...parsed.options, module, importHelpers: false });
    const diagnostics = [...parsed.errors, ...ts.getPreEmitDiagnostics(program)];
    if (diagnostics.length) {
        process.stderr.write(
            ts.formatDiagnosticsWithColorAndContext(diagnostics, {
                getCanonicalFileName: (file) => file,
                getCurrentDirectory: ts.sys.getCurrentDirectory,
                getNewLine: () => '\n',
            }),
        );
        process.exitCode = 1;
        return;
    }
    const emitted = program.emit(
        undefined,
        (file, contents) => {
            const destination = outputPath(file, extension);
            if (file.endsWith('.map')) {
                const map = JSON.parse(contents);
                map.file = outputPath(map.file, extension);
                contents = JSON.stringify(map);
            } else {
                contents = contents.replace(
                    /(\/\/# sourceMappingURL=)(.+)$/m,
                    (_, prefix, map) => prefix + outputPath(map, extension),
                );
            }
            fs.mkdirSync(path.dirname(destination), { recursive: true });
            fs.writeFileSync(destination, contents);
        },
        undefined,
        false,
        {
            before: [modulePaths(extension)],
            afterDeclarations: [modulePaths(extension)],
        },
    );
    if (emitted.emitSkipped) process.exitCode = 1;
}

try {
    build('.js', ts.ModuleKind.CommonJS);
    if (!process.exitCode) build('.mjs', ts.ModuleKind.ESNext);
} catch (error) {
    console.error('TypeScript build failed:', error instanceof Error ? error.message : 'unknown error');
    process.exitCode = 1;
}
