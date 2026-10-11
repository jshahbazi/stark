import type { JestConfigWithTsJest } from 'ts-jest';

const config: JestConfigWithTsJest = {
    preset: 'ts-jest/presets/default-esm',
    testEnvironment: 'node',
    transform: {
        '^.+\\.(t|j)sx?$': ['@swc/jest', { sourceMaps: 'inline' }],
    },
    moduleNameMapper: {
        '^@jshahbazi/stark$': '<rootDir>/src/index.ts',
        '^@jshahbazi/stark/(.*)$': '<rootDir>/src/$1',
    },
    modulePathIgnorePatterns: [
        '<rootDir>/dist/',
        '<rootDir>/deno/',
        '<rootDir>/deno_tests/',
        '<rootDir>/packages/',
    ],
    testPathIgnorePatterns: ['scripts'],
    // prettierPath: require.resolve('prettier-2'),
};

export default config;
