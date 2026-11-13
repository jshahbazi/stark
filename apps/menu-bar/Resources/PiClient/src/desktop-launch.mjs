import { spawn } from 'node:child_process';
import { readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import { parseArgs } from 'node:util';
import { fileURLToPath } from 'node:url';
import { discoverProvider, PilotError } from './config.mjs';
import { privateEnvironment, probePi, readProfile } from './desktop-setup.mjs';

export async function launchDesktop({ profile, workspace, key, extraArgs = [] }) {
  const receipt = await readProfile(profile);
  await probePi(receipt.piPath);
  if (!path.isAbsolute(workspace) || !(await stat(workspace)).isDirectory()) throw new PilotError('workspace_unavailable');
  const provider = await discoverProvider(receipt.baseUrl, receipt.modelId, key);
  provider.models[0].maxTokens = Math.min(1024, provider.models[0].maxTokens);
  const config = JSON.parse(await readFile(path.join(profile, 'agent/models.json'), 'utf8'));
  if (JSON.stringify(config.providers?.['stark-local']) !== JSON.stringify(provider)) throw new PilotError('setup_stale');
  const child = spawn(receipt.piPath, [
    '--provider', 'stark-local', '--model', receipt.modelId, '--thinking', 'off',
    '--offline', '--no-session', '--no-extensions', '--extension', path.join(profile, 'extensions/stark-stream-guard.mjs'),
    '--no-mcp', '--no-skills', '--no-prompt-templates', '--no-themes', '--no-context-files', '--no-approve',
    '--tools', 'read,edit,write,bash', ...extraArgs,
  ], {
    cwd: workspace, stdio: 'inherit',
    env: { ...privateEnvironment(receipt.piPath), STARK_API_KEY: key, PI_TELEMETRY: '0', PI_CODING_AGENT_DIR: path.join(profile, 'agent') },
  });
  return await new Promise((resolve, reject) => {
    child.on('error', () => reject(new PilotError('pi_launch_failed')));
    child.on('exit', (code) => resolve(code ?? 1));
  });
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const { values } = parseArgs({ options: { profile: { type: 'string' }, workspace: { type: 'string' } } });
    process.exitCode = await launchDesktop({ profile: values.profile, workspace: values.workspace, key: process.env.STARK_API_KEY });
  } catch (error) {
    const code = error instanceof PilotError ? error.code : 'pi_launch_failed';
    process.stderr.write(`Stark Pi: ${code}. Update setup in Connections.\n`);
    process.exitCode = 1;
  }
}
