import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  site: 'https://starkengine.app',
  base: '/docs',
  outDir: '../dist/docs',
  integrations: [starlight({
    title: 'Stark docs',
    favicon: '/assets/stark-engine.jpg',
    customCss: ['./src/styles.css'],
    social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/jshahbazi/stark' }],
    sidebar: [
      { label: 'Start here', items: [{ slug: 'index' }, { slug: 'install' }] },
      { label: 'Use Stark', items: [{ slug: 'models' }, { slug: 'playground' }, { slug: 'manage-models' }] },
      { label: 'Connect applications', items: [{ slug: 'openwork' }, { slug: 'opencode' }, { slug: 'other-apps' }] },
      { label: 'Build with Stark', items: [{ slug: 'first-request' }, { slug: 'api' }, { slug: 'workers' }] },
      { label: 'Help and compatibility', items: [{ slug: 'troubleshooting' }, { slug: 'compatibility' }, { slug: 'updates-privacy' }] },
    ],
  })],
});
