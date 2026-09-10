import { copyFileSync, readFileSync, writeFileSync } from 'node:fs';
const root = new URL('../', import.meta.url);
copyFileSync(new URL('packages/js/dist/index.js',root),new URL('site/js/runtime.js',root));
copyFileSync(new URL('LICENSE',root),new URL('site/license.txt',root));
console.log('Synced browser runtime and license.');

writeFileSync(new URL('site/js/dodge-runtime.js',root),readFileSync(new URL('packages/js/dist/dodge.js',root),'utf8').replace('./index.js','./runtime.js'));
