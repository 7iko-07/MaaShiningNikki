// Use the same official artifact client as actions/upload-artifact, once per ZIP.
import { createHash } from 'node:crypto';
import { readFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

export async function uploadIncrementalArtifacts(directory, client) {
  const root = path.resolve(directory);
  const metadata = path.join(root, 'metadata');
  const index = JSON.parse(await readFile(path.join(metadata, 'update-index.json'), 'utf8'));
  const metadataName = 'update-metadata-' + createHash('sha256').update(index.version).digest('hex').slice(0, 24);
  const existing = new Set((await client.listArtifacts()).artifacts.map(artifact => artifact.name));
  // A rerun must not expose an old index while its payloads are being replaced.
  if (existing.has(metadataName)) await client.deleteArtifact(metadataName);
  const deltaNames = [...new Set(index.deltas.map(delta => delta.asset))];
  for (const name of deltaNames) {
    if (!/^delta-[0-9a-f]{24}\.zip$/.test(name)) throw new Error(`Invalid delta name: ${name}`);
    if (existing.has(name.slice(0, -4))) await client.deleteArtifact(name.slice(0, -4));
    await client.uploadArtifact(name.slice(0, -4), [path.join(root, 'deltas', name)],
      path.join(root, 'deltas'), { retentionDays: 90, compressionLevel: 0 });
  }
  const files = (await readdir(metadata))
    .filter(name => name === 'update-index.json' || /^manifest-[0-9a-f]{24}\.json$/.test(name))
    .map(name => path.join(metadata, name));
  // Upload the index last: it is only discoverable after all payloads exist.
  await client.uploadArtifact(metadataName, files, metadata, { retentionDays: 90 });
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const { DefaultArtifactClient } = await import(pathToFileURL(path.resolve(process.argv[3])).href);
  await uploadIncrementalArtifacts(process.argv[2], new DefaultArtifactClient());
}
