#!/usr/bin/env node
/**
 * ua-arch-analyze.js
 *
 * Structural analysis script for architecture layering.
 * Reads an input JSON with fileNodes, importEdges, and allEdges,
 * then computes:
 *   1. Directory groups (files grouped by parent directory)
 *   2. Import adjacency (in-degree, out-degree per node)
 *   3. Pattern matches (tag-based and name-based heuristic layer hints)
 *
 * Usage:
 *   node ua-arch-analyze.js <input.json> <output.json>
 */

const fs = require('fs');
const path = require('path');

const inputPath = process.argv[2];
const outputPath = process.argv[3];

if (!inputPath || !outputPath) {
  console.error('Usage: node ua-arch-analyze.js <input.json> <output.json>');
  process.exit(1);
}

const input = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
const { fileNodes, importEdges, allEdges } = input;

// ─── 1. Directory Groups ───────────────────────────────────────────────────────
const dirGroups = {};
for (const node of fileNodes) {
  const dir = path.dirname(node.filePath) || '.';
  if (!dirGroups[dir]) dirGroups[dir] = [];
  dirGroups[dir].push(node.id);
}

// ─── 2. Import Adjacency ───────────────────────────────────────────────────────
// Build in-degree and out-degree maps from import edges only
const inDegree = {};
const outDegree = {};
const importedBy = {};   // target -> [sources]
const imports = {};      // source -> [targets]

for (const node of fileNodes) {
  inDegree[node.id] = 0;
  outDegree[node.id] = 0;
  importedBy[node.id] = [];
  imports[node.id] = [];
}

for (const edge of importEdges) {
  inDegree[edge.target] = (inDegree[edge.target] || 0) + 1;
  outDegree[edge.source] = (outDegree[edge.source] || 0) + 1;
  importedBy[edge.target].push(edge.source);
  imports[edge.source].push(edge.target);
}

const adjacency = {};
for (const node of fileNodes) {
  adjacency[node.id] = {
    inDegree: inDegree[node.id],
    outDegree: outDegree[node.id],
    importedBy: importedBy[node.id],
    imports: imports[node.id]
  };
}

// ─── 3. Pattern Matches ────────────────────────────────────────────────────────
// Heuristic layer hints based on tags, filename patterns, and structural role

const layerPatterns = [
  {
    layer: 'training-core',
    label: 'Training Core',
    tagPatterns: ['training', 'behavior-cloning', 'dppo', 'ppo', 'math', 'buffer'],
    namePatterns: [/train/i, /dppo/i]
  },
  {
    layer: 'neural-networks',
    label: 'Neural Networks',
    tagPatterns: ['neural-network', 'diffusion-model', 'actor-critic', 'policy'],
    namePatterns: [/network/i, /model/i, /actor/i]
  },
  {
    layer: 'environment',
    label: 'Environment',
    tagPatterns: ['environment-wrapper', 'gymnasium', 'action-chunking', 'normalization', 'reward-scaling'],
    namePatterns: [/wrapper/i, /env/i]
  },
  {
    layer: 'data-pipeline',
    label: 'Data Pipeline',
    tagPatterns: ['data-pipeline', 'expert-data', 'preprocessing'],
    namePatterns: [/download/i, /data/i, /preprocess/i]
  },
  {
    layer: 'evaluation',
    label: 'Evaluation & Visualization',
    tagPatterns: ['evaluation', 'visualization', 'video-recording', 'comparison'],
    namePatterns: [/visuali/i, /eval/i, /plot/i]
  },
  {
    layer: 'project-support',
    label: 'Project Support',
    tagPatterns: ['metadata', 'configuration', 'documentation', 'checkpoint', 'model-weights', 'notebook', 'entry-point'],
    namePatterns: [/\.DS_Store/i, /readme/i, /\.ipynb$/i, /\.pt$/i, /ignore/i]
  }
];

const patternMatches = {};
for (const node of fileNodes) {
  const matches = [];
  for (const pattern of layerPatterns) {
    let tagScore = 0;
    let nameScore = 0;

    // Tag matching
    for (const tag of (node.tags || [])) {
      if (pattern.tagPatterns.includes(tag)) {
        tagScore++;
      }
    }

    // Name matching
    for (const re of pattern.namePatterns) {
      if (re.test(node.name) || re.test(node.filePath)) {
        nameScore++;
      }
    }

    const totalScore = tagScore + nameScore;
    if (totalScore > 0) {
      matches.push({
        layer: pattern.layer,
        label: pattern.label,
        tagScore,
        nameScore,
        totalScore
      });
    }
  }

  // Sort by total score descending
  matches.sort((a, b) => b.totalScore - a.totalScore);

  patternMatches[node.id] = {
    bestMatch: matches.length > 0 ? matches[0].layer : 'unassigned',
    bestLabel: matches.length > 0 ? matches[0].label : 'Unassigned',
    allMatches: matches
  };
}

// ─── 4. Structural role analysis ────────────────────────────────────────────────
// Classify nodes by their structural role in the import graph
const structuralRoles = {};
for (const node of fileNodes) {
  const adj = adjacency[node.id];
  let role = 'isolated'; // no imports, not imported

  if (adj.inDegree > 0 && adj.outDegree > 0) {
    role = 'intermediary'; // both imports and is imported
  } else if (adj.inDegree > 0 && adj.outDegree === 0) {
    role = 'dependency'; // imported by others but imports nothing (foundational)
  } else if (adj.outDegree > 0 && adj.inDegree === 0) {
    role = 'consumer'; // imports others but not imported (top-level)
  }

  structuralRoles[node.id] = role;
}

// ─── 5. Build document-edge analysis ────────────────────────────────────────────
const documentEdges = allEdges.filter(e => e.type === 'documents');
const documentedBy = {};
for (const edge of documentEdges) {
  if (!documentedBy[edge.target]) documentedBy[edge.target] = [];
  documentedBy[edge.target].push(edge.source);
}

// ─── Assemble results ──────────────────────────────────────────────────────────
const results = {
  summary: {
    totalNodes: fileNodes.length,
    totalImportEdges: importEdges.length,
    totalAllEdges: allEdges.length,
    directoryCount: Object.keys(dirGroups).length
  },
  directoryGroups: dirGroups,
  adjacency,
  structuralRoles,
  patternMatches,
  documentedBy,
  suggestedAssignments: {}
};

// Build final suggested assignment (best match per node)
for (const node of fileNodes) {
  results.suggestedAssignments[node.id] = {
    layer: patternMatches[node.id].bestMatch,
    label: patternMatches[node.id].bestLabel,
    structuralRole: structuralRoles[node.id],
    directory: path.dirname(node.filePath) || '.'
  };
}

// Write output
fs.writeFileSync(outputPath, JSON.stringify(results, null, 2), 'utf8');

console.log('=== Architecture Analysis Complete ===');
console.log(`Nodes analyzed: ${fileNodes.length}`);
console.log(`Import edges: ${importEdges.length}`);
console.log(`Directory groups: ${Object.keys(dirGroups).length}`);
console.log('');
console.log('Directory groups:');
for (const [dir, ids] of Object.entries(dirGroups)) {
  console.log(`  ${dir}: ${ids.length} files`);
}
console.log('');
console.log('Structural roles:');
const roleCounts = {};
for (const [id, role] of Object.entries(structuralRoles)) {
  roleCounts[role] = (roleCounts[role] || 0) + 1;
}
for (const [role, count] of Object.entries(roleCounts)) {
  console.log(`  ${role}: ${count}`);
}
console.log('');
console.log('Suggested layer assignments:');
const layerCounts = {};
for (const [id, assignment] of Object.entries(results.suggestedAssignments)) {
  const layer = assignment.label;
  if (!layerCounts[layer]) layerCounts[layer] = [];
  layerCounts[layer].push(id);
}
for (const [layer, ids] of Object.entries(layerCounts)) {
  console.log(`  ${layer} (${ids.length}):`);
  for (const id of ids) {
    console.log(`    - ${id} [${structuralRoles[id]}]`);
  }
}
console.log('');
console.log(`Results written to: ${outputPath}`);
