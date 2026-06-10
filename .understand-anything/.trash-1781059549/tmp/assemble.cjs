#!/usr/bin/env node
const fs = require('fs');

const basePath = '/Users/cheng/Desktop/Projects/vscodebased/dppodex/.understand-anything/intermediate';

const graph = JSON.parse(fs.readFileSync(basePath + '/assembled-graph.json', 'utf8'));
const layers = JSON.parse(fs.readFileSync(basePath + '/layers.json', 'utf8'));
const tour = JSON.parse(fs.readFileSync(basePath + '/tour.json', 'utf8'));

// Fix missing tags
graph.nodes.forEach(n => {
  if (!n.tags || !n.tags.length) {
    n.tags = ["untagged"];
  }
});

const assembled = {
  "version": "1.0.0",
  "project": {
    "name": "dppodex",
    "languages": ["ipynb", "markdown", "python"],
    "frameworks": [],
    "description": "A custom implementation of Diffusion Policy Policy Optimization (DPPO) tailored for high-dimensional, state-based dexterous manipulation tasks, targeting the Adroit Hand Pen task with expert demonstrations from Minari.",
    "analyzedAt": new Date().toISOString(),
    "gitCommitHash": "3659215b684fcb891e74cfde1083899f87953afc"
  },
  "nodes": graph.nodes,
  "edges": graph.edges,
  "layers": layers,
  "tour": tour
};

fs.writeFileSync(basePath + '/assembled-graph.json', JSON.stringify(assembled, null, 2));
console.log('Assembled graph written successfully.');
console.log('Nodes:', assembled.nodes.length);
console.log('Edges:', assembled.edges.length);
console.log('Layers:', assembled.layers.length);
console.log('Tour steps:', assembled.tour.length);
