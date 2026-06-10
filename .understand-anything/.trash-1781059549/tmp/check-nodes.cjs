#!/usr/bin/env node
const fs = require('fs');
const g = JSON.parse(fs.readFileSync('/Users/cheng/Desktop/Projects/vscodebased/dppodex/.understand-anything/intermediate/assembled-graph.json', 'utf8'));
g.nodes.forEach((n, i) => {
  if (!n.tags || !n.tags.length) console.log('Missing tags:', i, n.id);
  if (!n.name && !n.label) console.log('Missing name/label:', i, n.id);
});
