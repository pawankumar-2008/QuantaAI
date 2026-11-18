import assert from "node:assert/strict";
import { test } from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import MarkdownReply from "../src/components/MarkdownReply.js";

function renderMarkdown(content) {
  return renderToStaticMarkup(
    createElement(MarkdownReply, { content }),
  );
}

test("renders heading levels, emphasis, nested lists, and ordered steps", () => {
  const html = renderMarkdown(`# Overview
## Details
### Definition

This is **bold** and *italic*.

- Parent
  - Nested **item**
1. First
2. Second`);

  assert.match(html, /<h1>Overview<\/h1>/);
  assert.match(html, /<h2>Details<\/h2>/);
  assert.match(html, /<h3>Definition<\/h3>/);
  assert.match(html, /<strong>bold<\/strong>/);
  assert.match(html, /<em>italic<\/em>/);
  assert.match(html, /<ul>[\s\S]*<ul>[\s\S]*Nested <strong>item<\/strong>[\s\S]*<\/ul>[\s\S]*<\/ul>/);
  assert.match(html, /<ol>[\s\S]*<li>First<\/li>[\s\S]*<li>Second<\/li>[\s\S]*<\/ol>/);
});

test("highlights fenced code and renders inline and display LaTeX", () => {
  const html = renderMarkdown([
    "Inline `tree` and $\\log n + \\lceil x \\rceil$.",
    "",
    "\\[",
    "\\sum_{i=1}^{n} i",
    "\\]",
    "",
    "```js",
    "const answer = 42;",
    "```",
  ].join("\n"));

  assert.match(html, /<code>tree<\/code>/);
  assert.match(html, /class="katex"/);
  assert.match(html, /class="[^"]*language-js[^"]*"/);
  assert.match(html, /class="hljs-keyword"/);
});

test("normalizes escaped Markdown markers without changing code or math", () => {
  const html = renderMarkdown([
    String.raw`\### Definition`,
    "",
    String.raw`\* \*\*Leaf nodes\*\* store all data pointers.`,
    "",
    "The value is \\(\\log n + \\lceil x \\rceil\\) and the literal code is `\\*\\*`.",
    "",
    "&#x20;Internal nodes store search-key values.",
  ].join("\n"));

  assert.match(html, /<h3>Definition<\/h3>/);
  assert.match(html, /<ul>[\s\S]*<li><strong>Leaf nodes<\/strong> store all data pointers\.<\/li>[\s\S]*<\/ul>/);
  assert.match(html, /class="katex"/);
  assert.match(html, /<code>\\\*\\\*<\/code>/);
  assert.match(html, /Internal nodes/);
  assert.doesNotMatch(html, /&#x20;|\\###/);
});

test("renders PDF-extracted Markdown and preserves paragraph and hard-break spacing", () => {
  const html = renderMarkdown(`## PDF definition

A B+ tree is a balanced multi-level tree index.  
This is the next line in the same paragraph.

Extracted paragraph.`);

  assert.match(html, /<h2>PDF definition<\/h2>/);
  assert.match(html, /<p>A B\+ tree is a balanced multi-level tree index\.<br\/?>[\s\S]*This is the next line/);
  assert.match(html, /<p>Extracted paragraph\.<\/p>/);
});

test("renders long responses and does not inject raw HTML", () => {
  const response = Array.from(
    { length: 300 },
    (_, index) => `Paragraph ${index + 1} with **formatted text**.`,
  ).join("\n\n");
  const html = renderMarkdown(`${response}\n\n<img src=x onerror=alert(1)>`);

  assert.equal((html.match(/<p>/g) || []).length, 300);
  assert.doesNotMatch(html, /<img\b/);
  assert.match(html, /&lt;img/);
});
