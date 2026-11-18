import { createElement } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

const protectedMarkdownPattern =
  /(```[\s\S]*?```|~~~[\s\S]*?~~~|`+[^`\n]*`+|\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]|\$\$[\s\S]*?\$\$|\$[^$\n]+\$)/g;

function normalizeMarkdown(content) {
  return content
    .split(protectedMarkdownPattern)
    .map((part) => {
      if (
        part.startsWith("```") ||
        part.startsWith("~~~") ||
        part.startsWith("`")
      ) {
        return part;
      }

      if (part.startsWith("\\(")) {
        return `$${part.slice(2, -2)}$`;
      }
      if (part.startsWith("\\[")) {
        return `$$${part.slice(2, -2)}$$`;
      }
      if (part.startsWith("$")) {
        return part;
      }

      return part
        .replace(/(^|\n)([ \t]*)\\(#{1,6})(?=\s)/g, "$1$2$3")
        .replace(/\\\*\\\*([^*\r\n]+?)\\\*\\\*/g, "**$1**")
        .replace(/\\\*([^*\r\n]+?)\\\*/g, "*$1*")
        .replace(/^([ \t]*)\\([*+-])(?=\s)/gm, "$1$2")
        .replace(/^([ \t]*)\\(\d+[.)])(?=\s)/gm, "$1$2");
    })
    .join("");
}

export default function MarkdownReply({ content }) {
  return createElement(
    ReactMarkdown,
    {
      remarkPlugins: [remarkGfm, remarkMath],
      rehypePlugins: [
        [rehypeKatex, { strict: false }],
        [rehypeHighlight, { detect: true, ignoreMissing: true }],
      ],
    },
    normalizeMarkdown(content),
  );
}
