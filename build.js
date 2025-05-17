// build.js  ― Node 16+ (type:module)
import { readdirSync, readFileSync, writeFileSync } from 'fs';
import { mkdir } from 'fs/promises';
import { join } from 'path';
import matter from 'gray-matter';
import { marked } from 'marked';
import fse from 'fs-extra';

const SRC    = process.cwd();
const DIST   = join(SRC, 'dist');
const ASSETS = ['assets', 'images'];

/* ISO → "Feb 17, 2023" */
const formatDate = iso =>
  new Date(iso).toLocaleDateString('en-US', {
    year: 'numeric', month: 'short', day: 'numeric'
  });

/* 0. dist 초기화 */
await fse.remove(DIST);
await mkdir(join(DIST, 'posts'), { recursive: true });

const posts = [];

for (const file of readdirSync(join(SRC, 'posts'))) {
  if (!file.endsWith('.md')) continue;
  const raw = readFileSync(join(SRC, 'posts', file), 'utf-8');
  const { data: attr, content } = matter(raw);
  posts.push({ file, attr, content });          // ★ collect
}

/* 1-b. ★ 날짜(내림차순) 정렬 */
posts.sort((a, b) => new Date(b.attr.date) - new Date(a.attr.date));

/* 1-c. HTML 생성 */
let cardsHTML = '';
const cats = new Set();
for (const { file, attr, content } of posts) {   // ← 여기만 변경
  const htmlBody = marked.parse(content);

  const slug     = file.replace(/\.md$/, '');
  const postDir  = join(DIST, 'posts', slug);
  await mkdir(postDir, { recursive: true });
  const outPath  = join(postDir, 'index.html');

  const formattedDate = formatDate(attr.date);
  const category = Array.isArray(attr.category)
                     ? attr.category[0] : attr.category || 'Misc';
  cats.add(category);

  const root = '../../';

  /* 글 페이지 작성 */
  writeFileSync(outPath, `<!DOCTYPE html>
<html lang="ko"><head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Sangmin Blog | ${attr.title}</title>
  <link rel="stylesheet"
        href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css">
  <link rel="stylesheet" href="${root}assets/styles/main.css">
  <link rel="stylesheet"
        href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
  <link rel="stylesheet"
     href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">

</head><body>
  <main class="article">
    <h1>${attr.title}</h1>
    <div class="meta">${formattedDate} · ${attr.readingTime} min read</div>
    <img class="hero" src="${root}${attr.thumbnail}" alt="cover image">
    ${htmlBody}
  </main>

  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
  <script>hljs.highlightAll();</script>
  <script>
    window.MathJax = {
      tex: { inlineMath: [['$', '$'], ['\\\\(', '\\\\)']] },
      options: { linebreaks: { automatic: true, width: 'container' } }
    };
  </script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
  <script src="${root}assets/js/main.js" defer></script>
</body></html>`);

  /* 카드 HTML (뱃지 포함) */
  cardsHTML += `
    <a class="card" data-cat="${category}" href="posts/${slug}/">
      <div class="thumb-wrap">
        <img src="${attr.thumbnail}" class="thumb" alt="">
        <span class="badge">${category}</span>
      </div>
      <div class="card-body">
        <h2 class="card-title">${attr.title}</h2>
        <div class="meta">${formattedDate} · ${attr.readingTime} min read</div>
      </div>
    </a>`;
}

const catButtons = [...cats]
  .sort()
  .map(c => `<button class="chip" data-filter="${c}">${c}</button>`)
  .join('');
const categoryNav =
  `<button class="chip active" data-filter="all">All</button>${catButtons}`;

/* 3. index.html 주입 */
const indexSrc = readFileSync(join(SRC, 'index.html'), 'utf-8')
  .replace('<!--AUTO-CARDS-->', cardsHTML)
  .replace('<!--AUTO-CATEGORIES-->', categoryNav);
writeFileSync(join(DIST, 'index.html'), indexSrc);

/* 4. 정적 자원 복사 */
for (const dir of ASSETS) {
  await fse.copy(join(SRC, dir), join(DIST, dir));
}

console.log('✅ Build complete → dist/');
