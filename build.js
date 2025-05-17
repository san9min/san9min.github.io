// build.js  ── Node 16+ (type:module)
import { readdirSync, readFileSync, writeFileSync } from 'fs';
import { mkdir } from 'fs/promises';
import { join } from 'path';
import matter from 'gray-matter';
import { marked } from 'marked';
import fse from 'fs-extra';

const SRC    = process.cwd();          // 프로젝트 루트
const DIST   = join(SRC, 'dist');
const ASSETS = ['assets', 'images'];   // 그대로 복사할 정적 폴더

/* ─ helper: ISO → "Feb 17, 2023" ──────────────────────────────── */
const formatDate = iso =>
  new Date(iso).toLocaleDateString('en-US', {
    year: 'numeric', month: 'short', day: 'numeric'
  });

/* ─ 0. dist 초기화 ─────────────────────────────────────────────── */
await fse.remove(DIST);
await mkdir(join(DIST, 'posts'), { recursive: true });

/* ─ 1. posts/*.md → HTML 변환 & 카드 수집 ─────────────────────── */
let cardsHTML = '';

for (const file of readdirSync(join(SRC, 'posts'))) {
  if (!file.endsWith('.md')) continue;

  const raw                = readFileSync(join(SRC, 'posts', file), 'utf-8');
  const { data: attr, content } = matter(raw);
  const htmlBody           = marked.parse(content);
  const outName            = file.replace('.md', '.html');
  const outPath            = join(DIST, 'posts', outName);
  const formattedDate      = formatDate(attr.date);

  /* 글 HTML 생성 */
  writeFileSync(
    outPath,
    `<!DOCTYPE html><html lang="ko"><head>
       <meta charset="utf-8">
       <meta name="viewport" content="width=device-width,initial-scale=1">
       <title>${attr.title}</title>

       <!-- Pretendard Font -->
       <link rel="stylesheet"
             href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css">
       <!-- 메인 CSS -->
       <link rel="stylesheet" href="../assets/styles/main.css">
       <!-- 코드 하이라이트 -->
       <link rel="stylesheet"
             href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
     </head><body>
       <main class="article">
         <h1>${attr.title}</h1>
         <div class="meta">${formattedDate} · ${attr.readingTime} min read</div>

         <img class="hero" src="../${attr.thumbnail}" alt="cover image">

         ${htmlBody}
       </main>

       <!-- Highlight.js -->
       <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
       <script>hljs.highlightAll();</script>

       <!-- MathJax (자동 줄바꿈) -->
       <script>
         window.MathJax = {
           tex: { inlineMath: [['$', '$'], ['\\\\(', '\\\\)']] },
           options: { linebreaks: { automatic: true, width: 'container' } }
         };
       </script>
       <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
     </body></html>`
  );

  /* 메인 카드 누적 */
  cardsHTML += `
    <a class="card" href="posts/${outName}">
      <img src="${attr.thumbnail}" class="thumb" alt="">
      <div class="card-body">
        <h2 class="card-title">${attr.title}</h2>
        <div class="meta">${formattedDate} · ${attr.readingTime} min read</div>
      </div>
    </a>`;
}

/* ─ 2. index.html 템플릿에 카드 삽입 ─────────────────────────── */
const indexSrc  = readFileSync(join(SRC, 'index.html'), 'utf-8');
const indexDist = indexSrc.replace('<!--AUTO-CARDS-->', cardsHTML);
writeFileSync(join(DIST, 'index.html'), indexDist);

/* ─ 3. 정적 자원 복사 ─────────────────────────────────────────── */
for (const dir of ASSETS) {
  await fse.copy(join(SRC, dir), join(DIST, dir));
}

console.log('✅ Build complete → dist/');
